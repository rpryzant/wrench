"""
main entrypoint 

packages
    numba
    pytokenizations
    sentence_transformers
    seqeval
    flyingsquid
    numbskull
    pgmpy
    snorkel-metal
    tensorboardX
    optuna
"""
import math
import logging
import torch
import os
import numpy as np
from tqdm import tqdm
from wrench.dataset import load_dataset
from wrench.logging import LoggingHandler
from wrench.search import grid_search
from wrench.synthetic import ConditionalIndependentGenerator, NGramLFGenerator
from wrench.labelmodel import FlyingSquid
from wrench.labelmodel import Snorkel
import random
from wrench.endmodel import MLPModel, BertClassifierModel
from wrench.endmodel import EndClassifierModel
import copy
from wrench.classification import AutoRule, SelfTrain
import rules
from collections import Counter, defaultdict
import sklearn.metrics as metrics
import sys
from argparse import ArgumentParser


device = torch.device('cuda')


parser = ArgumentParser()
# default/required args
parser.add_argument('--data_root', type=str, default='../datasets/', help='Root directory containing datasets (/mnt/exp/data if cluster)')
parser.add_argument('--dataset', type=str, default='sms', help='Name of data dir to use.')
parser.add_argument('--expt_type', type=str, default='rules', help='Type of experiment.', choices=['bert', 'rules', 'self', 'pretrain', 'bert'])
parser.add_argument('--out', type=str, default='OUT', help='Path to output dir.')

# Optional modeling arguments.
# Defaults are None and will be overridden.
parser.add_argument('--rule_type', type=str, default=None, 
    help='Rule generation strategy.', choices=['ngram', 'ngram-pca', 'ngram-tree'])
parser.add_argument('--soft_labels', type=bool, default=None, 
    help='Whether to use soft or hard labels for distillation, etc.')
parser.add_argument('--distill_threshold', type=int, default=-1, 
    help='Whether to use a threshold on distillation (1) or not (-1, default). TODO make bool.')
parser.add_argument('--teacher_inference', type=bool, default=None, 
    help='Whether to force teacher inference.')
parser.add_argument('--n_rules', type=int, default=None, 
    help='Number of rules to generate.')
parser.add_argument('--tree_threshold', type=float, default=None, 
    help='Prediction threshold for decision tree rules.')
parser.add_argument('--valid_filter', type=float, default=None, 
    help='The fraction of examples beyond 1.0 will be filtered out. E.g. 1.2, n_rules=15 => build 18 rules then throw out bottom 3')
parser.add_argument('--semantic_filter', type=int, default=None, 
    help='Cosine similarity threshold for filtering out rule activations (0 for off).')
parser.add_argument('--train_filter', type=int, default=None, 
    help='Percent of mistakes on training set to throw out (0 for off).')





parser.add_argument('--overrides', type=str, help='Optional overrides. TODO REFACTOR -- HACKY.')

ARGS = parser.parse_args()


def get_dataset_params(dataset, expt_type='best'):

    if dataset == 'agnews':
        return 16, 1.2, 80, 50, 'ngram', True, False, 0.8

    elif dataset == 'cdr':
        return 16, 1, 80, 0, 'ngram-pca', True, False, 0.8

    elif dataset == 'chemprot':
        return 16, 1, 80, 50, 'ngram-tree', True, True, 0.8

    elif dataset == 'imdb':
        return 32, 1, 80, 50, 'ngram-tree', True, False, 0.8

    elif dataset == 'scicite':
        return 32, 1.2, 80, 0, 'ngram-pca', True, False, 0.95

    elif dataset == 'semeval':
        return 32, 1, 80, 50, 'ngram', False, True, 0.8

    elif dataset == 'sms':
        return 16, 1.2, 80, 50, 'ngram-pca', False, True, 0.8

    elif dataset == 'trec':
        return 32, 1.2, 80, 50, 'ngram-pca', False, False, 0.8

    elif dataset == 'youtube':
        return 16, 1, 80, 50, 'ngram', True, False, 0.8


def coverage(weak_labs, k=1):
    n = 0
    for wl in weak_labs:
        x = [y for y in wl if y != -1]
        if len(x) >= k:
            n += 1
    return float(n) / len(weak_labs)


def pre_f1(weak_labs, labs):
    yhat, y = [], []
    for wl, l in zip(weak_labs, labs):
        x = [y for y in wl if y != -1]
        if len(x) == 0:
            continue
        pred = Counter(x).most_common(1)[0][0]
        yhat.append(pred)
        y.append(l)

    return metrics.precision_score(y, yhat, average='macro'), metrics.f1_score(y, yhat, average='macro')


def get_train_unlabeled_valid_test(dataset_home, data, prop_labeled, rule_gen_params):
    train_dataset, valid_data, test_data = load_dataset(dataset_home, data, extract_feature=False)

    # subsample large avlid sets tso that we can finish the jobs in a timely basis
    if len(valid_data) > 2000:
        valid_data = valid_data.sample(1500)

    full_train = copy.deepcopy(train_dataset)

    keep_idxs = random.sample(list(range(len(train_dataset))), int(len(train_dataset) * prop_labeled))

    train_dataset, unlabeled_dataset = train_dataset.create_split(idx=keep_idxs)
    unlabeled_dataset.labels = [-1 for _ in unlabeled_dataset.labels]

    train_texts = [ex['text'] for ex in train_dataset.examples]
    train_labs = train_dataset.labels

    applier = rules.AutoRuleGenerator(**rule_gen_params)
    applier.train(texts=train_texts, labels=train_labs, 
        unlabeled_texts=[ex['text'] for ex in unlabeled_dataset.examples],
        valid_text=[ex['text'] for ex in valid_data.examples],
        valid_labs=valid_data.labels)

    out_fp = os.path.join(os.getenv('AMLT_OUTPUT_DIR', ARGS.out), 'rules.txt')
    print('WRITING RULES')
    if os.path.exists(out_fp):
        os.remove(out_fp)
    out_str = '\n'.join([f'{i}\n{s}' for i, s in enumerate(applier.report_rules())])
    # print(out_str)
    with open(out_fp, 'w') as f:
        f.write(out_str)

    train_dataset.weak_labels = applier.apply(train_texts, ignore_semantic_filter=True,
        labels=train_labs, filter_train_disagree=rule_gen_params['filter_train_disagree']).tolist()
    train_dataset.n_lf = len(train_dataset.weak_labels[0])

    unlabeled_dataset.weak_labels = applier.apply([ex['text'] for ex in unlabeled_dataset.examples]).tolist()
    unlabeled_dataset.n_lf = len(unlabeled_dataset.weak_labels[0])

    valid_data.weak_labels = applier.apply([ex['text'] for ex in valid_data.examples]).tolist()
    valid_data.n_lf = len(valid_data.weak_labels[0])

    test_data.weak_labels = applier.apply([ex['text'] for ex in test_data.examples]).tolist()
    test_data.n_lf = len(test_data.weak_labels[0])

    print("RULE QUALITY:")
    print(f"\tTest coverage: {coverage(test_data.weak_labels, k=1)}")
    print(f"\tPrecision, F1: {pre_f1(test_data.weak_labels, test_data.labels)}")

    return train_dataset, unlabeled_dataset, valid_data, test_data, full_train




def run_expt(dataset_home, data_name, unsup_prop, lm_threshold, prop_labeled, teacher_inference, fit_params, rule_gen_params, expt_type, seed):
    results = {}

    fit_params['seed'] = seed
    random.seed(seed)

    train_dataset, unlabeled_dataset, valid_data, test_data, full_train = get_train_unlabeled_valid_test(
        dataset_home, data_name, prop_labeled, rule_gen_params
    )
    results['n_train'] = len(train_dataset)

    print('#' * 25)
    print('Preparing experiment')
    print(f"\t'# train: {len(train_dataset)}")
    print(f"\t'# unlabeled: {len(unlabeled_dataset)}")
    print(f"\t'# valid: {len(valid_data)}")
    print(f"\t'# test: {len(test_data)}")

    ### BERT
    if expt_type == 'bert':
        sup_fit_params = copy.copy(fit_params)
        sup_fit_params['n_steps'] *= 25 # match self-training steps

        model = BertClassifierModel(
            model_name='bert-base-cased',
            batch_size=24, real_batch_size=-1, test_batch_size=48, max_tokens=128)
        model.fit(dataset_train=train_dataset, dataset_valid=valid_data, device=device, evaluation_step=100, 
            **sup_fit_params)
        results['valid'], _ = model.test(valid_data, sup_fit_params['metric'])
        results['test'], preds = model.test(test_data, sup_fit_params['metric'])

    ### PRETRAIN ON UNLABELED
    elif expt_type == 'pretrain':
        sup_fit_params = copy.copy(fit_params)
        sup_fit_params['n_steps'] *= 25 # match self-training steps

        model = BertClassifierModel(
            model_name='bert-base-cased',
            batch_size=24, real_batch_size=-1, test_batch_size=48, max_tokens=128)
        model.pretrain(unlabeled_dataset, int(sup_fit_params['n_steps'] / 10), './tmp', device=device)
        model.fit(dataset_train=train_dataset, dataset_valid=valid_data, device=device, evaluation_step=100, 
            **sup_fit_params)
        results['valid'], _ = model.test(valid_data, sup_fit_params['metric'])
        results['test'], preds = model.test(test_data, sup_fit_params['metric'])

    ### SELF-TRAINING
    elif expt_type == 'self':
        model = SelfTrain(
            batch_size=24, real_batch_size=-1, test_batch_size=48, unsup_prop=unsup_prop,
            teacher_inference=teacher_inference, outer_patience=5, rule_embed_size=128,
            backbone='BERT', backbone_model_name='bert-base-cased', optimizer='default')
        model.fit(dataset_train=(train_dataset, unlabeled_dataset), dataset_valid=valid_data, device=device, evaluation_step=100, 
            **fit_params)
        results['valid'], _ = valid_value = model.test(valid_data, fit_params['metric'])
        results['test'], preds = test_value = model.test(test_data, fit_params['metric'])

    ### AUTO RULES + SELF-TRAIN
    elif expt_type == 'rules':
        model = AutoRule(
            batch_size=24, real_batch_size=-1, test_batch_size=48, unsup_prop=unsup_prop,
            teacher_inference=teacher_inference, outer_patience=5, rule_embed_size=128,
            backbone='BERT', backbone_model_name='bert-base-cased', optimizer='default')
        model.fit(dataset_train=(train_dataset, unlabeled_dataset), dataset_valid=valid_data, device=device, evaluation_step=100, 
            **fit_params)
        results['valid'], _ = valid_value = model.test(valid_data, fit_params['metric'])
        results['test'], preds = test_value = model.test(test_data, fit_params['metric'], write_attns=False)

    return results, preds


dataset_home = ARGS.data_root
dataset = ARGS.dataset
expt_type = ARGS.expt_type
distill_threshold = ARGS.distill_threshold

# load in per-dataset default args and overrides
(n_rules, stepwise_if, semantic_filter_threshold, filter_train_disagree, 
    feature_type, soft_labels, teacher_inference, pca_tree_threshold, 
    ) = get_dataset_params(dataset)
if ARGS.rule_type is not None:
    feature_type = ARGS.rule_type
if ARGS.soft_labels is not None:
    soft_labels = ARGS.soft_labels
if ARGS.teacher_inference is not None:
    teacher_inference = ARGS.teacher_inference
if ARGS.n_rules is not None:
    n_rules = ARGS.n_rules
if ARGS.valid_filter is not None:
    stepwise_if = ARGS.valid_filter
if ARGS.semantic_filter is not None:
    semantic_filter_threshold = ARGS.semantic_filter
if ARGS.train_filter is not None:
    filter_train_disagree = ARGS.train_filter
if ARGS.tree_threshold is not None:
    pca_tree_threshold = ARGS.tree_threshold

unsup_prop = 0.7
lm_threshold = 0.99
prop_labeled = 0.05
fit_params = {
    'n_steps': 500,
    'optimizer_lr': 5e-5,
    'n_iter': 25,
    'all_teacher': False,
    'soft_labels': soft_labels,
    'distill_threshold': distill_threshold,
    'use_unif': True
}
rule_gen_params = {
    'feature_type': feature_type,
    'num_features': 1600, 
    'num_rules': n_rules,
    'reg_type': 'l2',
    'reg_strength': 1.0,
    'max_df': 0.95,
    'min_df': 4,
    'use_stops': True,
    'dataset': dataset,
    'stepwise_inflation_factor': stepwise_if,
    'semantic_filter_threshold': semantic_filter_threshold,
    'filter_train_disagree': filter_train_disagree,
    'pca_tree_threshold': pca_tree_threshold
}

results = defaultdict(list)

final_results = {}

if dataset in {'cdr', 'youtube', 'sms', 'imdb'}:
    fit_params['metric'] = 'f1_binary'
else:
    fit_params['metric'] = 'f1_macro'

for seed in [42, 1337, 420, 8, 23]:
    out, preds = run_expt(dataset_home, dataset,
        unsup_prop, lm_threshold, prop_labeled, teacher_inference, fit_params, 
        rule_gen_params, expt_type, seed)
    for k, v in out.items():
        if math.isnan(v):
            continue
        results[k].append(v)

for k, v in results.items():
    final_results[k] = round(np.mean(v) * 100, 2)

out_row = '\t'.join([str(x) for x in [
    dataset,
    expt_type,
    final_results['valid'],
    final_results['test'],
]])
out_fp = os.path.join(os.getenv('AMLT_OUTPUT_DIR', ARGS.out), 'results.tsv')

print('#' * 100)
print('WRITING TO', out_fp)
print(out_row)
with open(out_fp, 'a') as f:
    f.write(out_row + '\n')

print('WRITING PREDS')
preds_fp =  os.path.join(os.getenv('AMLT_OUTPUT_DIR', ARGS.out), 'preds.tsv')
with open(preds_fp, 'a') as f:
    for row in preds:
        f.write('\t'.join([str(x) for x in row]) + '\n')
