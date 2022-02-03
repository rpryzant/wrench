"""
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
from wrench.classification import Astra, SelfTrain
import rules
from collections import Counter, defaultdict
import sklearn.metrics as metrics
import utils

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)


#### Load dataset 
import sys
OUTF = 'out'# open(sys.argv[1], 'a')


working_dir = 'test_wd'
device = torch.device('cuda')


def apply_snork(train, unlabeled, lm_threshold=0.99):
    new_train = copy.deepcopy(train)
    new_unlabeled = copy.deepcopy(unlabeled)
    label_model = Snorkel(lr=0.01, l2=0.0, n_epochs=500)
    label_model.fit(dataset_train=new_train)

    # unlabeled_covered, unlabeled_uncovered = new_unlabeled.get_covered_subset()
    soft_labs = label_model.predict_proba(new_unlabeled)
    transfer_idxs = []
    for i in range(len(new_unlabeled)):
        if np.max(soft_labs[i]) > lm_threshold:
            pred_lab = np.argmax(soft_labs[i])
            new_unlabeled.labels[i] = pred_lab
            transfer_idxs.append(i)
    new_labeled, still_unlabeled = new_unlabeled.create_split(transfer_idxs)

    new_train = new_train.concat(new_labeled)
    # new_unlabeled = unlabeled_uncovered.concat(still_unlabeled)

    return new_train, still_unlabeled


def get_train_unlabeled_valid_test(dataset_home, data, prop_labeled, rule_gen_params):
    train_dataset, valid_data, test_data = load_dataset(dataset_home, data, extract_feature=False)

    # subsample large avlid sets tso that we can finish the jobs
    if len(valid_data) > 2000:
        valid_data = valid_data.sample(1500)

    full_train = copy.deepcopy(train_dataset)

    keep_idxs = random.sample(list(range(len(train_dataset))), int(len(train_dataset) * prop_labeled))

    train_dataset, unlabeled_dataset = train_dataset.create_split(idx=keep_idxs)
    unlabeled_dataset.labels = [-1 for _ in unlabeled_dataset.labels]

    train_texts = [ex['text'] for ex in train_dataset.examples]
    train_labs = train_dataset.labels

    # print(utils.coverage(test_data.weak_labels, k=1))
    # print(utils.coverage(test_data.weak_labels, k=2))
    # print(utils.acc_pre(test_data.weak_labels, test_data.labels))
    # print(len(train_dataset), len(unlabeled_dataset), len(valid_data), len(test_data), len(set(test_data.labels)))
    # quit()

    applier = rules.AutoRuleGenerator(**rule_gen_params)
    applier.train(texts=train_texts, labels=train_labs, 
        unlabeled_texts=[ex['text'] for ex in unlabeled_dataset.examples],
        unlabeled_labels=unlabeled_dataset.labels,  # TODO remove labs b4 eval
        valid_text=[ex['text'] for ex in valid_data.examples],
        valid_labs=valid_data.labels)


    out_fp = os.path.join(os.getenv('AMLT_OUTPUT_DIR', 'TEST_OUT'), 'rules.txt')
    print('WRITING RULES')
    if os.path.exists(out_fp):
        os.remove(out_fp)
    out_str = '\n'.join([f'{i}\n{s}' for i, s in enumerate(applier.report_rules())])
    print(out_str)
    with open(out_fp, 'w') as f:
        f.write(out_str)

    # quit()


    train_dataset.weak_labels = applier.apply(train_texts, ignore_semantic_filter=True,
        labels=train_labs, filter_train_disagree=rule_gen_params['filter_train_disagree']).tolist()
    train_dataset.n_lf = len(train_dataset.weak_labels[0])

    unlabeled_dataset.weak_labels = applier.apply([ex['text'] for ex in unlabeled_dataset.examples]).tolist()
    unlabeled_dataset.n_lf = len(unlabeled_dataset.weak_labels[0])

    valid_data.weak_labels = applier.apply([ex['text'] for ex in valid_data.examples]).tolist()
    valid_data.n_lf = len(valid_data.weak_labels[0])

    test_data.weak_labels = applier.apply([ex['text'] for ex in test_data.examples]).tolist()
    test_data.n_lf = len(test_data.weak_labels[0])

    # print(test_data.weak_labels)
    # print(utils.acc_pre(train_dataset.weak_labels, train_dataset.labels))
    # print(utils.acc_pre(unlabeled_dataset.weak_labels, unlabeled_dataset.labels))

    print(utils.coverage(test_data.weak_labels, k=1))
    print(utils.coverage(test_data.weak_labels, k=2))
    print(utils.acc_pre(test_data.weak_labels, test_data.labels))
    # quit()

    return train_dataset, unlabeled_dataset, valid_data, test_data, full_train




def run_expt(dataset_home, data_name, unsup_prop, lm_threshold, prop_labeled, teacher_inference, fit_params, rule_gen_params, expt_type, seed):
    results = {}

    fit_params['seed'] = seed
    random.seed(seed)

    train_dataset, unlabeled_dataset, valid_data, test_data, full_train = get_train_unlabeled_valid_test(
        dataset_home, data_name, prop_labeled, rule_gen_params
    )
    results['n_train'] = len(train_dataset)

    print(len(train_dataset), len(unlabeled_dataset), len(valid_data), len(test_data))

    ### BERT
    if expt_type == 'bert':
        sup_fit_params = copy.copy(fit_params)
        sup_fit_params['n_steps'] *= 25 # bump up to match step count of astra

        model = BertClassifierModel(
            model_name='bert-base-cased',
            batch_size=24, real_batch_size=-1, test_batch_size=48, max_tokens=128)
        model.fit(dataset_train=train_dataset, dataset_valid=valid_data, device=device, evaluation_step=100, 
            **sup_fit_params)
        results['valid'], _ = model.test(valid_data, sup_fit_params['metric'])
        results['test'], preds = model.test(test_data, sup_fit_params['metric'])

    elif expt_type == 'pretrain':
        sup_fit_params = copy.copy(fit_params)
        sup_fit_params['n_steps'] *= 25 # bump up to match step count of astra

        model = BertClassifierModel(
            model_name='bert-base-cased',
            batch_size=24, real_batch_size=-1, test_batch_size=48, max_tokens=128)
        model.pretrain(unlabeled_dataset, int(sup_fit_params['n_steps'] / 10), './tmp', device=device)
        model.fit(dataset_train=train_dataset, dataset_valid=valid_data, device=device, evaluation_step=100, 
            **sup_fit_params)
        results['valid'], _ = model.test(valid_data, sup_fit_params['metric'])
        results['test'], preds = model.test(test_data, sup_fit_params['metric'])


    ## SELF-TRAINING
    elif expt_type == 'self':
        model = SelfTrain(
            batch_size=24, real_batch_size=-1, test_batch_size=48, unsup_prop=unsup_prop,
            teacher_inference=teacher_inference, outer_patience=5, rule_embed_size=128,
            backbone='BERT', backbone_model_name='bert-base-cased', optimizer='default')
        model.fit(dataset_train=(train_dataset, unlabeled_dataset), dataset_valid=valid_data, device=device, evaluation_step=100, 
            **fit_params)
        results['valid'], _ = valid_value = model.test(valid_data, fit_params['metric'])
        results['test'], preds = test_value = model.test(test_data, fit_params['metric'])


    ## ASTRA SEMISUP
    elif expt_type == 'ran':
        model = Astra(
            batch_size=24, real_batch_size=-1, test_batch_size=48, unsup_prop=unsup_prop,
            teacher_inference=teacher_inference, outer_patience=5, rule_embed_size=128,
            backbone='BERT', backbone_model_name='bert-base-cased', optimizer='default')
        model.fit(dataset_train=(train_dataset, unlabeled_dataset), dataset_valid=valid_data, device=device, evaluation_step=100, 
            **fit_params)
        results['valid'], _ = valid_value = model.test(valid_data, fit_params['metric'])
        results['test'], preds = test_value = model.test(test_data, fit_params['metric'], write_attns=True)


    elif expt_type == 'snork':
        snork_train, snork_unlabeled = apply_snork(train_dataset, unlabeled_dataset, lm_threshold=lm_threshold)
        sup_fit_params = copy.copy(fit_params)
        sup_fit_params['n_steps'] *= 25 # bump up to match step count of astra

        model = BertClassifierModel(
            model_name='bert-base-cased',
            batch_size=24, real_batch_size=-1, test_batch_size=48, max_tokens=128)
        model.fit(dataset_train=snork_train, dataset_valid=valid_data, device=device, evaluation_step=100, 
            **sup_fit_params)
        results['valid'], _ = model.test(valid_data, sup_fit_params['metric'])
        results['test'], preds = model.test(test_data, sup_fit_params['metric'])


    return results, preds


# dataset_home = '../datasets/'
dataset_home = '/mnt/exp/data'

# dataset = 'sms'
# n_rules = 10
# train_prop = 0.1
# stepwise_if = 1.0
# semantic_filter_threshold = 0
# filter_train_disagree = 0
# feature_type = 'ngram' # -pca'
# pca_tree_threshold = 0.9   # only useful for ngram-pca
# soft_labels = False
# teacher_inference = False
# pca_tree_threshold = 0.98 # confidence threshold for pca decision tree to fire rule
# distill_threshold = -1 #-1 for off. only bigger than this is allowed for training
# unsup_prop = 0.7
# expt_type = 'ran'
# lm_threshold = 0.99
# use_unif = False

# TODO GET RID OF UNIFORM WEIGHTING THING???

parseBool = lambda x:  True if x.lower == 'true'  else False

dataset =  sys.argv[1]
expt_type = sys.argv[2]
train_prop = 0.05
# train_prop = float(sys.argv[3])
# soft_labels = parseBool(sys.argv[4])
# distill_threshold = float(sys.argv[5])
distill_threshold = -1


import params
# TODO REMOVED THING IS SOFT_LABELS
(n_rules, stepwise_if, semantic_filter_threshold, filter_train_disagree, 
    feature_type, soft_labels, teacher_inference, pca_tree_threshold, 
    ) = params.get_params(dataset)

# feature_type = sys.argv[3]

# n_rules = 50
# teacher_inference = True

use_unif = True
# feature_type = 'ngram-pca'

# overrides
if len(sys.argv) > 3:
    overrides = sys.argv[3].split('_')

    if overrides[0] != 'X':
        stepwise_if = float(overrides[0])

    if overrides[1] != 'X':
        semantic_filter_threshold = int(overrides[1])

    if overrides[2] != 'X':
        filter_train_disagree = int(overrides[2])

    if overrides[3] != 'X':
        feature_type = overrides[3]

    if overrides[4] != 'X':
        soft_labels = parseBool(overrides[4])

    if overrides[5] != 'X':
        teacher_inference = parseBool(overrides[5])

    if overrides[6] != 'X':
        use_unif = parseBool(overrides[6])

    if overrides[7] != 'X':
        feature_type = overrides[7]

# feature_type ='ngram'
# n_rules = 100


unsup_prop = 0.7
# train_prop = 0.05
# distill_threshold = -1
# teacher_inference = True



# dataset ='sms' # sys.argv[1]
n_steps = 500
n_iter = 25
all_teacher = False
lm_threshold = 0.99
prop_labeled = train_prop
fit_params = {
    'n_steps': n_steps, # random.choice([50, 100, 150]),
    'optimizer_lr': 5e-5, #  random.choice([5e-5, 5e-6]),
    'n_iter': n_iter, # random.choice([10, 20, 30]),
    'all_teacher': all_teacher,
    'soft_labels': soft_labels,
    'distill_threshold': distill_threshold,
    'use_unif': use_unif
}
rule_gen_params = {
    'feature_type': feature_type, #random.choice(['ngram', 'neuron', 'merge']), 
    'num_features': 1600, 
    'num_rules': n_rules, # random.choice([8, 16, 32]), 
    'reg_type': 'l2',
    'reg_strength': 1.0,
    'max_df': 0.95, # random.choice([0.7, 0.9, 0.95]),
    'min_df': 4, # random.choice([4, 8, 16, 32]),
    'use_stops': True,
    'dataset': dataset,
    'stepwise_inflation_factor': stepwise_if,
    'semantic_filter_threshold': semantic_filter_threshold,
    'filter_train_disagree': filter_train_disagree,
    'pca_tree_threshold': pca_tree_threshold
}
# teacher_inference = random.choice([True, False])

results = defaultdict(list)


final_results = {}

if dataset in {'cdr', 'youtube', 'sms', 'imdb'}:
    fit_params['metric'] = 'f1_binary'
else:
    fit_params['metric'] = 'f1_macro'

# fit_params['metric'] = 'precision_macro'
# sdz = random.randint(0, 1000)
for seed in [42, 1337, 420, 8, 23]: # , 234, 94, 3, 124, 999]:
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
    n_rules,
    stepwise_if,
    semantic_filter_threshold,
    filter_train_disagree,
    feature_type,
    pca_tree_threshold,
    soft_labels,
    teacher_inference,
    use_unif,
    final_results['valid'],
    final_results['test'],
]])
out_fp = os.path.join(os.getenv('AMLT_OUTPUT_DIR', 'TEST_OUT'), 'results.tsv')
# if os.path.exists(out_fp):
#     os.remove(out_fp)

# out_fp = 'results.tsv'
print('#' * 100)
print('WRITING TO', out_fp)
print(out_row)
with open(out_fp, 'a') as f:
    f.write(out_row + '\n')

print('WRITING PREDS')
preds_fp =  os.path.join(os.getenv('AMLT_OUTPUT_DIR', 'TEST_OUT'), 'preds.tsv')
# if os.path.exists(preds_fp):
#     os.remove(preds_fp)

with open(preds_fp, 'a') as f:
    for row in preds:
        f.write('\t'.join([str(x) for x in row]) + '\n')
