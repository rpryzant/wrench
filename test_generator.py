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
from wrench.classification import Astra
import rules
from collections import Counter, defaultdict


#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)


#### Load dataset 
import sys
OUTF = 'out'# open(sys.argv[1], 'a')

dummy1 = sys.argv[1]
dummy2 = sys.argv[2]
dummy3 = sys.argv[3]
dummy4 = sys.argv[4]
dummy5 = sys.argv[5]


working_dir = 'test_wd'
device = torch.device('cuda')


def apply_snork(train, unlabeled, lm_threshold=0.99):
    new_train = copy.deepcopy(train)
    new_unlabeled = copy.deepcopy(unlabeled)
    label_model = Snorkel(lr=0.01, l2=0.0, n_epochs=500)
    label_model.fit(dataset_train=new_train)

    unlabeled_covered, unlabeled_uncovered = new_unlabeled.get_covered_subset()
    soft_labs = label_model.predict_proba(unlabeled_covered)
    transfer_idxs = []
    for i in range(len(unlabeled_covered)):
        if np.max(soft_labs[i]) > lm_threshold:
            pred_lab = np.argmax(soft_labs[i])
            unlabeled_covered.labels[i] = pred_lab
            transfer_idxs.append(i)
    new_labeled, still_unlabeled = unlabeled_covered.create_split(transfer_idxs)

    new_train = new_train.concat(new_labeled)
    new_unlabeled = unlabeled_uncovered.concat(still_unlabeled)

    return new_train, new_unlabeled


def get_train_unlabeled_valid_test(dataset_home, data, prop_labeled, rule_gen_params):
    train_dataset, valid_data, test_data = load_dataset(dataset_home, data, extract_feature=False)

    # some datasets have huge valid sets...speed up
    # valid_data, _ = valid_data.create_split(list(range(min(2000, len(valid_data)))))

    keep_idxs = random.sample(list(range(len(train_dataset))), int(len(train_dataset) * prop_labeled))

    train_dataset, unlabeled_dataset = train_dataset.create_split(idx=keep_idxs)
    unlabeled_dataset.labels = [-1 for _ in unlabeled_dataset.labels]

    train_texts = [ex['text'] for ex in train_dataset.examples]
    train_labs = train_dataset.labels

    applier = rules.AutoRuleGenerator(**rule_gen_params)
    applier.train(texts=train_texts, labels=train_labs)

    train_dataset.weak_labels = applier.apply(train_texts).tolist()
    train_dataset.n_lf = len(train_dataset.weak_labels[0])

    unlabeled_dataset.weak_labels = applier.apply([ex['text'] for ex in unlabeled_dataset.examples]).tolist()
    unlabeled_dataset.n_lf = len(unlabeled_dataset.weak_labels[0])

    valid_data.weak_labels = applier.apply([ex['text'] for ex in valid_data.examples]).tolist()
    valid_data.n_lf = len(valid_data.weak_labels[0])

    test_data.weak_labels = applier.apply([ex['text'] for ex in test_data.examples]).tolist()
    test_data.n_lf = len(test_data.weak_labels[0])

    return train_dataset, unlabeled_dataset, valid_data, test_data




def run_expt(dataset_home, data_name, lm_threshold, prop_labeled, teacher_inference, fit_params, rule_gen_params, seed):
    results = {}

    fit_params['seed'] = seed
    random.seed(seed)

    train_dataset, unlabeled_dataset, valid_data, test_data = get_train_unlabeled_valid_test(
        dataset_home, data_name, prop_labeled, rule_gen_params
    )
    print(len(train_dataset), len(unlabeled_dataset), len(valid_data), len(test_data))
    ### SUPERVISED
    model = BertClassifierModel(
        batch_size=32, real_batch_size=-1, test_batch_size=48, max_tokens=128)
    model.fit(dataset_train=train_dataset, dataset_valid=valid_data, device=device, evaluation_step=20, 
        **fit_params)

    results['sup_valid'] = valid_value = model.test(valid_data, 'f1_binary')
    results['sup_test'] = test_value = model.test(test_data, 'f1_binary')

    ### SNORK
    model = BertClassifierModel(
        batch_size=32, real_batch_size=-1, test_batch_size=48, max_tokens=128)
    snork_train, snork_unlabeled = apply_snork(train_dataset, unlabeled_dataset, lm_threshold=lm_threshold)
    model.fit(dataset_train=snork_train, dataset_valid=valid_data, device=device, evaluation_step=20, 
        **fit_params)

    results['snork_valid'] = valid_value = model.test(valid_data, 'f1_binary')
    results['snork_test'] = test_value = model.test(test_data, 'f1_binary')

    ### RAN
    model = Astra(
        batch_size=32, real_batch_size=-1, test_batch_size=48,
        teacher_inference=teacher_inference, outer_patience=5, rule_embed_size=100,
        backbone='BERT', backbone_model_name='bert-base-cased', optimizer='default')

    model.fit(dataset_train=(train_dataset, unlabeled_dataset), dataset_valid=valid_data, device=device, evaluation_step=20, 
        **fit_params)

    results['ran_valid'] = valid_value = model.test(valid_data, 'f1_binary')
    results['ran_test'] = test_value = model.test(test_data, 'f1_binary')

    return results


# dataset_home = '../datasets/'
dataset_home = '/mnt/exp/data'

for _ in range(10):

    lm_threshold = random.choice([0.8, 0.9, 0.95, 0.99])
    prop_labeled = random.choice([0.03, 0.1, 0.2, 0.5, 0.8, 0.9, 0.99])
    fit_params = {
        'n_steps': random.choice([50, 100, 150]),
        'optimizer_lr': random.choice([5e-5, 5e-6]),
        'n_iter': random.choice([10, 20, 30]),
        'stu_sup': random.choice([10]),
        'stu_unsup': random.choice([0, 2, 4, 8]),
        'tea_sup': random.choice([2, 4, 8, 10]),
        'tea_unsup': random.choice([0, 2, 4, 8]),
    }

    rule_gen_params = {
        'feature_type': random.choice(['ngram', 'neuron', 'merge']), 
        'num_features': 800, 
        'num_rules': random.choice([8, 16, 32]), 
        'reg_type': 'l1',
        'reg_strength': 1.0,
        'max_df': random.choice([0.7, 0.9, 0.95]),
        'min_df': random.choice([4, 8, 16, 32]),
        'use_stops': True
    }
    teacher_inference = random.choice([True, False])

    results = defaultdict(list)

    for data_name in ['sms', 'agnews', 'semeval']:

        if data_name in {'sms'}:
            fit_params['metric'] = 'f1_binary'
        elif data_name in {'agnews', 'semeval'}:
            fit_params['metric'] = 'f1_macro'

        for seed in [42, 1337, 420, 8, 23]:
        # seed = 1
            out = run_expt(dataset_home, data_name, lm_threshold, prop_labeled, teacher_inference, fit_params, 
                rule_gen_params, seed)
            for k, v in out.items():
                results[k].append(v)

        out_row = '\t'.join([str(x) for x in [
            data_name,
            lm_threshold,
            prop_labeled,
            rule_gen_params['feature_type'],
            rule_gen_params['num_rules'],
            rule_gen_params['max_df'],
            rule_gen_params['min_df'],
            fit_params['n_steps'],
            fit_params['optimizer_lr'],
            fit_params['n_iter'],
            fit_params['stu_sup'],
            fit_params['stu_unsup'],
            fit_params['tea_sup'],
            fit_params['tea_unsup'],
            np.mean(results['sup_valid']),
            np.std(results['sup_valid']),
            np.mean(results['sup_test']),        
            np.std(results['sup_test']),        
            np.mean(results['snork_valid']),
            np.std(results['snork_valid']),
            np.mean(results['snork_test']),        
            np.std(results['snork_test']),        
            np.mean(results['ran_valid']),
            np.std(results['ran_valid']),
            np.mean(results['ran_test']),  
            np.std(results['ran_test']),  
        ]])
        out_fp = os.path.join(os.getenv('AMLT_OUTPUT_DIR', 'TEST_OUT'), 'results.tsv')
        # out_fp = 'results.tsv'
        print('#' * 100)
        print('WRITING TO', out_fp)
        print(out_row)
        with open(out_fp, 'a') as f:
            f.write(out_row + '\n')


