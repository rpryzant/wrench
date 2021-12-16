import logging
import torch

import numpy as np
from tqdm import tqdm
from wrench.dataset import load_dataset
from wrench.logging import LoggingHandler
from wrench.synthetic import ConditionalIndependentGenerator, NGramLFGenerator
from wrench.labelmodel import FlyingSquid
from wrench.labelmodel import Snorkel
import random
from wrench.endmodel import MLPModel, BertClassifierModel

from wrench.classification import Astra

import rule_induction
from collections import Counter


#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)


#### Load dataset 
dataset_home = 'datasets/datasets/'
data = 'sms'

#### Load real-world dataset
train_dataset, valid_data, test_data = load_dataset(dataset_home, data, extract_feature=False)
import sys
OUTF = open(sys.argv[1], 'a')

for _ in tqdm(range(100)):


    induction_algo = random.choice(['mine', 'wrench'])

    induction_algo = 'mine'
        
    if induction_algo == 'mine':
        feat_type = 'neuron' # random.choice(['ngram', 'neuron', 'merge'])
        num_feats = 800 # random.choice([100, 400, 500, 1000, 2000, 4000])
        max_df = 0.8 # random.choice([0.5, 0.8, 0.9])
        score_type = 'f1' # random.choice(['f1', 'acc'])
        num_rules = 60 # random.choice([32, 64, 128])
        reg_type = 'l2' # random.choice(['l2', 'l1'])
        reg_strength = 0.5 # random.choice([0.1, 0.5, 1.0])
        out = '\t'.join([
            'mine', 
            feat_type,
            str(num_feats),
            str(max_df),
            str(score_type),
            str(num_rules),
            str(reg_type),
            str(reg_strength)
        ])

    elif induction_algo == 'wrench':
        min_acc_gain = random.choice([0.05, 0.01, 0.001])
        min_support = random.choice([0.01, 0.001])
        mode = random.choice(['accurate', 'correlated', 'cluster_dependent'])
        num_rules = random.choice([32, 64, 128])
        out = '\t'.join([
            'wrench', 
            mode,
            str(min_acc_gain),
            str(min_support),
            str(num_rules)
        ])
    # mode = 'accurate'

    metrics = []
    print(out)
    for seed in [1, 43, 657, 34, 23]:
        train_data = train_dataset.sample(0.001)
        train_data.weak_labels = None

        if induction_algo == 'mine':
            rg = rule_induction.AutoRuleGenerator(train_data, 
                feature_type=feat_type,
                num_features=num_feats,
                max_df=max_df,
                score_type=score_type)
            train_data.weak_labels = rg.get_weak_labels(train_data, reg_type=reg_type, reg_strength=reg_strength, num_rules=num_rules)
            valid_data.weak_labels = rg.get_weak_labels(valid_data, reg_type=reg_type, reg_strength=reg_strength, num_rules=num_rules)
            test_data.weak_labels = rg.get_weak_labels(test_data, reg_type=reg_type, reg_strength=reg_strength, num_rules=num_rules)
        else:
            generator = NGramLFGenerator(dataset=train_data, min_acc_gain=min_acc_gain, min_support=min_support, ngram_range=(1, 1))
            applier = generator.generate(mode=mode, n_lfs=num_rules)
            train_data.weak_labels = applier.apply(train_data)
            valid_data.weak_labels = applier.apply(valid_data)
            test_data.weak_labels = applier.apply(test_data)

        label_model = Snorkel(
            lr=0.01,
            l2=0.0,
            n_epochs=10
        )
        label_model.fit(
            dataset_train=train_data,
            dataset_valid=valid_data
        )
        acc = label_model.test(test_data, 'f1_macro')
        metrics.append(acc)
        logger.info(f'label model test f1: {acc}')

    logger.info(f'MEAN: {np.mean(metrics)}')
    out += '\t' + str(np.mean(metrics))  + '\n'
    print('WRITING...')
    print(out)
    OUTF.write(out)

outf.close()

quit()


print(Counter(feats.flatten().tolist()))
print(feats)
print(feats.shape)
quit()


# L_train = rule_induction.auto_rules(
#     train_data, 
#     feature_type='neuron',
#     num_rules=12)

working_dir = './test_sandbox/'

device = torch.device('cuda:0')
n_steps = 100
batch_size = 32
test_batch_size = 10
patience = 200
evaluation_step = 50
target='acc'

model = BertClassifierModel(n_steps=n_steps, batch_size=batch_size, test_batch_size=test_batch_size)

model.pretrain(train_data, epochs=3, output_dir=working_dir, device=device)

history = model.fit(dataset_train=train_data, dataset_valid=valid_data, device=device, metric=target, 
                    patience=patience, evaluation_step=evaluation_step)

#### Evaluate the trained model
metric_value = model.test(test_data, target)



quit()



#### Generate semi-synthetic labeling functions
L_train = applier.apply(train_data)


#### Evaluate label model on real-world dataset with semi-synthetic labeling functions
label_model = FlyingSquid()

train_data.weak_labels = L_train.tolist()
test_data.weak_labels = L_test.tolist()

label_model.fit(dataset_train=train_data, dataset_valid=valid_data)
target_value = label_model.test(test_data, metric_fn='auc')
print(target_value)