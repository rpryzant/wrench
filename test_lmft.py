import logging
import torch

import numpy as np

from wrench.dataset import load_dataset
from wrench.logging import LoggingHandler
from wrench.synthetic import ConditionalIndependentGenerator, NGramLFGenerator
from wrench.labelmodel import FlyingSquid

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
train_data, valid_data, test_data = load_dataset(dataset_home, data, extract_feature=False)

train_data = train_data.sample(alpha=0.1)


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

print(metric_value)

quit()



#### Generate semi-synthetic labeling functions
generator = NGramLFGenerator(dataset=train_data, min_acc_gain=0.1, min_support=0.01, ngram_range=(1, 2))
# mode=correlated doesn't work (scailability)
applier = generator.generate(mode='accurate', n_lfs=10)

L_test = applier.apply(test_data)
L_train = applier.apply(train_data)


#### Evaluate label model on real-world dataset with semi-synthetic labeling functions
label_model = FlyingSquid()

train_data.weak_labels = L_train.tolist()
test_data.weak_labels = L_test.tolist()

label_model.fit(dataset_train=train_data, dataset_valid=valid_data)
target_value = label_model.test(test_data, metric_fn='auc')
print(target_value)