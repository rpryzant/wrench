import logging
import torch

from wrench.dataset import load_dataset
from wrench.logging import LoggingHandler
from wrench.synthetic import ConditionalIndependentGenerator, NGramLFGenerator
from wrench.labelmodel import FlyingSquid

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)


#### Load dataset 
dataset_home = 'datasets'
data = 'sms'

#### Load real-world dataset
train_data, valid_data, test_data = load_dataset(dataset_home, data, extract_feature=False)

#### Generate semi-synthetic labeling functions
generator = NGramLFGenerator(dataset=train_data, min_acc_gain=0.1, min_support=0.01, ngram_range=(1, 2))
applier = generator.generate(mode='correlated', n_lfs=10)
L_test = applier.apply(test_data)
L_train = applier.apply(train_data)


#### Evaluate label model on real-world dataset with semi-synthetic labeling functions
label_model = FlyingSquid()
label_model.fit(dataset_train=L_train, dataset_valid=valid_data)
target_value = label_model.test(L_test, metric_fn='auc')
