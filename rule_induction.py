from numpy import ma
from numpy.core.getlimits import _register_type
from numpy.lib.function_base import cov
from transformers import BertTokenizer, BertForMaskedLM
from torch.nn import functional as F
import torch 
import json
import numpy as np
from tqdm import tqdm
import scipy.stats as stats
import heapq
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sklearn.metrics as metrics
from sklearn.linear_model._logistic import LogisticRegression
import scipy
import math
from joblib import delayed, Parallel
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import random
import os



def get_hiddens_for_corpus(corpus, tokenizer, model, batch_size=32):
    """
    returns: [data, num neurons]
    """
    print("HIDDENS FOR CORPUS")
    covariates = []
    labs = []

    def get_every_n(a, n=2):
        if len(a) % n == 0:
            rng = range(int(len(a) / n))
        else:
            rng = range((len(a) // n) + 1)

        for i in rng:
            yield a[n*i: n*(i+1)]

    batches = [x for x in get_every_n(corpus, n=batch_size)]
    for bi, texts in tqdm(list(enumerate(batches))):
        input = {
            k: v.cuda() for k, v in 
            tokenizer.batch_encode_plus(
                texts,
                return_tensors = "pt",
                padding='max_length',
                truncation=True,
                max_length=128).items()
        }

        with torch.no_grad():
            output = model(**input, output_hidden_states=True)

        for i in range(len(texts)):
            example_covariates = np.concatenate((
                output.hidden_states[0][i].squeeze().cpu().numpy().flatten(),
                output.hidden_states[4][i].squeeze().cpu().numpy().flatten(),
                output.hidden_states[8][i].squeeze().cpu().numpy().flatten(),
                output.hidden_states[-1][i].squeeze().cpu().numpy().flatten()
            ))
            covariates.append(example_covariates)

    return np.array(covariates)


def rules_from_covariates(covariates, labels, num_rules=4000, score_type='f1'):
    rules = {}
    label_types = set(labels)
    rules_per_level = int(num_rules / len(label_types) / 2)

    for class_idx in label_types:
        tmp_labels = [1 if x == class_idx else 0 for x in labels]

        label_prior = float(sum(tmp_labels)) / len(tmp_labels)

        # TODO(rpryzant -- loop overt score type)
        # from section 4 of https://www.aaai.org/Papers/ICML/2003/ICML03-060.pdf
        def f_score(y, yhat):
            recall = recall_score(y, yhat)
            prior = float(sum(y)) / len(y)
            f_score = (recall ** 2) / prior
            return f_score

        def get_acc(preds, i):
            return (accuracy_score(tmp_labels, preds), i)

        def get_f1(preds, i):
            return (f1_score(tmp_labels, preds), i)

        def get_fscore(preds, i):
            return (f_score(tmp_labels, preds), i)

        if score_type == 'f1':
            score_fn = get_fscore
        elif score_type == 'acc':
            score_fn = get_acc
        elif score_type == 'fs':
            score_fn = get_fscore
        else:
            raise Exception('unknown score type ', score_type)

        over_preds = np.array(covariates > 0, dtype=np.int)
        over_accs = Parallel(n_jobs=8)(delayed(get_f1)(over_preds[:, i], i) for i in tqdm(range(over_preds.shape[1])))
        best_over_neurons = sorted(over_accs, key=lambda x: x[0], reverse=True)[:rules_per_level]

        under_preds = np.array(covariates < 0, dtype=np.int)
        under_accs = Parallel(n_jobs=8)(delayed(get_f1)(under_preds[:, i], i) for i in tqdm(range(under_preds.shape[1])))
        best_under_neurons = sorted(under_accs, key=lambda x: x[0], reverse=True)[:rules_per_level]

        rules[class_idx] = [('+', neuron_idx) for _, neuron_idx in best_over_neurons] + [('-', neuron_idx) for _, neuron_idx in best_under_neurons] 

    return rules


def rules2onehot(activations, rules):
    # use rules to get one-hot data for regression
    corpus_counts = []
    for i in range(len(activations)):
        feats = []
        for cls_idx in rules:
            for sign, idx in rules[cls_idx]:
                if sign == '+' and activations[i, idx] > 0:
                    feats.append(1)
                elif sign == '-' and activations[i, idx] < 0:
                    feats.append(1)
                else:
                    feats.append(0)
        corpus_counts.append(feats)
    return corpus_counts


def weak_labels_from_features(X, labels,
        reg_type='l2', 
        reg_strength=0.5, 
        num_rules=100):

    model = LogisticRegression(
        penalty=reg_type, 
        C=reg_strength, 
        fit_intercept=False, 
        solver='liblinear')
    model.fit(X, labels)

    n_classes = model.coef_.shape[0]

    name_rulefn_score = []

    if n_classes == 1:
        for idx, weight in enumerate(model.coef_[0]):
            if weight == 0: 
                continue

            if weight > 0:
                name_rulefn_score.append( (idx, 1, weight) )
            else:
                name_rulefn_score.append( (idx, 0, weight) )

    else:
        for class_id in range(n_classes):
            for idx, weight in enumerate(model.coef_[class_id]):
                if weight <= 0:
                    continue

                name_rulefn_score.append( (idx, class_id, weight) )

    name_rulefn_score = sorted(name_rulefn_score, key=lambda x: abs(x[-1]), reverse=True)
    name_rulefn_score = name_rulefn_score[:num_rules]

    weak_labels = []
    for row in X:
        weak_labs = []
        for idx, label, _ in name_rulefn_score:
            if row[idx] == 0:
                weak_labs.append(-1)
            else:
                weak_labs.append(label)
        weak_labels.append(weak_labs)

    return np.array(weak_labels)


def neuron_rules(texts, labels, score_type='f1', num_features=4000):

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased', return_dict=True).cuda()

    hiddens = get_hiddens_for_corpus(texts, tokenizer, model)
    covariates = stats.zscore(hiddens, axis=1)

    rules = rules_from_covariates(
        covariates, 
        labels, 
        score_type=score_type,
        num_rules=num_features)  

    X = rules2onehot(covariates, rules)
    return X


def ngram_rules(
        texts, labels, 
        num_features=100, 
        ngram=(1, 1), 
        max_df=0.8):

    cv = CountVectorizer(
        tokenizer=lambda x: word_tokenize(x),
        ngram_range=ngram,
        max_features=num_features,
        lowercase=False,
        binary=True,
        max_df=0.8
    )
    cv.fit(texts)
    corpus_counts = cv.transform(texts)
    corpus_counts = corpus_counts.toarray()
    return corpus_counts

    # weak_labels = weak_labels_from_features(features, labels,
    #     reg_type=reg_type,
    #     reg_strength=reg_strength,
    #     num_rules=num_rules)


class AutoRuleGenerator:

    def __init__(self, dataset, feature_type, num_features=100, ngram=(1, 1), max_df=0.8, score_type='f1'):
        if feature_type == 'ngram':
            self.featurizer = self.build_ngram_featurizer(
                texts=[x['text'] for x in dataset.examples],
                labels=[x for x in dataset.labels],
                num_features=num_features,
                ngram=ngram,
                max_df=max_df
            )
        elif feature_type == 'neuron':
            self.featurizer = self.build_neuron_featurizer(
                texts=[x['text'] for x in dataset.examples],
                labels=[x for x in dataset.labels],
                num_features=num_features,
                score_type=score_type
            )
        elif feature_type == 'merge':
            ngram_feats = self.build_ngram_featurizer(
                texts=[x['text'] for x in dataset.examples],
                labels=[x for x in dataset.labels],
                num_features=num_features,
                ngram=ngram,
                max_df=max_df
            )            
            neuron_feats = self.build_neuron_featurizer(
                texts=[x['text'] for x in dataset.examples],
                labels=[x for x in dataset.labels],
                num_features=num_features,
                score_type=score_type
            )
            self.featurizer = self.build_joint_featurizer(ngram_feats, neuron_feats)


    def get_weak_labels(self, dataset, num_rules, reg_type='l2', reg_strength=0.5):
        texts = [x['text'] for x in dataset.examples]
        labels = [x for x in dataset.labels]
        return weak_labels_from_features(
            self.featurize(dataset),
            labels,
            num_rules=num_rules,
            reg_type=reg_type,
            reg_strength=reg_strength,
        )


    def featurize(self, dataset):
        return self.featurizer([x['text'] for x in dataset.examples])

    def build_joint_featurizer(self, featA, featB):
        
        def featurize(texts):
            featsA = featA(texts)
            featsB = featB(texts)
            feats = np.concatenate( (featsA, featsB), axis=1 )
            return feats

        return featurize
            

    def build_neuron_featurizer(self, texts, labels, num_features, score_type):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForMaskedLM.from_pretrained('bert-base-uncased', return_dict=True).cuda()
        hiddens = get_hiddens_for_corpus(texts, tokenizer, model)

        covariates = stats.zscore(hiddens, axis=1)
        rules = rules_from_covariates(
            covariates, 
            labels, 
            score_type=score_type,
            num_rules=num_features)  
        del tokenizer
        del model

        def featurize(texts):
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertForMaskedLM.from_pretrained('bert-base-uncased', return_dict=True).cuda()
            hiddens = get_hiddens_for_corpus(texts, tokenizer, model)
            covariates = stats.zscore(hiddens, axis=1)
            X = rules2onehot(covariates, rules)
            del tokenizer
            del model
            return np.array(X)

        return featurize


    def build_ngram_featurizer(self, texts, labels, num_features, ngram, max_df):
        cv = CountVectorizer(
            tokenizer=lambda x: word_tokenize(x),
            ngram_range=ngram,
            max_features=num_features,
            lowercase=False,
            binary=True,
            max_df=0.8
        )
        cv.fit(texts)

        def featurize(texts):
            corpus_counts = cv.transform(texts)
            corpus_counts = corpus_counts.toarray()
            return corpus_counts

        return featurize
