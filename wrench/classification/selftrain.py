import copy
import logging
from typing import Any, Optional, Union, Callable, List
from transformers import BertForSequenceClassification, BartForSequenceClassification
from ..utils import cross_entropy_with_probs, get_bert_model_class, get_bert_torch_dataset_class, construct_collate_fn_trunc_pad

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import trange
from transformers import AutoTokenizer

from ..backbone import BackBone
from ..basemodel import BaseTorchClassModel
from ..config import Config
from ..dataset import BaseDataset
from ..utils import cross_entropy_with_probs

collate_fn = construct_collate_fn_trunc_pad('mask')


logger = logging.getLogger(__name__)

ABSTAIN = -1



def update_state_dict(model, state_dict: dict, mode: str):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(mode):
            new_state_dict[k[len(mode) + 1:]] = v
    getattr(model, mode).load_state_dict(new_state_dict)




class SelfTrain(BaseTorchClassModel):
    def __init__(self,
                 n_iter: Optional[int] = 25,
                 outer_patience: Optional[int] = 3,
                 rule_embed_size: Optional[int] = 100,
                 dropout: Optional[float] = 0.3,
                 unsup_prop=1.0,

                 batch_size: Optional[int] = 16,
                 real_batch_size: Optional[int] = 16,
                 test_batch_size: Optional[int] = 16,
                 n_steps: Optional[int] = 10000,
                 grad_norm: Optional[float] = -1,
                 use_lr_scheduler: Optional[bool] = False,
                 binary_mode: Optional[bool] = False,
                 teacher_inference = False,
                 **kwargs: Any
                 ):
        super().__init__()
        self.hyperparas = {
            'n_iter'          : n_iter,
            'outer_patience'  : outer_patience,
            'rule_embed_size' : rule_embed_size,
            'dropout'         : dropout,

            'batch_size'      : batch_size,
            'real_batch_size' : real_batch_size,
            'test_batch_size' : test_batch_size,
            'n_steps'         : n_steps,
            'grad_norm'       : grad_norm,
            'use_lr_scheduler': use_lr_scheduler,
            'binary_mode'     : binary_mode,
        }
        self.model: Optional[SelfTrain] = None
        self.config = Config(
            self.hyperparas,
            use_optimizer=True,
            use_lr_scheduler=use_lr_scheduler,
            use_backbone=True,
            use_label_model=False,
            **kwargs
        )
        self.teacher_inference = teacher_inference
        self.is_bert = self.config.backbone_config['name'] == 'BERT'
        if self.is_bert:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.backbone_config['paras']['model_name'])
        self.unsup_prop=unsup_prop

    def fit(self,
            dataset_train: None,
            dataset_valid: Optional[BaseDataset] = None,
            y_valid: Optional[np.ndarray] = None,
            pretrained_model: str = None,
            valid_mode: Optional[str] = 'student',
            soft_labels: Optional[bool] = False,
            evaluation_step: Optional[int] = 100,
            metric: Optional[Union[str, Callable]] = 'acc',
            direction: Optional[str] = 'auto',
            patience: Optional[int] = 20,
            tolerance: Optional[float] = -1.0,
            device: Optional[torch.device] = None,
            verbose: Optional[bool] = True,
            stu_sup = None,
            stu_unsup = None,
            tea_sup = None,
            tea_unsup = None,
            all_teacher = False,
            distill_threshold = -1,
            **kwargs: Any):

        if not verbose:
            logger.setLevel(logging.ERROR)

        config = self.config.update(**kwargs)
        hyperparas = self.config.hyperparas
        config.backbone_config['paras']['max_tokens'] = 256  # TODO hack to get this working
        logger.info(config)

        dataset_train, dataset_unlabeled = dataset_train

        n_class = dataset_train.n_class
        # labeled_dataset, unlabeled_dataset = dataset_train.create_split(labeled_data_idx)
        labeled_dataset = dataset_train
        unlabeled_dataset = dataset_unlabeled

        n_steps = hyperparas['n_steps']

        if hyperparas['real_batch_size'] == -1 or hyperparas['batch_size'] < hyperparas['real_batch_size'] or not self.is_bert:
            hyperparas['real_batch_size'] = hyperparas['batch_size']
        accum_steps = hyperparas['batch_size'] // hyperparas['real_batch_size']

        assert config.backbone_config['name'] != 'LogReg'
        backbone = self._init_model(
            dataset=dataset_train,
            n_class=dataset_train.n_class,
            config=config,
            is_bert=self.is_bert
        )
        
        model = backbone # BertForSequenceClassification.from_pretrained(
                    # 'bert-base-cased', num_labels=n_class)
                    
        self.model = model.to(device)

        self.teacher_model = copy.deepcopy(self.model)

        # labeled_dataloader = self._init_train_dataloader(
        #     labeled_dataset,
        #     n_steps=n_steps,
        #     config=config,
        #     return_weak_labels=True,
        #     return_labels=True,
        # )

        # unlabeled_dataloader = self._init_train_dataloader(
        #     unlabeled_dataset,
        #     n_steps=n_steps,
        #     config=config,
        #     return_weak_labels=True,
        # )

        torch_dataset = get_bert_torch_dataset_class(dataset_train)(dataset_train, self.tokenizer, 128,
                                                                    n_data=n_steps * hyperparas['batch_size'])

        labeled_dataloader = DataLoader(torch_dataset, batch_size=hyperparas['real_batch_size'], shuffle=True, collate_fn=collate_fn)
        y_train = torch.Tensor(dataset_train.labels).to(device)


        torch_dataset = get_bert_torch_dataset_class(unlabeled_dataset)(unlabeled_dataset, self.tokenizer, 128,
                                                                    n_data=n_steps * hyperparas['batch_size'])
        unlabeled_dataloader = DataLoader(torch_dataset, batch_size=hyperparas['real_batch_size'], shuffle=True, collate_fn=collate_fn)


        valid_flag = self._init_valid_step(
            dataset_valid,
            y_valid,
            metric,
            direction,
            patience,
            tolerance,
            return_weak_labels=True,
        )

        STEPS = 0

        history = {}

        history_train = {}
        last_step_log = {}
        n_iter = hyperparas['n_iter']

        if valid_flag:
            outer_patience = hyperparas['outer_patience']
            outer_no_improve_cnt = 0
            outer_best_model = None
            if self.direction == 'maximize':
                outer_best_metric_value = -np.inf
            else:
                outer_best_metric_value = np.inf
        for i in range(n_iter):
            if valid_flag:
                self._reset_valid()
                self._valid_step(-1, mode='teacher')

            # pseudo_probas_u, features_u = self.collect_pseudodataset_from_teacher(unlabeled_dataset)
            # pseudo_probas_l, features_l = self.collect_pseudodataset_from_teacher(labeled_dataset)

            optimizer, scheduler = self._init_optimizer_and_lr_scheduler(self.model, config)

            history_finetune_student = {}
            with trange(n_steps, desc=f"[FINETUNE@{i}] ASTRA-student-sup", unit="steps", disable=not verbose, ncols=150, position=0, leave=True) as pbar:
                cnt = 0
                step = 0
                model.train()
                optimizer.zero_grad()
                for label_batch in labeled_dataloader:
                    # if step_balance['student-supervised'] == 0: break                
                    predict_l = model(label_batch)
                    target = y_train[label_batch['ids'].to(device)]
                    loss = cross_entropy_with_probs(predict_l, target)
                    loss.backward()
                    cnt += 1
                    if cnt % accum_steps == 0:
                        if hyperparas['grad_norm'] > 0:
                            nn.utils.clip_grad_norm_(model.parameters(), hyperparas['grad_norm'])
                        optimizer.step()
                        if scheduler is not None:
                            scheduler.step()
                        optimizer.zero_grad()
                        step += 1
                        STEPS += 1
                        if valid_flag and STEPS % evaluation_step == 0:
                            STEPS = 0
                            metric_value, early_stop_flag, info = self._valid_step(step, mode='student')
                            if early_stop_flag:
                                logger.info(info)
                                break

                            history_finetune_student[step] = {
                                'loss'              : loss.item(),
                                f'val_{metric}'     : metric_value,
                                f'best_val_{metric}': self.best_metric_value,
                                'best_step'         : self.best_step,
                            }
                            last_step_log.update(history_finetune_student[step])

                        last_step_log['loss'] = loss.item()
                        pbar.update()
                        pbar.set_postfix(ordered_dict=last_step_log)

                        # if step >= step_balance['student-supervised']:
                        #     break
            if valid_flag:
                update_state_dict(self.model, self.best_model, 'model')
                self._reset_valid()
                self._valid_step(-1, mode='student')

            optimizer, scheduler = self._init_optimizer_and_lr_scheduler(self.model, config)

            # load student into teacher for next round
            self.teacher_model.load_state_dict(self.model.state_dict())

            pseudo_probas_u = self.collect_pseudodataset_teacher(unlabeled_dataset)

            if distill_threshold > 0:
                loss_weights = (torch.max(torch.nn.functional.softmax(pseudo_probas_u, dim=-1), dim=-1).values > distill_threshold).float()
            else:
                loss_weights = torch.tensor([1.] * len(pseudo_probas_u), dtype=float).to(device)

            if not soft_labels:
                pseudo_probas_u = torch.argmax(pseudo_probas_u, dim=-1)

            history_train_student = {}
            with trange(int(n_steps * self.unsup_prop), desc=f"[TRAIN@{i}] ASTRA-student-unsup", unit="steps", disable=not verbose, ncols=150, position=0, leave=True) as pbar:
                cnt = 0
                step = 0
                model.train()
                optimizer.zero_grad()
                for unlabeled_batch in unlabeled_dataloader:
                    # if step_balance['student-unsupervised'] == 0: break                                
                    idx_u = unlabeled_batch['ids'].long().to(device)
                    predict_u = model(unlabeled_batch)
                    loss = cross_entropy_with_probs(predict_u, pseudo_probas_u[idx_u],
                        tok_weight=loss_weights[idx_u])

                    loss.backward()
                    cnt += 1
                    if cnt % accum_steps == 0:
                        if hyperparas['grad_norm'] > 0:
                            nn.utils.clip_grad_norm_(model.parameters(), hyperparas['grad_norm'])
                        optimizer.step()
                        if scheduler is not None:
                            scheduler.step()
                        optimizer.zero_grad()
                        step += 1
                        STEPS += 1
                        if valid_flag and STEPS % evaluation_step == 0:
                            STEPS = 0
                            metric_value, early_stop_flag, info = self._valid_step(step, mode='student')
                            if early_stop_flag:
                                logger.info(info)
                                break

                            history_train_student[step] = {
                                'loss'              : loss.item(),
                                f'val_{metric}'     : metric_value,
                                f'best_val_{metric}': self.best_metric_value,
                                'best_step'         : self.best_step,
                            }
                            last_step_log.update(history_train_student[step])

                        last_step_log['loss'] = loss.item()
                        pbar.update()
                        pbar.set_postfix(ordered_dict=last_step_log)

                        # if step >= step_balance['student-unsupervised']:
                        #     break

            if valid_flag:
                update_state_dict(self.model, self.best_model, 'model')
                metric_value, _, _ = self._valid_step(i, mode=valid_mode)
                if (self.direction == 'maximize' and metric_value > outer_best_metric_value) or \
                        (self.direction == 'minimize' and metric_value < outer_best_metric_value):
                    outer_best_metric_value = metric_value
                    outer_no_improve_cnt = 0
                    outer_best_model = copy.deepcopy(self.model.state_dict())
                else:
                    outer_no_improve_cnt += 1
                    if outer_patience > 0 and outer_no_improve_cnt >= outer_patience:
                        logger.info(f'[INFO] early stop outer loop @ iteration {i}')
                        break

        self._finalize()
        if valid_flag and outer_best_model is not None:
            self.model.load_state_dict(outer_best_model)

        history['train'] = history_train
        return history

    @torch.no_grad()
    def collect_pseudodataset_from_teacher(self, dataset):
        model = self.teacher_model
        model.eval()
        if isinstance(dataset, BaseDataset):
            valid_dataloader = self._init_valid_dataloader(
                dataset,
            )
        else:
            valid_dataloader = dataset
        features, probas = [], []
        for batch in valid_dataloader:
            output, feature = model(batch, return_features=True)
            proba = F.softmax(output, dim=-1)
            probas.append(proba)
            features.append(feature)

        return torch.vstack(probas), torch.vstack(features)

    @torch.no_grad()
    def collect_pseudodataset_teacher(self, dataset):
        model = self.teacher_model
        model.eval()
        if isinstance(dataset, BaseDataset):
            valid_dataloader = self._init_valid_dataloader(
                dataset,
                return_weak_labels=True,
            )
        else:
            valid_dataloader = dataset
        probas = []
        for batch in valid_dataloader:
            proba = model(batch)
            probas.append(proba)

        return torch.vstack(probas)

    @torch.no_grad()
    def predict_proba(self, dataset: Union[BaseDataset, DataLoader], mode: Optional[str] = 'student',
                      device: Optional[torch.device] = None, **kwargs: Any):
        assert mode in ['teacher', 'student'], f'mode: {mode} not support!'
        if self.teacher_inference:
            mode = 'teacher'

        if device is not None:
            model = self.model.to(device)
        else:
            model = self.model
        model.eval()
        if isinstance(dataset, BaseDataset):
            valid_dataloader = self._init_valid_dataloader(
                dataset,
                return_weak_labels=mode == 'teacher',
            )
        else:
            valid_dataloader = dataset
        probas = []
        for batch in valid_dataloader:
            output = model(batch)
            proba = F.softmax(output, dim=-1)

            probas.append(proba.cpu().numpy())

        return np.vstack(probas)


