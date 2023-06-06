# -*- ecoding: utf-8 -*-
# @ModuleName: Task_FGM_PGD
# @Function: 
# @Author: long
# @Time: 2022/10/8 10:03
# *****************conding***************
import warnings
from collections import defaultdict

from ark_nlp.factory.utils import conlleval
from ark_nlp.model.ner.global_pointer_bert import Task
from torch.utils.data import DataLoader
from ark_nlp.factory.optimizer import get_optimizer

import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn

class GlobalPointerCrossEntropy(nn.Module):
    '''Multi-class Focal loss implementation'''
    def __init__(self, ):
        super(GlobalPointerCrossEntropy, self).__init__()

    @staticmethod
    def multilabel_categorical_crossentropy(y_true, y_pred):
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 1e12
        y_pred_pos = y_pred - (1 - y_true) * 1e12
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

        return neg_loss + pos_loss

    def forward(self, logits, target):
        """
        logits: [N, C, L, L]
        """
        bh = logits.shape[0] * logits.shape[1]
        target = torch.reshape(target.to_dense(), (bh, -1))
        logits = torch.reshape(logits, (bh, -1))
        return torch.mean(GlobalPointerCrossEntropy.multilabel_categorical_crossentropy(target, logits))


class FGM(object):
    """
    基于FGM算法的攻击机制

    Args:
        module (:obj:`torch.nn.Module`): 模型

    Examples::


    Reference:
        [1]  https://zhuanlan.zhihu.com/p/91269728
    """
    def __init__(self, module):
        self.module = module
        self.backup = {}

    def attack(
        self,
        epsilon=1.,
        emb_name='word_embeddings'
    ):
        for name, param in self.module.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(
        self,
        emb_name='word_embeddings'
    ):
        for name, param in self.module.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

class PGD:
    def __init__(self, model, eps=1., alpha=0.3):
        self.model = (
            model.module if hasattr(model, "module") else model
        )
        self.eps = eps
        self.alpha = alpha
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, emb_name='word_embeddings', is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data)

    def restore(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > self.eps:
            r = self.eps * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]

class AWP:
    """
    Implements weighted adverserial perturbation
    adapted from: https://www.kaggle.com/code/wht1996/feedback-nn-train/notebook


    adv_param(str): 要攻击的layer name，一般攻击第一层或者全部weight参数效果较好

    adv_lr(float): 攻击步长，这个参数相对难调节，如果只攻击第一层embedding，一般用1比较好，全部参数用0.1比较好。

    adv_eps(float): 参数扰动最大幅度限制，范围（0~ +∞），一般设置（0，1）之间相对合理一点。start_epoch(int): （0~ +∞）什么时候开始扰动，默认是0，如果效果不好可以调节值模型收敛一半的时候再开始攻击。
   """

    def __init__(self, model, optimizer, adv_param="word_embeddings", adv_lr=1.0, adv_eps=0.0001):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}
        self.backup_eps = {}
        self.loss = GlobalPointerCrossEntropy()
    def _compute_loss(
        self,
        inputs,
        logits,
        verbose=True,
        **kwargs
    ):
        loss = self.loss(logits, inputs['label_ids'])

        return loss

    def attack_backward(self, inputs):
        if self.adv_lr == 0:
            return
        self._save()
        self._attack_step()
        y_preds = self.model(**inputs)
        ## y_pre直接从相应的数据中得到
        adv_loss = self._compute_loss(inputs, y_preds)

        self.optimizer.zero_grad()
        return adv_loss

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    # 在损失函数之前获得梯度
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self,):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}


class AttackTask_FGM(Task):

    def _on_train_begin(
            self,
            train_data,
            validation_data,
            batch_size,
            lr,
            params,
            shuffle,
            num_workers=0,
            train_to_device_cols=None,
            **kwargs
    ):
        if hasattr(train_data, 'id2cat'):
            self.id2cat = train_data.id2cat
            self.cat2id = {v_: k_ for k_, v_ in train_data.id2cat.items()}

        # 在初始化时会有class_num参数，若在初始化时不指定，则在训练阶段从训练集获取信息
        if self.class_num is None:
            if hasattr(train_data, 'class_num'):
                self.class_num = train_data.class_num
            else:
                warnings.warn("The class_num is None.")

        if train_to_device_cols is None:
            self.train_to_device_cols = train_data.to_device_cols
        else:
            self.train_to_device_cols = train_to_device_cols

        train_generator = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=self._train_collate_fn
        )
        self.train_generator_lenth = len(train_generator)

        self.optimizer = get_optimizer(self.optimizer, self.module, lr, params)
        self.optimizer.zero_grad()



        self.module.train()

        self.fgm = FGM(self.module)
        # self.pgd = PGD(self.module)  # 尝试加入相应的不一样的扰动函数，进行对抗训练测试

        self._on_train_begin_record(**kwargs)

        return train_generator

    def _on_backward(
            self,
            inputs,
            outputs,
            logits,
            loss,
            gradient_accumulation_steps=1,
            **kwargs
    ):

        # 如果GPU数量大于1
        if self.n_gpu > 1:
            loss = loss.mean()
        # 如果使用了梯度累积，除以累积的轮数
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps

        loss.backward()
        #在此使用pgd方式进行对抗学习扰动

        self.fgm.attack(epsilon=1.0)    # 在这个地方可以调整扰动的数值概率
        logits = self.module(**inputs)
        _, attck_loss = self._get_train_loss(inputs, logits, **kwargs)
        attck_loss.backward()
        self.fgm.restore()

        self._on_backward_record(loss, **kwargs)

        return loss

class AttackTask_PGD(Task):


    def _on_train_begin(
            self,
            train_data,
            validation_data,
            batch_size,
            lr,
            params,
            shuffle,
            num_workers=0,
            train_to_device_cols=None,
            **kwargs
    ):


        if hasattr(train_data, 'id2cat'):
            self.id2cat = train_data.id2cat
            self.cat2id = {v_: k_ for k_, v_ in train_data.id2cat.items()}

        # 在初始化时会有class_num参数，若在初始化时不指定，则在训练阶段从训练集获取信息
        if self.class_num is None:
            if hasattr(train_data, 'class_num'):
                self.class_num = train_data.class_num
            else:
                warnings.warn("The class_num is None.")

        if train_to_device_cols is None:
            self.train_to_device_cols = train_data.to_device_cols
        else:
            self.train_to_device_cols = train_to_device_cols

        train_generator = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=self._train_collate_fn
        )
        self.train_generator_lenth = len(train_generator)

        self.optimizer = get_optimizer(self.optimizer, self.module, lr, params)
        self.optimizer.zero_grad()



        self.module.train()

        #self.fgm = FGM(self.module)
        self.pgd = PGD(self.module)  # 尝试加入相应的不一样的扰动函数，进行对抗训练测试

        self._on_train_begin_record(**kwargs)

        return train_generator

    def _on_backward(
            self,
            inputs,
            outputs,
            logits,
            loss,
            gradient_accumulation_steps=1,
            **kwargs
    ):

        # 如果GPU数量大于1
        if self.n_gpu > 1:
            loss = loss.mean()
        # 如果使用了梯度累积，除以累积的轮数
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps

        loss.backward()

        # 加入pdg方法进行参数的调节使用
        self.pgd.backup_grad()
        pgd_k = 3
        for _t in range(pgd_k):
            self.pgd.attack(is_first_attack=(_t == 0))
            if _t != pgd_k - 1:
                self.module.zero_grad()
            else:
                self.pgd.restore_grad()
            output_pgd = self.module(**inputs)
            logists_pgd, loss_adv = self._get_train_loss(inputs, output_pgd, **kwargs)
            loss_adv.backward()
        self.pgd.restore()



        self._on_backward_record(loss, **kwargs)

        return loss


    def _on_evaluate_step_end(self, inputs, outputs, **kwargs):

        with torch.no_grad():

            # compute loss
            logits, loss = self._get_evaluate_loss(inputs, outputs, **kwargs)

            numerate, denominator = conlleval.global_pointer_f1_score(
                inputs['label_ids'].to_dense().cpu(),
                logits.cpu()
            )

            y_pred = logits.cpu()
            y_true = inputs['label_ids'].to_dense().cpu()
#           这里将大于0的数全部变成1
            zero = torch.zeros_like(y_pred)
            one = torch.ones_like(y_pred)


            margin = 0.5

            # a中大于0.5的用one(1)替换,否则a替换,即不变
            a = torch.where(y_pred > margin, one, y_pred)

            # a中小于0.5的用zero(0)替换,否则a替换,即不变
            a = torch.where(a <= margin, zero, a)
            y_pred = a
            match_bianjie = torch.sum(y_true * y_pred).item()
            pre_bianjie = torch.sum(y_pred).item()
            true_bianjie = torch.sum(y_true).item()


            self.evaluate_logs['bianjie_match'] += match_bianjie
            self.evaluate_logs['bianjie_pre'] += pre_bianjie
            self.evaluate_logs['bianjie_true'] += true_bianjie






            self.evaluate_logs['numerate'] += numerate
            self.evaluate_logs['denominator'] += denominator

        self.evaluate_logs['eval_example'] += len(inputs['label_ids'])
        self.evaluate_logs['eval_step'] += 1
        self.evaluate_logs['eval_loss'] += loss.item()

    def _on_evaluate_epoch_end(
        self,
        validation_data,
        epoch=1,
        is_evaluate_print=True,
        id2cat=None,
        **kwargs
    ):

        if id2cat is None:
            id2cat = self.id2cat

        if is_evaluate_print:
            print('eval loss is {:.6f}, precision is:{}, recall is:{}, f1_score is:{}'.format(
                self.evaluate_logs['eval_loss'] / self.evaluate_logs['eval_step'],
                self.evaluate_logs['numerate'],
                self.evaluate_logs['denominator'],
                2*self.evaluate_logs['numerate']/self.evaluate_logs['denominator'])
            )
            p = self.evaluate_logs['bianjie_match']/(self.evaluate_logs['bianjie_pre'] + 1)
            r = self.evaluate_logs['bianjie_match']/(self.evaluate_logs['bianjie_true'] + 1)
            f1 = 2*p*r/(p+r+1e-3)
            print('边界识别的p:', p,
                  '边界识别的r:', r,
                  '边界识别的f1:', f1)

    def evaluate(
            self,
            validation_data,
            evaluate_batch_size=16,
            **kwargs
    ):
        """
        验证方法

        Args:
            validation_data (:obj:`ark_nlp dataset`): 训练的batch文本
            evaluate_batch_size (:obj:`int`, optional, defaults to 32): 验证阶段batch大小
            **kwargs (optional): 其他可选参数
        """  # noqa: ignore flake8"

        self.evaluate_logs = dict()
        self.evaluate_logs['bianjie_match'] = 0
        self.evaluate_logs['bianjie_pre'] = 0
        self.evaluate_logs['bianjie_true'] = 0

        evaluate_generator = self._on_evaluate_begin(
            validation_data,
            evaluate_batch_size,
            shuffle=False,
            **kwargs
        )

        with torch.no_grad():
            self._on_evaluate_epoch_begin(**kwargs)

            for step, inputs in enumerate(evaluate_generator):
                inputs = self._get_module_inputs_on_eval(inputs, **kwargs)

                # forward
                outputs = self.module(**inputs)

                self._on_evaluate_step_end(inputs, outputs, **kwargs)

            self._on_evaluate_epoch_end(validation_data, **kwargs)

        self._on_evaluate_end(**kwargs)



class AttackTask_PGD_logist_adjustment(Task):

    def _on_train_begin(
            self,
            train_data,
            validation_data,
            batch_size,
            lr,
            params,
            shuffle,
            num_workers=0,
            train_to_device_cols=None,
            **kwargs
    ):
        if hasattr(train_data, 'id2cat'):
            self.id2cat = train_data.id2cat
            self.cat2id = {v_: k_ for k_, v_ in train_data.id2cat.items()}

        # 在初始化时会有class_num参数，若在初始化时不指定，则在训练阶段从训练集获取信息
        if self.class_num is None:
            if hasattr(train_data, 'class_num'):
                self.class_num = train_data.class_num
            else:
                warnings.warn("The class_num is None.")

        if train_to_device_cols is None:
            self.train_to_device_cols = train_data.to_device_cols
        else:
            self.train_to_device_cols = train_to_device_cols

        train_generator = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=self._train_collate_fn
        )
        self.train_generator_lenth = len(train_generator)

        self.optimizer = get_optimizer(self.optimizer, self.module, lr, params)
        self.optimizer.zero_grad()



        self.module.train()

        #self.fgm = FGM(self.module)
        self.pgd = PGD(self.module)  # 尝试加入相应的不一样的扰动函数，进行对抗训练测试

        self._on_train_begin_record(**kwargs)

        return train_generator

    def _on_backward(
            self,
            inputs,
            outputs,
            logits,
            loss,
            gradient_accumulation_steps=1,
            **kwargs
    ):

        # 如果GPU数量大于1
        ###### 定义两个模型，实现相应的计算过程 ###########

        if self.n_gpu > 1:
            loss = loss.mean()
        # 如果使用了梯度累积，除以累积的轮数
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps
        ####### 在这个地方尝试加入相应的数据分布模块

        loss.backward()

        # 加入pdg方法进行参数的调节使用
        self.pgd.backup_grad()
        pgd_k = 3
        for _t in range(pgd_k):
            self.pgd.attack(is_first_attack=(_t == 0))
            if _t != pgd_k - 1:
                self.module.zero_grad()
            else:
                self.pgd.restore_grad()
            output_pgd = self.module(**inputs)
            logists_pgd, loss_adv = self._get_train_loss(inputs, output_pgd, **kwargs)
            loss_adv.backward()
        self.pgd.restore()
        self._on_backward_record(loss, **kwargs)
        return loss




class AttackTask_PGD_FGM(Task):

    def _on_train_begin(
            self,
            train_data,
            validation_data,
            batch_size,
            lr,
            params,
            shuffle,
            num_workers=0,
            train_to_device_cols=None,
            **kwargs
    ):
        if hasattr(train_data, 'id2cat'):
            self.id2cat = train_data.id2cat
            self.cat2id = {v_: k_ for k_, v_ in train_data.id2cat.items()}

        # 在初始化时会有class_num参数，若在初始化时不指定，则在训练阶段从训练集获取信息
        if self.class_num is None:
            if hasattr(train_data, 'class_num'):
                self.class_num = train_data.class_num
            else:
                warnings.warn("The class_num is None.")

        if train_to_device_cols is None:
            self.train_to_device_cols = train_data.to_device_cols
        else:
            self.train_to_device_cols = train_to_device_cols

        train_generator = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=self._train_collate_fn
        )
        self.train_generator_lenth = len(train_generator)

        self.optimizer = get_optimizer(self.optimizer, self.module, lr, params)
        self.optimizer.zero_grad()



        self.module.train()

        self.fgm = FGM(self.module)
        self.pgd = PGD(self.module)  # 尝试加入相应的不一样的扰动函数，进行对抗训练测试

        self._on_train_begin_record(**kwargs)

        return train_generator

    def _on_backward(
            self,
            inputs,
            outputs,
            logits,
            loss,
            gradient_accumulation_steps=1,
            **kwargs
    ):

        # 如果GPU数量大于1
        if self.n_gpu > 1:
            loss = loss.mean()
        # 如果使用了梯度累积，除以累积的轮数
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps

        loss.backward()

        ########### 直接使用相关的FGM进行处理 ###########
        self.fgm.attack(epsilon=1.0)    # 在这个地方可以调整扰动的数值概率
        logits = self.module(**inputs)
        _, attck_loss = self._get_train_loss(inputs, logits, **kwargs)
        attck_loss.backward()
        self.fgm.restore()

        # 加入pdg方法进行参数的调节使用
        self.pgd.backup_grad()
        pgd_k = 3
        for _t in range(pgd_k):
            self.pgd.attack(is_first_attack=(_t == 0))
            if _t != pgd_k - 1:
                self.module.zero_grad()
            else:
                self.pgd.restore_grad()
            output_pgd = self.module(**inputs)
            logists_pgd, loss_adv = self._get_train_loss(inputs, output_pgd, **kwargs)
            loss_adv.backward()
        self.pgd.restore()
        self._on_backward_record(loss, **kwargs)

        return loss

class AttackTask_AWP(Task):

    def _on_train_begin(
            self,
            train_data,
            validation_data,
            batch_size,
            lr,
            params,
            shuffle,
            num_workers=0,
            train_to_device_cols=None,
            **kwargs
    ):
        if hasattr(train_data, 'id2cat'):
            self.id2cat = train_data.id2cat
            self.cat2id = {v_: k_ for k_, v_ in train_data.id2cat.items()}

        # 在初始化时会有class_num参数，若在初始化时不指定，则在训练阶段从训练集获取信息
        if self.class_num is None:
            if hasattr(train_data, 'class_num'):
                self.class_num = train_data.class_num
            else:
                warnings.warn("The class_num is None.")

        if train_to_device_cols is None:
            self.train_to_device_cols = train_data.to_device_cols
        else:
            self.train_to_device_cols = train_to_device_cols

        train_generator = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=self._train_collate_fn
        )
        self.train_generator_lenth = len(train_generator)

        self.optimizer = get_optimizer(self.optimizer, self.module, lr, params)
        self.optimizer.zero_grad()



        self.module.train()

        #self.fgm = FGM(self.module)
        # self.pgd = PGD(self.module)  # 尝试加入相应的不一样的扰动函数，进行对抗训练测试
        self.awp = AWP(self.module, self.optimizer, adv_param="word_embeddings", adv_lr=1.0, adv_eps=0.0001)
        self._on_train_begin_record(**kwargs)

        return train_generator

    def _on_backward(
            self,
            inputs,
            outputs,
            logits,
            loss,
            gradient_accumulation_steps=1,
            **kwargs
    ):

        # 如果GPU数量大于1
        if self.n_gpu > 1:
            loss = loss.mean()
        # 如果使用了梯度累积，除以累积的轮数
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps

        loss.backward()

        ###### 在这个地方加上相应的awp攻击####
        # 表示从第几轮开始进行attack，一般是训练到某一个程度进行攻击
        ## 目前暂定10轮之后开始进行攻击。
        # print(self.logs)
        if self.logs['global_step'] > 27 * 11:
            print('正在添加扰动')
            # output_awp = self.module(**inputs)
            # logists_pgd, loss_adv = self._get_train_loss(inputs, output_pgd, **kwargs)
            loss_awp = self.awp.attack_backward(inputs)
            loss_awp.backward()
            self.awp._restore()
        self.optimizer.step()
        #### 需要调节相应的损失函数的获取方法#######


        self._on_backward_record(loss, **kwargs)

        return loss

class AttackTask_PGD_AWP(Task):


    def _on_train_begin(
            self,
            train_data,
            validation_data,
            batch_size,
            lr,
            params,
            shuffle,
            num_workers=0,
            train_to_device_cols=None,
            **kwargs
    ):
        self.best_f1 = 0.0
        if hasattr(train_data, 'id2cat'):
            self.id2cat = train_data.id2cat
            self.cat2id = {v_: k_ for k_, v_ in train_data.id2cat.items()}

        # 在初始化时会有class_num参数，若在初始化时不指定，则在训练阶段从训练集获取信息
        if self.class_num is None:
            if hasattr(train_data, 'class_num'):
                self.class_num = train_data.class_num
            else:
                warnings.warn("The class_num is None.")

        if train_to_device_cols is None:
            self.train_to_device_cols = train_data.to_device_cols
        else:
            self.train_to_device_cols = train_to_device_cols

        train_generator = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=self._train_collate_fn
        )
        self.train_generator_lenth = len(train_generator)

        self.optimizer = get_optimizer(self.optimizer, self.module, lr, params)
        self.optimizer.zero_grad()



        self.module.train()

        #self.fgm = FGM(self.module)
        self.pgd = PGD(self.module)  # 尝试加入相应的不一样的扰动函数，进行对抗训练测试
        self.awp = AWP(self.module, self.optimizer, adv_param="word_embeddings", adv_lr=0.1, adv_eps=0.0001)
        self._on_train_begin_record(**kwargs)

        return train_generator

    def _on_backward(
            self,
            inputs,
            outputs,
            logits,
            loss,
            gradient_accumulation_steps=1,
            **kwargs
    ):

        # 如果GPU数量大于1
        if self.n_gpu > 1:
            loss = loss.mean()
        # 如果使用了梯度累积，除以累积的轮数
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps

        loss.backward()

        # 加入pdg方法进行参数的调节使用
        self.pgd.backup_grad()
        pgd_k = 3
        for _t in range(pgd_k):
            self.pgd.attack(is_first_attack=(_t == 0))
            if _t != pgd_k - 1:
                self.module.zero_grad()
            else:
                self.pgd.restore_grad()
            output_pgd = self.module(**inputs)
            logists_pgd, loss_adv = self._get_train_loss(inputs, output_pgd, **kwargs)
            loss_adv.backward()
        self.pgd.restore()

        # 加入awp方法进行参数的调节使用
        if self.logs['global_step'] > 100:
            # print('正在添加扰动')
            # output_awp = self.module(**inputs)
            # logists_pgd, loss_adv = self._get_train_loss(inputs, output_pgd, **kwargs)
            loss_awp = self.awp.attack_backward(inputs)
            loss_awp.backward()
            self.awp._restore()
        self.optimizer.step()



        self._on_backward_record(loss, **kwargs)

        return loss

    def _on_evaluate_epoch_end(
            self,
            validation_data,
            epoch=1,
            is_evaluate_print=True,
            id2cat=None,
            **kwargs
    ):


        if id2cat is None:
            id2cat = self.id2cat

        if is_evaluate_print:
            print('eval loss is {:.6f}, precision is:{}, recall is:{}, f1_score is:{}'.format(
                self.evaluate_logs['eval_loss'] / self.evaluate_logs['eval_step'],
                self.evaluate_logs['numerate'],
                self.evaluate_logs['denominator'],
                2 * self.evaluate_logs['numerate'] / self.evaluate_logs['denominator'])
            )
        ###### 对模型进行存储###
        f1_score = 2 * self.evaluate_logs['numerate'] / self.evaluate_logs['denominator']
        if f1_score > self.best_f1:
            self.best_f1 = f1_score
            torch.save(self.module.state_dict(), '' + '.pth')
            print('目前最高的f1值为：', self.best_f1, ',完成模型的存储')





