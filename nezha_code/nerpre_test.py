# Coding by long
# Datatime:2022/4/11 9:23
# Filename:nerpre.py
# Toolby: PyCharm
# description:
# ______________coding_____________
import json
import warnings

import torch
import numpy as np
from ark_nlp.factory.optimizer import get_optimizer
from ark_nlp.factory.task import Task
from ark_nlp.factory.utils.attack import FGM
from torch.utils.data import DataLoader
from nezha_code.Task_FGM_PGD import AttackTask_PGD, AttackTask_FGM
from nezha_code.model import globalpointer_NER, globalpointer_tcn_NER, globalpointer_ro_NER, globalpointer_tcn_lstm_NER
from ark_nlp.model.ner.global_pointer_bert import get_default_model_optimizer
from ark_nlp.model.ner.global_pointer_bert import Tokenizer
from tqdm import tqdm

torch.backends.cudnn.enabled = False

class GlobalPointerNERPredictor(object):
    """
    GlobalPointer命名实体识别的预测器

    Args:
        module: 深度学习模型
        tokernizer: 分词器
        cat2id (:obj:`dict`): 标签映射
    """  # noqa: ignore flake8"

    def __init__(
            self,
            module,
            tokernizer,
            cat2id
    ):
        self.module = module
        self.module.task = 'TokenLevel'

        self.cat2id = cat2id
        self.tokenizer = tokernizer
        self.device = list(self.module.parameters())[0].device

        self.id2cat = {}
        for cat_, idx_ in self.cat2id.items():
            self.id2cat[idx_] = cat_

    def _convert_to_transfomer_ids(
            self,
            text
    ):

        tokens = self.tokenizer.tokenize(text)
        token_mapping = self.tokenizer.get_token_mapping(text, tokens)

        input_ids = self.tokenizer.sequence_to_ids(tokens)
        input_ids, input_mask, segment_ids = input_ids

        zero = [0 for i in range(self.tokenizer.max_seq_len)]
        span_mask = [input_mask for i in range(sum(input_mask))]
        span_mask.extend([zero for i in range(sum(input_mask), self.tokenizer.max_seq_len)])
        span_mask = np.array(span_mask)

        features = {
            'input_ids': input_ids,
            'attention_mask': input_mask,
            'token_type_ids': segment_ids,
            'span_mask': span_mask
        }

        return features, token_mapping

    def _get_input_ids(
            self,
            text
    ):
        if self.tokenizer.tokenizer_type == 'vanilla':
            return self._convert_to_vanilla_ids(text)
        elif self.tokenizer.tokenizer_type == 'transfomer':
            return self._convert_to_transfomer_ids(text)
        elif self.tokenizer.tokenizer_type == 'customized':
            return self._convert_to_customized_ids(text)
        else:
            raise ValueError("The tokenizer type does not exist")

    def _get_module_one_sample_inputs(
            self,
            features
    ):
        return {col: torch.Tensor(features[col]).type(torch.long).unsqueeze(0).to(self.device) for col in features}

    def predict_one_sample(
            self,
            text='',
            threshold=0.15
    ):
        """
        单样本预测
        阈值在0.15时效果最好，需要进行参数的调整和使用,9478

        Args:
            text (:obj:`string`): 输入文本
            threshold (:obj:`float`, optional, defaults to 0): 预测的阈值
        """  # noqa: ignore flake8"

        features, token_mapping = self._get_input_ids(text)
        self.module.eval()

        with torch.no_grad():
            inputs = self._get_module_one_sample_inputs(features)
            scores = self.module(**inputs)[0].cpu()

        scores[:, [0, -1]] -= np.inf
        scores[:, :, [0, -1]] -= np.inf

        entities = []
        # print(token_mapping)

        for category, start, end in zip(*np.where(scores > threshold)):
            # print(end, start)
            if end - 1 > token_mapping[-1][-1]:
                break
            try:
                if token_mapping[start - 1][0] <= token_mapping[end - 1][-1]:
                    entitie_ = {
                        "start_idx": token_mapping[start - 1][0],
                        "end_idx": token_mapping[end - 1][-1],
                        "entity": text[token_mapping[start - 1][0]: token_mapping[end - 1][-1] + 1],
                        "type": self.id2cat[category]
                    }

                    if entitie_['entity'] == '':
                        continue

                    entities.append(entitie_)
            except Exception as e:
                print(e)
                continue
        ################ 在文中还得注意两个实体，玉道，和回翌，可以后面尝试是用代码进行处理#############,在这一块结合相关的规则处理逻辑######
        ################ 对常见的实体信息进行检索处理，加上相关的处理规则逻辑###########
        entity_special = [{'entity':'五術', 'type': 'BOOK'},
                          {'entity':'御史臺臣', 'type': 'OFI'},
                          {'entity':'鎮守使', 'type': 'OFI'},
                          {'entity':'別將', 'type': 'OFI'},
                          {'entity':'列校', 'type': 'OFI'}]
        ######,
        entity_delete = ['廓王', '泰'] #，没有效果
        entity_addition = []
        # 在文本数据中找到这两个实体并进行标号即可
        for i in entity_special:
            if i['entity'] in text:
                index_start = text.index(i['entity'])
                index_end = index_start + len(i['entity'])
                entity_addition.append(
                    {'entity':i['entity'],
                     'type': i['type'],
                     "start_idx": index_start,
                     "end_idx": index_end})
                print({'entity':i['entity'],
                     'type': i['type'],
                     "start_idx": index_start,
                     "end_idx": index_end})
        entities = entities + entity_addition

        # 去除不需要关注的实体信息，包括‘帝’###############
        entities_entity = []
        ########### 判断要不要全删
        index_tag = 0 # tag为0表示不用做操作，1表示只有一个实体‘泰’，2表示两个实体都在里面
        entities_new = []
        for index, i in enumerate(entities, 0):
            entities_entity.append(i['entity'])
        for i in entity_delete:
            if i in entities_entity:
                index_tag+=1
        if index_tag==0:
            entities_new = entities
        elif index_tag==1:
            entities_new = entities
        else:
            for i in entities:
                if i['entity'] not in entity_delete:
                    entities_new.append(i)
                else:
                    print('error', i)



        entities_new.sort(key=lambda s: (s['start_idx'], s['end_idx'] * -1))  # #-1表示降序 ##### 对相应的错误的信息几行处理，选取更好的不易出错的实体信息

        return entities_new


class GlobalPointerNERPredictor_1(object):
    """
    GlobalPointer命名实体识别的预测器

    Args:
        module: 深度学习模型
        tokernizer: 分词器
        cat2id (:obj:`dict`): 标签映射
    """  # noqa: ignore flake8"

    def __init__(
            self,
            module,
            tokernizer,
            cat2id
    ):
        self.module = module
        self.module.task = 'TokenLevel'

        self.cat2id = cat2id
        self.tokenizer = tokernizer
        self.device = list(self.module.parameters())[0].device

        self.id2cat = {}
        for cat_, idx_ in self.cat2id.items():
            self.id2cat[idx_] = cat_

    def _convert_to_transfomer_ids(
            self,
            text
    ):

        tokens = self.tokenizer.tokenize(text)
        token_mapping = self.tokenizer.get_token_mapping(text, tokens)

        input_ids = self.tokenizer.sequence_to_ids(tokens)
        input_ids, input_mask, segment_ids = input_ids

        zero = [0 for i in range(self.tokenizer.max_seq_len)]
        span_mask = [input_mask for i in range(sum(input_mask))]
        span_mask.extend([zero for i in range(sum(input_mask), self.tokenizer.max_seq_len)])
        span_mask = np.array(span_mask)

        features = {
            'input_ids': input_ids,
            'attention_mask': input_mask,
            'token_type_ids': segment_ids,
            'span_mask': span_mask
        }

        return features, token_mapping

    def _get_input_ids(
            self,
            text
    ):
        if self.tokenizer.tokenizer_type == 'vanilla':
            return self._convert_to_vanilla_ids(text)
        elif self.tokenizer.tokenizer_type == 'transfomer':
            return self._convert_to_transfomer_ids(text)
        elif self.tokenizer.tokenizer_type == 'customized':
            return self._convert_to_customized_ids(text)
        else:
            raise ValueError("The tokenizer type does not exist")

    def _get_module_one_sample_inputs(
            self,
            features
    ):
        return {col: torch.Tensor(features[col]).type(torch.long).unsqueeze(0).to(self.device) for col in features}

    def predict_one_sample(
            self,
            text='',
            threshold=-0.5
    ):
        """
        单样本预测
        阈值在-0.5时效果最好，需要进行参数的调整和使用,94690

        Args:
            text (:obj:`string`): 输入文本
            threshold (:obj:`float`, optional, defaults to 0): 预测的阈值
        """  # noqa: ignore flake8"

        features, token_mapping = self._get_input_ids(text)
        self.module.eval()

        with torch.no_grad():
            inputs = self._get_module_one_sample_inputs(features)
            scores = self.module(**inputs)[0].cpu()

        scores[:, [0, -1]] -= np.inf
        scores[:, :, [0, -1]] -= np.inf

        entities = []
        # print(token_mapping)

        for category, start, end in zip(*np.where(scores > threshold)):
            # print(end, start)
            if end - 1 > token_mapping[-1][-1]:
                break
            try:
                if token_mapping[start - 1][0] <= token_mapping[end - 1][-1]:
                    entitie_ = {
                        "start_idx": token_mapping[start - 1][0],
                        "end_idx": token_mapping[end - 1][-1],
                        "entity": text[token_mapping[start - 1][0]: token_mapping[end - 1][-1] + 1],
                        "type": self.id2cat[category]
                    }

                    if entitie_['entity'] == '':
                        continue

                    entities.append(entitie_)
            except Exception as e:
                print(e)
                continue
        entities.sort(key=lambda s:s['start_idx'])

        return entities


class AttackTask(Task):

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

        self._on_train_begin_record(**kwargs)

        return train_generator


def ner_pre_cluster(in_file, out_file, model_path, true_fgm=True, max_len=128):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ner_dict = {'BOOK': 0, 'O': 1, 'OFI': 2, 'PER': 3}
    # ner_dict = {'O': 0, 'company_name': 1, 'location': 2, 'org_name': 3, 'person_name': 4, 'product_name': 5, 'time': 6}
    # model_ner = globalpointer_NER('/home/lijianlong/gaiic/nezha-cn-base', n_head=len(ner_dict))
    model_ner = globalpointer_tcn_NER('../nezha-cn-base', n_head=len(ner_dict))
    # model_ner = globalpointer_tcn_lstm_NER('/home/lijianlong/gaiic/nezha-cn-base', n_head=len(ner_dict))
    # model_ner = globalpointer_tcn_NER('/home/lijianlong/gaiic_semi_com/nezha_global_pointer_for_ccl/pre_train_nezha/mynezha', n_head=len(ner_dict))

    model_ner.to(device=device)

    print('hgenvoiurghesl')
    model_ner.load_state_dict(torch.load(model_path, map_location=device),strict=False)
    # ark-nlp提供该函数：from ark_nlp.model.ner.global_pointer_bert import Predictor
    # 这里主要是为了可以比较清晰地看到解码过程，所以将代码copy到这

    optimizer = get_default_model_optimizer(model_ner)
    if true_fgm:
        model = AttackTask_FGM(model_ner, optimizer, 'gpce', cuda_device=0)
    else:
        model = AttackTask_PGD(model_ner, optimizer, 'gpce', cuda_device=0)



    tokenizer = Tokenizer(vocab='../nezha-cn-base', max_seq_len=max_len)
    ner_predictor_instance = GlobalPointerNERPredictor(model.module, tokenizer, ner_dict)

    predict_results = []
    out_file = open(out_file, 'w', encoding='utf-8')

    with open(in_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for _line in tqdm(lines):
            _line = _line.strip('\n')
            sentence = _line
            # sentence = _line['fullText'].strip()


            sentence_result = ''
            start_id = 0

            ###### 需要把有覆盖的实体去除掉


            for _predict in ner_predictor_instance.predict_one_sample(sentence):# 按照类别区分,在此直接拿到相应的预测输出进行文件的写入
                try:
                    # print(_predict['entity'])
                    index = sentence.index(_predict['entity'])
                    sentcen_re = sentence[start_id:index]
                    #{惟明|PER}
                    entity_conta = '{' + _predict['entity'] + '|' + _predict['type'] +'}'
                    sentence_result += sentcen_re + entity_conta
                    sentence = sentence[index + len(_predict['entity']):]
                except Exception as e:
                    print(e)
                    print('出错的相关的实体信息',_predict)
                # finally:
                #     print(_predict)


            sentence_result += sentence






                    ###################### 在此加上一些后处理逻辑进行处理判断###############


            out_file.write(str(sentence_result) + '\n')
    out_file.close()

def ner_pre_cluster_vote(in_file, out_file, model_path_1,model_path_2, model_path_3, true_fgm=True, max_len=128):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    ner_dict = {'BOOK': 0, 'O': 1, 'OFI': 2, 'PER': 3}
    tokenizer = Tokenizer(vocab='../nezha-cn-base', max_seq_len=max_len)
    # ner_dict = {'O': 0, 'company_name': 1, 'location': 2, 'org_name': 3, 'person_name': 4, 'product_name': 5, 'time': 6}
    # model_ner = globalpointer_NER('/home/lijianlong/gaiic/nezha-cn-base', n_head=len(ner_dict))
    model_ner_1 = globalpointer_tcn_NER('../nezha-cn-base', n_head=len(ner_dict))
    # model_ner = globalpointer_tcn_NER('/home/lijianlong/gaiic_semi_com/nezha_global_pointer_for_ccl/pre_train_nezha/mynezha', n_head=len(ner_dict))
    model_ner_1.to(device=device)
    model_ner_1.load_state_dict(torch.load(model_path_1, map_location=device), strict=False)
    # ark-nlp提供该函数：from ark_nlp.model.ner.global_pointer_bert import Predictor
    # 这里主要是为了可以比较清晰地看到解码过程，所以将代码copy到这
    optimizer_1 = get_default_model_optimizer(model_ner_1)
    if true_fgm:
        model_1 = AttackTask(model_ner_1, optimizer_1, 'gpce', cuda_device=0)
    else:
        model_1 = AttackTask_PGD(model_ner_1, optimizer_1, 'gpce', cuda_device=0)
    ner_predictor_instance_1 = GlobalPointerNERPredictor_1(model_1.module, tokenizer, ner_dict)




################## 加载第二个模型
    model_ner_2 = globalpointer_tcn_NER('../nezha-cn-base', n_head=len(ner_dict))
    # model_ner = globalpointer_tcn_NER('/home/lijianlong/gaiic_semi_com/nezha_global_pointer_for_ccl/pre_train_nezha/mynezha', n_head=len(ner_dict))
    model_ner_2.to(device=device)
    model_ner_2.load_state_dict(torch.load(model_path_2, map_location=device), strict=False)
    # ark-nlp提供该函数：from ark_nlp.model.ner.global_pointer_bert import Predictor
    # 这里主要是为了可以比较清晰地看到解码过程，所以将代码copy到这
    optimizer_2 = get_default_model_optimizer(model_ner_2)
    if true_fgm:
        model_2 = AttackTask(model_ner_2, optimizer_2, 'gpce', cuda_device=0)
    else:
        model_2 = AttackTask_PGD(model_ner_2, optimizer_2, 'gpce', cuda_device=0)
    ner_predictor_instance_2 = GlobalPointerNERPredictor(model_2.module, tokenizer, ner_dict)


    ############################# 加载第三个模型
    model_ner_3 = globalpointer_tcn_NER('../nezha-cn-base', n_head=len(ner_dict))
    # model_ner = globalpointer_tcn_NER('/home/lijianlong/gaiic_semi_com/nezha_global_pointer_for_ccl/pre_train_nezha/mynezha', n_head=len(ner_dict))
    model_ner_3.to(device=device)
    model_ner_3.load_state_dict(torch.load(model_path_3, map_location=device), strict=False)
    # ark-nlp提供该函数：from ark_nlp.model.ner.global_pointer_bert import Predictor
    # 这里主要是为了可以比较清晰地看到解码过程，所以将代码copy到这
    optimizer_3 = get_default_model_optimizer(model_ner_3)
    if true_fgm:
        model_3 = AttackTask(model_ner_3, optimizer_3, 'gpce', cuda_device=0)
    else:
        model_3 = AttackTask_PGD(model_ner_1, optimizer_3, 'gpce', cuda_device=0)
    ner_predictor_instance_3 = GlobalPointerNERPredictor_1(model_3.module, tokenizer, ner_dict)

    ###############################################################################
    predict_results = []
    out_file = open(out_file, 'w', encoding='utf-8')

    with open(in_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for _line in tqdm(lines):
            _line = _line.strip('\n')
            sentence = _line
            # sentence = _line['fullText'].strip()

            sentence_result = ''
            start_id = 0

            ###### 需要把有覆盖的实体去除掉
            result_1 = ner_predictor_instance_1.predict_one_sample(sentence)
            result_2 = ner_predictor_instance_2.predict_one_sample(sentence)
            result_3 = ner_predictor_instance_3.predict_one_sample(sentence)
            result_all = result_1 + result_2 + result_3

            result_all_hash = {}
            for i in result_all:
                i = str(i)
                if i not in result_all_hash:
                    result_all_hash[i] = 1
                else:
                    result_all_hash[i] += 1

            ###### 对相应的数据进行排序

            result_list = []
            for i in result_all_hash:
                if result_all_hash[i] >= 3:
                    result_list.append(eval(i))

            result_list.sort(key=lambda s:s['start_idx'])



            ############ 对相关的数据进行投票处理 ###########



            for _predict in result_list:  # 按照类别区分,在此直接拿到相应的预测输出进行文件的写入
                try:
                    # print(_predict['entity'])
                    index = sentence.index(_predict['entity'])
                    sentcen_re = sentence[start_id:index]
                    # {惟明|PER}
                    entity_conta = '{' + _predict['entity'] + '|' + _predict['type'] + '}'
                    sentence_result += sentcen_re + entity_conta
                    sentence = sentence[index + len(_predict['entity']):]
                except Exception as e:
                    print(e)
                    print('出错的相关的实体信息', _predict)
                # finally:
                #     print(_predict)

            sentence_result += sentence

            ###################### 在此加上一些后处理逻辑进行处理判断###############

            out_file.write(str(sentence_result) + '\n')
    out_file.close()

# ner_pre_cluster_vote('../data_set/GuNER2023_test_public.txt', 'result_vote.txt', 'train_fgm.pth','train_fgm_2.pth','train_fgm_3.pth')

ner_pre_cluster('../data_set/GuNER2023_test_public.txt', 'result.txt', 'train_fgm_2.pth') # 92.28
#ner_pre_cluster('/opt/lijianlong/gaiic/ner_for_laic/data_set/选手数据集/test.json', '/opt/lijianlong/gaiic/ner_for_laic/data_set/选手数据集/test_out_nezhaTCN_tcn_pgd_pretrain_0_80.txt', '/opt/lijianlong/gaiic/ner_for_laic/nezha_code/nezhatcn_global_tcn_pgd_0_90_1.pth')   #### 91.53
# ner_pre_cluster('../ner_ejournal/data_sets/cluener_public/dev.txt', '../ner_ejournal/data_sets/cluener_public/dev_out_nofgm.txt', 'roberta_clu_nofgm.pth', true_fgm= False)
# ner_pre_cluster('/opt/lijianlong/gaiic/ner_for_laic/data_set/选手数据集/test.json', '/opt/lijianlong/gaiic/ner_for_laic/data_set/选手数据集/test_out_nezhaTCN_tcn_pgd_pretrain_0_80_seed_1.txt', '/opt/lijianlong/gaiic/ner_for_laic/nezha_code/nezhatcn_global_tcn_pgd_alldata_0_seed_1.pth')
# ner_pre_cluster('/opt/lijianlong/gaiic/ner_for_laic/data_set/无标注数据集/危险驾驶罪-样本标签集-8000', '/opt/lijianlong/gaiic/ner_for_laic/data_set/选手数据集/wubiaozhu_8000_out_nezhatcn_tcn_pgd_pretrain.txt', '/opt/lijianlong/gaiic/ner_for_laic/nezha_code/nezhatcn_global_tcn_pgd_10_100_best.pth')




