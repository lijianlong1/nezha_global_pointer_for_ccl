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

from nezha_global_pointer_for_ccl.my_code.model import globalpointer_NER
from ark_nlp.model.ner.global_pointer_bert import get_default_model_optimizer
torch.backends.cudnn.enabled = False

def ner_pre_cluster(in_file, out_file, model_path, true_fgm=True):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ner_dict = {'11339': 0, '11340': 1, '11341': 2, '11342': 3, '11343': 4, '11344': 5, '11345': 6, '11346': 7, '11347': 8, '11348': 9, '11349': 10, '11350': 11, 'O': 12}
    # ner_dict = {'O': 0, 'company_name': 1, 'location': 2, 'org_name': 3, 'person_name': 4, 'product_name': 5, 'time': 6}
    model_ner = globalpointer_NER('/opt/lijianlong/gaiic/chinese-roberta-wwm', n_head=len(ner_dict))
    model_ner.to(device=device)
    model_ner.load_state_dict(torch.load(model_path, map_location=device),strict=False)
    # ark-nlp提供该函数：from ark_nlp.model.ner.global_pointer_bert import Predictor
    # 这里主要是为了可以比较清晰地看到解码过程，所以将代码copy到这
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
    optimizer = get_default_model_optimizer(model_ner)
    if true_fgm:
        model = AttackTask(model_ner, optimizer, 'gpce', cuda_device=0)
    else:
        model = Task(model_ner, optimizer, 'gpce', cuda_device=0)

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
                threshold=0
        ):
            """
            单样本预测
    
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

            return entities
    from ark_nlp.model.ner.global_pointer_bert import Tokenizer
    tokenizer = Tokenizer(vocab='/opt/lijianlong/gaiic/chinese-roberta-wwm', max_seq_len=480)
    ner_predictor_instance = GlobalPointerNERPredictor(model.module, tokenizer, ner_dict)
    from tqdm import tqdm
    predict_results = []
    out_file = open(out_file, 'w', encoding='utf-8')

    with open(in_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for _line in tqdm(lines):
            _line = eval(_line)
            sentence = _line['context'].strip()


            dict_result = {}


            dict_result['context'] = sentence
            dict_result['id'] = _line['id']
            dict_result['entities'] = []

            hash_leibie = {}

            ######## 在这再判断一下超出最大长度256的情况
            for _preditc in ner_predictor_instance.predict_one_sample(sentence):
                # 按照类别区分
                if _preditc['type'] not in hash_leibie:
                    hash_leibie[_preditc['type']] = [_preditc]
                elif _preditc['type'] in hash_leibie:
                    hash_leibie[_preditc['type']].append(_preditc)

            entities = []
            for i in hash_leibie:
                entity_one = {}
                temp_lable = hash_leibie[i]
                entity_one['label'] = i
                entity_span = []
                entity_text = []
                for t in temp_lable:
                    start = t['start_idx']
                    end = t['end_idx'] + 1
                    entity = t['entity']
                    entity_span.append([start, end])
                    entity_text.append(entity)

                entity_one['span'] = entity_span
                entity_one['text'] = entity_text
                ##### 直接进行相应的数据处理
                entities.append(entity_one)
            dict_result['entities'] = entities
            out_file.write(str(dict_result).replace("'",'"') + '\n')

    out_file.close()
ner_pre_cluster('/opt/lijianlong/gaiic/ner_for_laic/data_set/选手数据集/test.json', '/opt/lijianlong/gaiic/ner_for_laic/data_set/选手数据集/test_out.txt', '/opt/lijianlong/gaiic/ner_for_laic/my_code/bert_global_fgm_fanzui_quan.pth')
# ner_pre_cluster('../ner_ejournal/data_sets/cluener_public/dev.txt', '../ner_ejournal/data_sets/cluener_public/dev_out_nofgm.txt', 'roberta_clu_nofgm.pth', true_fgm= False)

