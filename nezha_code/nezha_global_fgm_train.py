# Coding by long
# Datatime:2022/4/2 21:27
# Filename:global_fgm_train.py
# Toolby: PyCharm
# ______________coding_____________
import warnings
warnings.filterwarnings("ignore")

import torch
import pandas as pd
from ark_nlp.factory.utils.seed import set_seed
from ark_nlp.model.ner.global_pointer_bert import Dataset
from ark_nlp.model.ner.global_pointer_bert import Task
from ark_nlp.model.ner.global_pointer_bert import get_default_model_optimizer
from ark_nlp.model.ner.global_pointer_bert import Tokenizer

set_seed(42)

### 后面使用不一样的seed，对相应的数据进行投票，得到最好的三个模型，然后对2000条数据进行
### 进行投票伪标。大于等于2的。

from ark_nlp.factory.utils.conlleval import get_entity_bio

torch.backends.cudnn.enabled = False
# 记录最好的模型

datalist = []# bio_tran_step2_step1.txt
with open('../data_set/data_bio_delete', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    lines.append('\n')

    text = []
    labels = []
    label_set = set()

    for line in lines:
        try:
            if line == '\n':
                text = ''.join(text)
                entity_labels = []
                for _type, _start_idx, _end_idx in get_entity_bio(labels, id2label=None):
                    entity_labels.append({
                        'start_idx': _start_idx,
                        'end_idx': _end_idx,
                        'type': _type,
                        'entity': text[_start_idx: _end_idx + 1]
                    })

                if text == '':
                    continue

                datalist.append({
                    'text': text,
                    'label': entity_labels
                })

                text = []
                labels = []

            elif line == '  O\n':
                text.append(' ')
                labels.append('O')
            else:
                line = line.strip('\n').split()
                if len(line) == 1:
                    term = ' '
                    label = line[0]
                else:
                    # print(line)
                    term, label = line
                # print(text)
                text.append(term)
                label_set.add(label.split('-')[-1])
                labels.append(label)
        except Exception as e:
            print(e)


train_data_df = pd.DataFrame(datalist[:2000])
print(len(datalist[:2000]))

train_data_df['label'] = train_data_df['label'].apply(lambda x: str(x))

dev_data_df = pd.DataFrame(datalist[2000:])
print(len(datalist[2000:]))

dev_data_df['label'] = dev_data_df['label'].apply(lambda x: str(x))


label_list = sorted(list(label_set))
print(label_list)

print(label_list)

ner_train_dataset = Dataset(train_data_df, categories=label_list)
ner_dev_dataset = Dataset(dev_data_df, categories=ner_train_dataset.categories)


tokenizer = Tokenizer(vocab='../nezha-cn-base', max_seq_len=128)


ner_train_dataset.convert_to_ids(tokenizer)
ner_dev_dataset.convert_to_ids(tokenizer)

print(ner_train_dataset.cat2id)
#config = NeZhaConfig.from_pretrained('../torch_bertner/nezha-cn-base',
                                                 #num_labels=len(ner_train_dataset.cat2id))


torch.cuda.empty_cache()


from nezha_code.model import globalpointer_NER, globalpointer_tcn_NER, globalpointer_ro_NER, globalpointer_tcn_lstm_NER, globalpointer_tcn_tcn_f1_NER, globalpointer_tcn_lstmpre_NER
# dl_module = globalpointer_NER('/home/lijianlong/gaiic/nezha-cn-base', n_head=len(label_list))
# 2e-5,64,13 fgm_train.pth

# dl_module = globalpointer_NER('../pre_train_nezha/mynezha/checkpoint-500000', n_head=len(label_list))


# dl_module = globalpointer_tcn_NER('/home/lijianlong/gaiic/nezha-cn-base', n_head=len(label_list))
# 2e-5,64,30(26)huozhe20,9161

# dl_module = globalpointer_tcn_lstmpre_NER('/home/lijianlong/gaiic/pre_train_nezha/mynezha_4', n_head=len(label_list))

dl_module = globalpointer_tcn_NER('../pre_train_nezha/mynezha/checkpoint-500000', n_head=len(label_list))
## warmup, 20(20),5e-5,64,预训练9431，删除没有实体的训练数据，加上fgm对抗,模型名称train_fgm.pth
## warmup, 20(20),5e-5,48,预训练94648，删除没有实体的训练数据，加上fgm对抗,模型名称train_fgm_2.pth

# dl_module = globalpointer_tcn_lstm_NER('/home/lijianlong/gaiic_semi_com/nezha_global_pointer_for_ccl/pre_train_nezha/mynezha_2/checkpoint-500000', n_head=len(label_list))

num_epoches = 20
batch_size = 48



from transformers import AdamW,get_cosine_schedule_with_warmup


def get_optimizer_grouped_parameters( model,
                                      weight_decay=5e-5,
                                      learning_rate=5e-5,
                                      eps: float = 1e-6,
                                      correct_bias: bool = True,
                                      ):
    no_decay = ["bias", "LayerNorm.weight"]


    group1=['layers.0','layers.1','layers.2','layers.3']
    group2=['layers.4','layers.5','layers.6','.layers.7']
    group3=['layers.8','layers.9','layers.10','layers.11']

    group4 = ['lstm', 'tcn', 'lm_2']
    group_all=['layers.0','layers.1','layers.2','encoder.layers.3','layers.4','layers.5','layers.6','layers.7','layers.8','layers.9','layers.10','layers.11', 'lstm', 'tcn', 'lm_2']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group1)],'weight_decay': weight_decay, 'lr': learning_rate/2.6},
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group2)],'weight_decay': weight_decay, 'lr': learning_rate},
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group3)],'weight_decay': weight_decay, 'lr': learning_rate*2.6},


        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group4)],'weight_decay': weight_decay, 'lr': learning_rate * 10},


        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group1)],'weight_decay': 0.0, 'lr': learning_rate/2.6},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group2)],'weight_decay': 0.0, 'lr': learning_rate},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group3)],'weight_decay': 0.0, 'lr': learning_rate*2.6},


        {'params': [p for n, p in model.named_parameters() if
                    any(nd in n for nd in no_decay) and any(nd in n for nd in group4)], 'weight_decay': weight_decay*10,
         'lr': learning_rate * 10}


        # {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in group_all)],
        #  'weight_decay': weight_decay, 'lr': 1e-3},
        # {'params': [p for n, p in model.named_parameters() if
        #             any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],
        #  'weight_decay': 0, 'lr': 1e-3}

        # {'params': [n for n, p in model.named_parameters() if 'lstm' or 'tcn' in n], 'lr':learning_rate*20, "weight_decay": 0.0},
    ]


    optimizer = AdamW(optimizer_grouped_parameters,
                      eps=eps,
                      correct_bias=correct_bias
                      )

    return optimizer

def get_default_nezha_optimizer(
        module,
        lr: float = 2e-5,  # 3e-5
        eps: float = 1e-6,
        correct_bias: bool = True,
        weight_decay: float = 1e-3,  # 1e-3
):
    no_decay = ["bias", "LayerNorm.weight"]
    other_params = ['lstm', "classifier", "global_pointer", 'tcn', 'tcn2']
    nezha_params = ["nezha"]
    is_main = nezha_params + no_decay
    param_group = [
        {'params': [p for n, p in module.named_parameters() if
                    any(nd in n for nd in no_decay) and any(nd in n for nd in nezha_params)], 'weight_decay': 0,
         'lr': 5e-5},
        {'params': [p for n, p in module.named_parameters() if
                    not any(nd in n for nd in no_decay) and any(nd in n for nd in nezha_params)],
         'weight_decay': weight_decay, 'lr': 5e-5},
        {'params': [p for n, p in module.named_parameters() if not any(nd in n for nd in is_main)],
         'weight_decay': weight_decay, 'lr': 1e-3},
        {'params': [p for n, p in module.named_parameters() if
                    any(nd in n for nd in no_decay) and not any(nd in n for nd in nezha_params)],
         'weight_decay': 0, 'lr': 1e-3},
    ]
    optimizer = AdamW(param_group,
                      # lr=lr,
                      eps=eps,
                      correct_bias=correct_bias)
    # weight_decay=weight_decay)
    return optimizer


# optimizer = get_default_model_optimizer(dl_module)



optimizer = get_default_nezha_optimizer(dl_module)
# optimizer = Lookahead(optimizer=optimizer,k=5,alpha=0.5)

schedule = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps= len(datalist)/batch_size,
                                           num_training_steps=20 * len(datalist) / batch_size)

from torch.utils.data import DataLoader
from ark_nlp.factory.optimizer import get_optimizer


import torch.nn.functional as F

from nezha_code.Task_FGM_PGD import AttackTask_FGM, AttackTask_PGD, AttackTask_AWP, AttackTask_PGD_AWP


#### 在这个地方尝试直接加入rdrop进行模型的训练，看看效果能否有提升


model = AttackTask_FGM(dl_module, optimizer, 'gpce', scheduler=schedule, cuda_device=0, ema_decay=0.999)
for name, para in model.module.named_parameters():
    print(name)



model.fit(ner_train_dataset,
          ner_dev_dataset,
          # lr=2e-5, 此时已经不需要这个学习率了
          epochs=num_epoches,
          batch_size=batch_size
         )


torch.save(model.module.state_dict(), 'train_fgm_2.pth')


################ 练五个模型，对训练数据进行重新的错标漏标排查############