# Coding by long
# Datatime:2022/4/8 20:10
# Filename:model.py
# Toolby: PyCharm
# description:
# ______________coding_____________
import torch.nn as nn
from transformers import BertModel, BertConfig
from ark_nlp.nn.layer.global_pointer_block import GlobalPointer, EfficientGlobalPointer
from nezha_code.nezhaconfig import NeZhaConfig
from nezha_code.nezhamodel import NeZhaModel, NeZhaTcnModel, NeZhaTcnLstmModel
from nezha_code.TCN import TemporalConvNet
import torch

# from ner_for_laic.nezha_code.nezha_global_fgm_train_2000 import device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class globalpointer_NER(nn.Module):
    def __init__(self, nezha_path, n_head):
        super(globalpointer_NER, self).__init__()
        self.config = NeZhaConfig.from_pretrained(nezha_path)  # 导入模型超参数
        self.nezha = NeZhaModel.from_pretrained(pretrained_model_name_or_path=nezha_path,config=self.config)  # 加载预训练模型权重.使用albert轻量级模型进行实体识别训练预测
        self.lstm = nn.LSTM(768, 384, batch_first=True, num_layers=2, bidirectional=True)
        self.dropout = nn.Dropout(0.1)

        self.global_pointer = GlobalPointer(heads=n_head, head_size=64, hidden_size=768, RoPE=True)
        # self.fc = nn.Linear(self.config.hidden_size, classes)  # 直接分类
        # self.dropout = nn.Dropout(0.2)
        # self.relu = nn.ReLU()  # 使用相应的激活函数
        # self.softmax = nn.Softmax(-1) # 表示在最后一个维度进行变化

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, **kwargs):
        outputs = self.nezha(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )[0]
       # print((outputs.shape))
        sequence_output = outputs
        sequence_output = self.dropout(sequence_output)
        sequence_output, _ = self.lstm(sequence_output)
        #sequence_output, (_, _) = self.lstm(sequence_output)
        logits = self.global_pointer(sequence_output, mask=attention_mask)

        return logits   # shape(batch-size, 512, 4) 每次训练的时候，考虑到消耗的问题，每次训练的batch为1或者2就可以了，不然会溢出

    ##### 在这个地方加上tcn的思路

class globalpointer_tcn_NER(nn.Module):
    def __init__(self, nezha_path, n_head):
        super(globalpointer_tcn_NER, self).__init__()
        self.config = NeZhaConfig.from_pretrained(nezha_path)  # 导入模型超参数
        self.nezha = NeZhaModel.from_pretrained(pretrained_model_name_or_path=nezha_path,config=self.config)  # 加载预训练模型权重.使用albert轻量级模型进行实体识别训练预测
        # self.lstm = nn.LSTM(768, 384, batch_first=True, num_layers=2, bidirectional=True)
        # self.dropout = nn.Dropout(0.1)

        self.tcn = TemporalConvNet(num_inputs=768, num_channels=[768, 768])

        self.global_pointer = GlobalPointer(heads=n_head, head_size=64, hidden_size=768, RoPE=True)
        # self.fc = nn.Linear(self.config.hidden_size, classes)  # 直接分类
        # self.dropout = nn.Dropout(0.2)
        # self.relu = nn.ReLU()  # 使用相应的激活函数
        # self.softmax = nn.Softmax(-1) # 表示在最后一个维度进行变化

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, **kwargs):
        outputs = self.nezha(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )[0]
       # print((outputs.shape))
        sequence_output = outputs
        # sequence_output = self.dropout(sequence_output)
        # sequence_output, _ = self.lstm(sequence_output)
        # sequence_output, (_, _) = self.lstm(sequence_output)
        sequence_output = sequence_output.transpose(1, 2)
        # 将其送入到模型中
        sequence_output = self.tcn(sequence_output)
        sequence_output = sequence_output.transpose(1, 2)

        logits = self.global_pointer(sequence_output, mask=attention_mask)

        return logits   # shape(batch-size, 512, 4) 每次训练的时候，考虑到消耗的问题，每次训练的batch为1或者2就可以了，不然会溢出


class globalpointer_ro_NER(nn.Module):
    def __init__(self, nezha_path, n_head):
        super(globalpointer_ro_NER, self).__init__()
        self.config = NeZhaConfig.from_pretrained(nezha_path)  # 导入模型超参数
        self.nezha = NeZhaTcnModel.from_pretrained(pretrained_model_name_or_path=nezha_path,config=self.config)  # 加载预训练模型权重.使用albert轻量级模型进行实体识别训练预测
        # self.lstm = nn.LSTM(768, 384, batch_first=True, num_layers=2, bidirectional=True)
        # self.dropout = nn.Dropout(0.1)

        # self.tcn = TemporalConvNet(num_inputs=768, num_channels=[768, 768])

        self.global_pointer = GlobalPointer(heads=n_head, head_size=64, hidden_size=768, RoPE=True)
        # self.fc = nn.Linear(self.config.hidden_size, classes)  # 直接分类
        # self.dropout = nn.Dropout(0.2)
        # self.relu = nn.ReL U()  # 使用相应的激活函数
        # self.softmax = nn.Softmax(-1) # 表示在最后一个维度进行变化

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, **kwargs):
        outputs = self.nezha(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )[0]
       # print((outputs.shape))
        sequence_output = outputs
        # sequence_output = self.dropout(sequence_output)
        # sequence_output, _ = self.lstm(sequence_output)
        # sequence_output_lstm, (_, _) = self.lstm(sequence_output)
        # sequence_output = sequence_output.transpose(1, 2)
        # # 将其送入到模型中
        # sequence_output = self.tcn(sequence_output)
        # sequence_output = sequence_output.transpose(1, 2)
        #### 在这个地方加上相应的cls，每一个词都进行变化，效果反而出现了下降
        #sequence_output_all = torch.cat([sequence_output, sequence_output_lstm], -1)

        logits = self.global_pointer(sequence_output, mask=attention_mask)

        return logits   # shape(batch-size, 512, 4) 每次训练的时候，考虑到消耗的问题，每次训练的batch为1或者2就可以了，不然会溢出


class globalpointer_tcn_tcn_f1_NER(nn.Module):
    def __init__(self, nezha_path, n_head):
        ########## 加入f1值优化的思想，直接将得到的最优矩阵构造为权重比例最好的矩阵######
        ########## logist-adjustment ##########直接在训练过程中进行相应的处理


        super(globalpointer_tcn_tcn_f1_NER, self).__init__()
        self.config = NeZhaConfig.from_pretrained(nezha_path)  # 导入模型超参数
        self.nezha = NeZhaTcnModel.from_pretrained(pretrained_model_name_or_path=nezha_path,config=self.config)  # 加载预训练模型权重.使用albert轻量级模型进行实体识别训练预测
        # self.lstm = nn.LSTM(768, 384, batch_first=True, num_layers=2, bidirectional=True)
        # self.dropout = nn.Dropout(0.1)

        self.tcn = TemporalConvNet(num_inputs=768, num_channels=[768, 768])

        # 后续考虑使用相应的efficient-globalpointer进行实验，可能会出现指标高一些的情况
        self.global_pointer = GlobalPointer(heads=n_head, head_size=64, hidden_size=768, RoPE=True)

        # self.global_pointer = GlobalPointer(heads=n_head, head_size=64, hidden_size=768, RoPE=True)
        # self.fc = nn.Linear(self.config.hidden_size, classes)  # 直接分类
        # self.dropout = nn.Dropout(0.2)
        # self.relu = nn.ReLU()  # 使用相应的激活函数
        # self.softmax = nn.Softmax(-1) # 表示在最后一个维度进行变化
    def compute_logist(self):
        """
        读取数据集，对相应的数据进行统计
        :return:
        """
        hash_map_dict = {}
        with open('/home/lijianlong/gaiic/ner_for_laic/data_set/选手数据集/train_all.json', 'r', encoding='utf-8') as t:
            data_all = t.readlines()
            for i in data_all:
                i = eval(i)
                entities = i['entities_text']
                for entity_type in entities:
                    if entity_type not in hash_map_dict:
                        hash_map_dict[entity_type] = entities[entity_type]
                    else:
                        hash_map_dict[entity_type].extend(entities[entity_type])
        dict_new = sorted(hash_map_dict.items(), key=lambda d: d[0])
        list_new = []
        sum_all = 0
        for i in dict_new:
            list_new.append(float(len(i[1])))
            sum_all += float(len(i[1]))
        all_list = []
        for i in list_new:
            new_i = i / sum_all
            all_list.append(new_i)
        all_list.append(list_new[-1])
        all_list_tensor = torch.tensor(all_list)

        adjustment = torch.log(all_list_tensor ** 0.2 + 1e-12)
        # adjustment = torch.softmax(adjustment, )
        # adjustment = torch.from_numpy(adjustment)
        # adjustment = adjustment  # .to(args.device)
        return adjustment





    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, **kwargs):
        outputs = self.nezha(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )[0]
       # print((outputs.shape))
        sequence_output = outputs
        # sequence_output = self.dropout(sequence_output)
        # sequence_output, _ = self.lstm(sequence_output)
        # sequence_output, (_, _) = self.lstm(sequence_output)
        sequence_output = sequence_output.transpose(1, 2)
        # 将其送入到模型中
        sequence_output = self.tcn(sequence_output)
        sequence_output = sequence_output.transpose(1, 2)
        #### 在这个地方加上相应的cls，每一个词都进行变化，效果反而出现了下降
        # sequence_output = (sequence_output + sequence_output[:, 0, :].unsqueeze(1).repeat(1, sequence_output.size(1), 1))/2



        logits = self.global_pointer(sequence_output, mask=attention_mask)

        logist_adjustment = self.compute_logist()
        ### logist_adjustment,改成正数就可，
        logist_adjustment = torch.softmax(logist_adjustment,dim=-1).to(device)

        logits = logits + logist_adjustment.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(logits.size())


        return logits   # shape(batch-size, 13, 480, 480) 每次训练的时候，考虑到消耗的问题，每次训练的batch为1或者2就可以了，不然会溢出


class globalpointer_tcn_lstm_NER(nn.Module):
    def __init__(self, nezha_path, n_head):
        super(globalpointer_tcn_lstm_NER, self).__init__()
        self.config = NeZhaConfig.from_pretrained(nezha_path)  # 导入模型超参数
        self.nezha = NeZhaModel.from_pretrained(pretrained_model_name_or_path=nezha_path,
                                                config=self.config)  # 加载预训练模型权重.使用albert轻量级模型进行实体识别训练预测
        # self.lstm = nn.LSTM(768, 384, batch_first=True, num_layers=2, bidirectional=True)
        # self.dropout = nn.Dropout(0.1)

        self.tcn = TemporalConvNet(num_inputs=768, num_channels=[768, 768])

        self.tcn2 = TemporalConvNet(num_inputs=768, num_channels=[768, 768])


        # self.tcn = TemporalConvNet(num_inputs=768, num_channels=[768, 768])

        self.global_pointer = GlobalPointer(heads=n_head, head_size=64, hidden_size=768, RoPE=True)
        # self.fc = nn.Linear(self.config.hidden_size, classes)  # 直接分类
        # self.dropout = nn.Dropout(0.2)
        # self.relu = nn.ReLU()  # 使用相应的激活函数
        # self.softmax = nn.Softmax(-1) # 表示在最后一个维度进行变化

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, **kwargs):
        outputs = self.nezha(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )[0]
       # print((outputs.shape))
        sequence_output = outputs
        #
        # sequence_output, _ = self.lstm(sequence_output)

        # sequence_output = self.dropout(sequence_output)
        # sequence_output, _ = self.lstm(sequence_output)
        # sequence_output, (_, _) = self.lstm(sequence_output)
        sequence_output = sequence_output.transpose(1, 2)
        # 将其送入到模型中
        sequence_output = self.tcn(sequence_output)

        sequence_output = self.tcn2(sequence_output)
        sequence_output = sequence_output.transpose(1, 2)



        # sequence_output = sequence_output.transpose(1, 2)
        # # 将其送入到模型中
        # sequence_output = self.tcn(sequence_output)
        # sequence_output = sequence_output.transpose(1, 2)
        #### 在这个地方加上相应的cls，每一个词都进行变化，效果反而出现了下降
        # sequence_output = (sequence_output + sequence_output[:, 0, :].unsqueeze(1).repeat(1, sequence_output.size(1), 1))/2

        logits = self.global_pointer(sequence_output, mask=attention_mask)

        return logits   # shape(batch-size, 512, 4) 每次训练的时候，考虑到消耗的问题，每次训练的batch为1或者2就可以了，不然会溢出


class globalpointer_tcn_lstmpre_NER(nn.Module):
    def __init__(self, nezha_path, n_head):
        super(globalpointer_tcn_lstmpre_NER, self).__init__()
        self.config = NeZhaConfig.from_pretrained(nezha_path)  # 导入模型超参数
        self.nezha = NeZhaTcnLstmModel.from_pretrained(pretrained_model_name_or_path=nezha_path,config=self.config)  # 加载预训练模型权重.使用albert轻量级模型进行实体识别训练预测
        # self.lstm = nn.LSTM(768, 384, batch_first=True, num_layers=2, bidirectional=True)
        # self.nezha = NeZhaTcnModel.from_pretrained(pretrained_model_name_or_path=nezha_path, config=self.config)

        # self.dropout = nn.Dropout(0.1)

        # self.tcn = TemporalConvNet(num_inputs=768, num_channels=[768, 768])

        self.global_pointer = GlobalPointer(heads=n_head, head_size=64, hidden_size=768, RoPE=True)

        # self.global_pointer = GlobalPointer(heads=n_head, head_size=64, hidden_size=768, RoPE=True)
        # self.fc = nn.Linear(self.config.hidden_size, classes)  # 直接分类
        # self.dropout = nn.Dropout(0.2)
        # self.relu = nn.ReLU()  # 使用相应的激活函数
        # self.softmax = nn.Softmax(-1) # 表示在最后一个维度进行变化

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, **kwargs):
        outputs = self.nezha(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )[0]
        # print((outputs.shape))
        sequence_output = outputs
        # sequence_output = self.dropout(sequence_output)
        # # sequence_output, _ = self.lstm(sequence_output)
        # # sequence_output, (_, _) = self.lstm(sequence_output)
        # sequence_output = sequence_output.transpose(1, 2)
        # # 将其送入到模型中
        # sequence_output = self.tcn(sequence_output)
        # sequence_output = sequence_output.transpose(1, 2)
        #### 在这个地方加上相应的cls，每一个词都进行变化，效果反而出现了下降
        # sequence_output = (sequence_output + sequence_output[:, 0, :].unsqueeze(1).repeat(1, sequence_output.size(1), 1))/2
        logits = self.global_pointer(sequence_output, mask=attention_mask)

        return logits   # shape(batch-size, 512, 4) 每次训练的时候，考虑到消耗的问题，每次训练的batch为1或者2就可以了，不然会溢出