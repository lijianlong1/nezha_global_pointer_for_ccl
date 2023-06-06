# -*- ecoding: utf-8 -*-
# @ModuleName: data_process_for_ner
# @Function: 注意相应的细节训练数据处理，对于超长的文本句子，我们需要对其进行分割处理
# 训练数据制作trick
# trick1：对于训练数据中的错标漏标的情况，对训练数据按照实体和标号进行rematch
# trick2：对于文本中长度超过480的进行相应的数据切分，将训练数据和预测数据的长度都限制在max-len之内
# @Author: long
# @Time: 2022/10/2 14:52
# *****************conding***************
import json

data_out = open('../data_set/选手数据集/bio_output9257_2000_data.txt', 'w', encoding='utf-8')
# 文件数据条数实际上有
with open('../data_set/选手数据集/output9257_test_data.json', 'r', encoding='utf-8') as f:
    data_all = f.readlines()
    count = 0

    max_end = 0
    # 在文段中，对于相应的错标的情况进行重新处理，
    for i in data_all:
        try:
            data_i = json.loads(i)
            #if len(data_i['context']) <= 480:
            data_sentence = data_i['context']
            label = ['O'] * len(data_sentence)
            labels = data_i['entities']
            # entity_text = data_i['entities_text']

            # entity_text_all = []
            # for i in entity_text:
            #     entity_text_all.extend(entity_text[i])

            for label_ in labels:
                entity_type = label_['label']
                entity_list_one_all = label_['span']
                for span_one in entity_list_one_all:
                    if type(span_one) != list:
                        start, end = map(int, span_one.split(';'))
                    elif type(span_one) == list:
                        start, end = span_one[0], span_one[1]
                    # 对相关的数据进行rematch，并使用
                    # if end > 480:
                    #     print(data_sentence[start:end])
                    # print(end)
                    label[start] = 'B-' + str(entity_type)
                    label[start + 1: end] = ['I-' + str(entity_type)] * (end-start-1)
            max_len = 480
            #### 求句子里面有多少个max_len
            n_sentence = len(data_sentence)//max_len
            n_len = len(data_sentence)//(n_sentence + 1)  # 计算相应的数据结果
            count_len = 0
            for word, bio in zip(data_sentence, label):
                ######### 在这个地方进行文本长度和其他相应关系的统计########
                count_len += 1
                if count_len > n_len and (word != '。'):
                    data_out.write('\n')
                    count_len = 0
                data_out.write(word+' '+bio+'\n')
            data_out.write('\n')
            count += 1

        except Exception as e:
            print(count)
            print(data_sentence)
            print(e)
            pass
    print(count)
    print(max_end)
    data_out.close()


