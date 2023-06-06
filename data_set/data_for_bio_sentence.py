# -*- ecoding: utf-8 -*-
# @ModuleName: data_for_bio_sentence
# @Function: 将相关的标注数据进行处理，完成
# @Author: long
# @Time: 2023/4/24 10:46
# *****************conding***************

import re
# from data_set.func_trans_easy import hant_2_hans
max_len = 0

with open('GuNER2023_train.txt', 'r', encoding='utf-8') as f, open ('data_bio_delete', 'w', encoding='utf-8') as data_out, open ('data_sentence_jianti', 'w', encoding='utf-8') as data_snetence_out:
    data_line = f.readlines()
    for i in data_line:

        ############# 将数据中的繁体字转化为简体字
        # i = hant_2_hans(i)

        ### 向数据中加入相应的实体信息
        p = re.findall(r'({.*?})', i)
        ### 对每一个匹配的字符进行记录
        index_start = 0
        sentence_ = ''
        lable = []
        i = i.strip('\n')
        for entity in p:
            sentence = i[index_start:]
            index_i = sentence.index(entity)
            #### 还是直接
            sentence_in = sentence[0:index_i]

            entity_strip = entity.strip('{').strip('}')
            # list_sentence_one.append(sentence_in)
            ########### 在这里加上句子和句子的lable
            sentence_ += sentence_in
            lable_O = ['O'] * len(sentence_in)
            lable.extend(lable_O)
            # list_sentence_one.append(entity_strip)
            ############# 在这里加上句子和句子的lable
            enyity_, lable_word = entity_strip.split('|')
            sentence_ += enyity_
            lable_B = ['B-' + lable_word]
            lable.extend(lable_B)
            lable_I = ['I-' + lable_word]*(len(enyity_)-1)
            lable.extend(lable_I)
            index_start = index_start + len(entity) + len(sentence_in)



        sentence_ += i[index_start:]
        lable.extend(['O']*len(i[index_start:]))


        print(len(sentence_))
        if len(sentence_)> max_len:
            max_len = len(sentence_)


        data_snetence_out.write(sentence_ + '\n')

        ########## 将数据写入txt文件#########
        if lable==lable[::-1]:
            print(lable)
        else:
            for word_i, lable_j in zip(sentence_, lable):
                data_out.write(word_i + ' ' + lable_j + '\n')
            data_out.write('\n')

print(max_len)











