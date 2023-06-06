# -*- ecoding: utf-8 -*-
# @ModuleName: func_trans_easy
# @Function: 简体字和繁体字之间的相互转化
# @Author: long
# @Time: 2023/4/24 14:53
# *****************conding***************

import zhconv


from transformers import BertTokenizer


tokenizer = BertTokenizer(vocab_file='/home/lijianlong/gaiic/nezha-cn-base/vocab.txt', max_len=128)

def hant_2_hans(hant_str: str):
    '''
    Function: 将 hant_str 由繁体转化为简体
    '''
    return zhconv.convert(hant_str, 'zh-hans')

def hant_2_fanti(hant_str: str):
    '''
    Function: 将 hant_str 由繁体转化为简体
    '''
    return zhconv.convert_for_mw(hant_str, 'zh-hant')

print(hant_2_hans("祥等並勸護廢帝。時綱總領禁兵，護乃遣綱入宮，召鳳等議事，及出，以次執送護第。因罷散宿衛兵，遣祥逼帝，幽於舊邸。"))
print(tokenizer.encode(hant_2_fanti("祥等并劝护废帝。时纲总领禁兵，“护乃遣纲入宫”，召凤等议事，及出，以次执送护第。因罢散宿卫兵，遣祥逼帝，幽于旧邸。")))


# with open('史藏/正史/元史.txt', 'r', encoding='utf-8') as f:
#     data = f.readlines()
#     for i in data:
#         line_ = i.strip('\n').strip()
#
#
#         # line_ = hant_2_fanti(line_)
#         if len(line_) > 5 and len(line_) < 128 and 'i' not in line_:
#             text_save = line_
#
#             print(text_save)
#
#         elif len(line_) > 128 and 'i' not in line_:
#             list_all = len(line_) // 128 + 1
#             list_text = line_.split('。')  # 只使用句号将其划分
#             ##### 计算每一个存储的text有多少个独立的句子
#             sentence_tem = ''
#             for i in list_text:
#                 if len(sentence_tem) > 100:
#                     text_save = sentence_tem
#
#                     print(text_save)
#
#                     sentence_tem = ''
#                 else:
#                     sentence_tem += i
#             text_save_2 = sentence_tem
#             print(text_save_2)
#
#
