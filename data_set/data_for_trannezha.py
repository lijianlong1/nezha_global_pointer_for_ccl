# -- coding: utf-8 --
# @Time : 2023/6/5 14:31
# @Author : long
# @Site : 人工智能研究所
# @File : data_for_trannezha.py
# @Software: PyCharm



import os
from data_set.func_trans_easy import hant_2_fanti
path = "史藏/正史" #文件夹目录
files= os.listdir(path) #得到文件夹下的所有文件名称
print(files)

data_for_24shi_ = open('data_24shi_for_pre_train.txt', 'w',encoding='utf-8')

count = 0
for i in files:
    file_here = '史藏/正史/' + i
    with open(file_here, 'r',encoding='utf-8') as f:
        data_lines = f.readlines()
        for line_ in data_lines:
            line_ = line_.strip('\n').strip()

            line_ = hant_2_fanti(line_)
            if len(line_)>5 and len(line_)<128 and 'i' not in line_:
                text_save = line_
                data_for_24shi_.write(text_save + '\n')
                print(text_save)
                count += 1
            elif len(line_) > 128 and 'i' not in line_:
                list_all = len(line_)//128 +1
                list_text = line_.split('。') # 只使用句号将其划分
                ##### 计算每一个存储的text有多少个独立的句子
                sentence_tem = ''
                for i in list_text:
                    if len(sentence_tem)>96:
                        text_save = sentence_tem
                        data_for_24shi_.write(text_save + '。\n')
                        print(text_save)
                        count += 1
                        sentence_tem = ''
                    else:
                        sentence_tem += i

                # text_save_2 = sentence_tem  # 最后面那句扔掉
                # if len(text_save_2)>5:
                #     print(text_save_2)
                #     data_for_24shi_.write(text_save_2 + '\n')
                #     count+=1
            # if count==20:
            #     break

data_for_24shi_.close()






