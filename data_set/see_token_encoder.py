# -*- ecoding: utf-8 -*-
# @ModuleName: see_token_encoder
# @Function: 查看相关的古文数据是否能够对繁体字进行编码
# @Author: long
# @Time: 2023/4/24 16:57
# *****************conding***************


import os
from data_set.func_trans_easy import hant_2_fanti
path = "史藏/正史" #文件夹目录
files= os.listdir(path) #得到文件夹下的所有文件名称
print(files)

fold_file = os.walk('史藏')
print(fold_file)
for root, dir, file in fold_file:
    print(dir)