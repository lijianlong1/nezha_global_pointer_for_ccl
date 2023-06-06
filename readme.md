# 古籍文本命名实体识别任务
## 项目方案
> 单折单模（细致调参），使用nezha-cn-base-Chinese作为模型的编码器，然后加上两层时序卷积神经网络，  
> 再加上FGM对抗学习，再加上全局指针网络对输出进行解码，再加上小量规则数据
## 主要工作内容及创新
> 1.构建了一套基于史藏文本的数据，用于模型的预训练，该数据一共有39万条古文文本。  
> 2.提出了一种全新的nezha-tcn预训练语言模型，在模型的构造上通过引入时序卷积神经网络，关注文本的局部特征信息，挖掘模型的实体识别潜力

## 项目整体流程方法
> 1.古文文本的搜集与预处理：得到用于预训练的古文文本数据  
> 2.nezha模型加入两层TCN使用39万条古文文本进行继续预训练  
> 3.训练数据的预处理与格式化标注：将训练数据进行处理，剔除无实体标注文本，分割训练集和测试集  
> 4.加入FGM进行模型的训练与参数的调整  
> 5.预测公榜上提供的测试集  
## 项目执行代码运行流程

> 1.（跳过这一步，数据已开源）运行data_set文件夹下面的data_file_for_trannezha.py（由于该部分还想训练一个所有古文的kongzi预训练模型，直接提交数据）  在同目录下得到：data_24shi_for_pre_train.txt用于模型的预训练，数据直接开源，  
若想从头开始进行数据处理，需要下载相关的史藏数据集进行数据的处理,需要到百度网盘下载，所有项目格式及数据集：  
链接：https://pan.baidu.com/s/1p5s0FbrwlF3fsVd3ZQWowA  
提取码：wvlm  
> 2.运行data_set文件夹下面的data_for_bio_sentence.py 在同目录下得到：data_bio_delete用于模型的训练 
> 3.运行pre_train_nezha文件夹下面的pretrain_code_nezha_best.py（训练时长在50个小时左右） 在同目录下得到：mynezha文件夹，存储各个step下面的预训练nezha-tcn模型  
> 4.运行nezha_code文件夹下面的nezha_global_fgm_train.py 在同目录下得到：train_fgm_2.pth模型  
> 5.运行nezha_code文件夹下面的nerpre_test.py 在同目录下得到：result.txt文件，直接提交评分  


## 代码运行主要关键环境
jieba==0.42.1  
transformers==4.7.0  
numpy==1.20.3  
tqdm==4.62.2  
torch==1.13.1  
six==1.16.0  
pandas==1.3.3
ark-nlp==0.0.9
zhconv==1.4.3


## 由于github不能上传文件大小高于100m的文件，因此还需要下载，预训练语言模型，和其他的一些模型与数据
> 1.nezha-cn-base模型需要下载，请移步到百度网盘： 
链接：https://pan.baidu.com/s/1Er-EKCggygZmuhcGvByu3w 
提取码：pn2y  
下载完成后将.bin后缀模型，放到nezha-cn-base文件夹下  
> 2.只需要查看榜上模型的效果，需要下载train_fgm_2.pth模型，移步到百度网盘:
链接：https://pan.baidu.com/s/1wVBhfhTNP-06azJkaFhllQ   
提取码：c5sb  
并将模型文件放到nezha_code文件夹下,然后直接运行运行nezha_code文件夹下面的nerpre_test.py，得到result.txt
>
>
>>联系方式：1436631592@qq.com
