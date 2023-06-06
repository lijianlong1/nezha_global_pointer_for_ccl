import os
import random
import torch
import numpy as np



os.environ['WANDB_MODE'] = 'dryrun'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'   # 直接在最前面加上相应的预训练指定代码
def set_seed(seed):
    """
    设置随机种子

    Args:
        seed (:obj:`int`): 随机种子
    """  # noqa: ignore flake8"

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)



from transformers import BertTokenizer, LineByLineTextDataset
from nezha_code.nezhamodel import NeZhaConfig, NeZhaForMaskedLM, NeZhaTcnForMaskedLM


tokenizer = BertTokenizer.from_pretrained('../nezha-cn-base')

from transformers import DataCollatorForLanguageModeling, DataCollatorForWholeWordMask

config = NeZhaConfig.from_pretrained('../nezha-cn-base')
# 不加时空膨胀卷积的预训练
# model = NeZhaForMaskedLM(config).from_pretrained('/opt/lijianlong/gaiic/nezha-cn-base')
# Num of parameters:  102487688   334467720
# 加入时空膨胀卷积的预训练，增加注意力之间的重要性交互
model = NeZhaTcnForMaskedLM(config).from_pretrained('../nezha-cn-base')
# Num of parameters:  107212424

# model = NeZhaTcnLstmForMaskedLM(config).from_pretrained('/opt/lijianlong/gaiic/nezha-cn-base')
# Num of parameters:  114302600

print('Num of parameters: ', model.num_parameters())

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)
# data_collator = DataCollatorForWholeWordMask(tokenizer=tokenizer, mlm=True, mlm_probability=0.2)
#### 0.35---->0.4


from transformers import Trainer, TrainingArguments

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path='../data_set/data_24shi_for_pre_train.txt',
    block_size=128  # maximum sequence length,#data_24shi_for_pre_train.txt'
)

print('No. of lines: ', len(dataset)) # No of lines in your datset

training_args = TrainingArguments(
    output_dir='mynezha',
    overwrite_output_dir=True,
    num_train_epochs=50,
    per_device_train_batch_size=32,
    save_steps=50000,
)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)
trainer.train()
trainer.save_model('mynezha')
