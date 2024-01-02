# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name :      dataset.py
   Author :         zmfy
   DateTime :       2023/12/8 19:04
   Description :    
-------------------------------------------------
"""
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
import unicodedata as ucd

mapping = {
    "玄幻": 0,
    "奇幻": 1,
    "武侠": 2,
    "仙侠": 3,
    "都市": 4,
    "历史": 5,
    "游戏": 6,
    "科幻": 7,
    "奇闻异事": 8,
    "古代言情": 9,
    "现代言情": 10,
    "幻想言情": 11,
}
mapping_rec = {val: key for key, val in mapping.items()}
num_class = 12
tokenizer = BertTokenizer.from_pretrained('../bert-base-chinese')  # 要跟预训练模型相匹配


class CustomDataset(Dataset):
    def __init__(self, excel_file):
        self.dataset = pd.read_excel(excel_file).values.tolist()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx][2].strip()
        text = ucd.normalize('NFKC', text).replace(" ", "")
        text = text.replace("\n", "").replace("……", "").replace("......", "")
        label = mapping[self.dataset[idx][1]]

        return text, label


def collate_fn(data):
    sents = [i[0] for i in data]
    labels = [i[1] for i in data]

    # 编码
    data = tokenizer.batch_encode_plus(batch_text_or_text_pairs=sents,
                                       truncation=True,  # 当句子长度大于max_length时，截断
                                       padding='max_length',  # 一律补0到max_length长度
                                       max_length=500,
                                       return_tensors='pt',  # 返回pytorch类型的tensor
                                       return_length=True)  # 返回length，标识长度

    input_ids = data['input_ids']  # input_ids:编码之后的数字
    attention_mask = data['attention_mask']  # attention_mask:补零的位置是0,其他位置是1
    token_type_ids = data['token_type_ids']  # 第一个句子和特殊符号的位置是0，第二个句子的位置是1(包括第二个句子后的[SEP])
    labels = torch.LongTensor(labels)

    return input_ids, attention_mask, token_type_ids, labels


def get_loader(data_dir, split='full', batch_size=16, pin_memory=True, num_workers=4, drop_last=False):
    train_dir = os.path.join(data_dir, "train.xlsx")
    val_dir = os.path.join(data_dir, "val.xlsx")
    test_dir = os.path.join(data_dir, "test.xlsx")

    if split == "train":
        train_dataset = CustomDataset(train_dir)
        train_loader = init_loader(train_dataset, batch_size, pin_memory, num_workers, drop_last)
        return train_loader
    elif split == 'val':
        val_dataset = CustomDataset(val_dir)
        val_loader = init_loader(val_dataset, batch_size, pin_memory, num_workers, drop_last)
        return val_loader
    elif split == 'test':
        test_dataset = CustomDataset(test_dir)
        test_loader = init_loader(test_dataset, batch_size, pin_memory, num_workers, drop_last)
        return test_loader
    else:
        train_dataset = CustomDataset(train_dir)
        val_dataset = CustomDataset(val_dir)
        test_dataset = CustomDataset(test_dir)
        train_loader = init_loader(train_dataset, batch_size, pin_memory, num_workers, drop_last)
        val_loader = init_loader(val_dataset, batch_size, pin_memory, num_workers, drop_last)
        test_loader = init_loader(test_dataset, batch_size, pin_memory, num_workers, drop_last)
        return train_loader, val_loader, test_loader


def init_loader(dataset: Dataset, batch_size=16, pin_memory=True, num_workers=4, drop_last=False):
    return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=True,
            pin_memory=pin_memory,
            num_workers=num_workers,
            drop_last=drop_last
        )
