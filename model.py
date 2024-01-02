# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name :      model.py
   Author :         zmfy
   DateTime :       2023/12/8 19:18
   Description :    
-------------------------------------------------
"""
import torch
from transformers import BertModel


# 定义下游任务模型
class Model(torch.nn.Module):
    def __init__(self, pretrained, num_class):
        super().__init__()
        self.pretrained = pretrained
        self.num_class = num_class
        self.fc = torch.nn.Linear(768, num_class)  # 单层网络模型，只包括了一个fc的神经网络

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = self.pretrained(input_ids=input_ids,  # 先拿预训练模型来做一个计算，抽取数据当中的特征
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids)

        # 把抽取出来的特征放到全连接网络中运算，且特征的结果只需要第0个词的特征(跟bert模型的设计方式有关。对句子的情感分类，只需要拿特征中的第0个词来进行分类就可以了)
        out = self.fc(out.last_hidden_state[:, 0])  # torch.Size([16, 768])

        # # 将softmax函数应用于一个n维输入张量，对其进行缩放，使n维输出张量的元素位于[0,1]范围内，总和为1
        # out = out.softmax(dim=1)

        return out


def get_full_model(pretrained_dir, num_classes, device, fc_checkpoints=None, only_head=False):
    pretrained = BertModel.from_pretrained(pretrained_dir).to(device)
    if only_head:
        for param in pretrained.parameters():
            param.requires_grad_(False)
    model = Model(pretrained, num_classes)
    if fc_checkpoints:
        model.fc.load_state_dict(torch.load(fc_checkpoints))
    return model.to(device)


if __name__ == '__main__':
    pretrained = BertModel.from_pretrained("../bert-base-chinese")
    print(pretrained)