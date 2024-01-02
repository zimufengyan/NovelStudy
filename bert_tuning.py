# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name :      lora_tuning
   Author :         zmfy
   DateTime :       2023/12/8 19:00
   Description :    
-------------------------------------------------
"""
import numpy as np
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import torch
import time

from dataset import get_loader
from bert_model import get_full_model

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
num_epoch = 10
lr, lr_gamma = 5e-4, 0.8
checkpoint_dir = "./checkpoints"
device = torch.device("cuda")
pretrained_dir = '../bert-base-chinese'

model = get_full_model("../bert-base-chinese", num_class, device,
                       checkpoint_dir + "/bert_fc_full_epoch_2.pth")
# model = BertModel.from_pretrained(pretrained_dir).to(device)
# model.final_fc = torch.nn.Linear(768, num_class)


def top_k_accuracy(output, target, k=1):
    with torch.no_grad():
        _, topk_indices = output.topk(k, dim=1, largest=True, sorted=True)
        topk_correct = topk_indices.eq(target.view(-1, 1).expand_as(topk_indices))
        topk_correct = topk_correct.float().sum()
        return topk_correct / target.size(0)


def val(model, loader):
    top_1 = []
    top_3 = []
    model.eval()
    for i, (input_ids, attention_mask, token_type_ids,
            labels) in enumerate(loader):
        with torch.no_grad():
            out = model(input_ids=input_ids.to(device),
                        attention_mask=attention_mask.to(device),
                        token_type_ids=token_type_ids.to(device)).detach().cpu()
        top_1.append(top_k_accuracy(out, labels, k=1).item())
        top_3.append(top_k_accuracy(out, labels, k=3).item())

    return top_1, top_3


# target_modules = [
#     'query', 'value', 'key', 'dense'
# ]
# peft_config = LoraConfig(
#     task_type=TaskType.SEQ_CLS, inference_mode=False,
#     target_modules=target_modules,
#     r=16, lora_alpha=32, lora_dropout=0.1
# )
# model = get_peft_model(model, peft_config)
# print(model.print_trainable_parameters())

print("[INFO] Loading dataset")
train_loader, val_loader, test_loader = get_loader("./data/full")

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
criterion = torch.nn.CrossEntropyLoss()
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=lr_gamma, step_size=1)

print("[INFO] Starting training")
losses, accuracies = [], []
val_top1s, val_top3s = [], []
total_iter = len(train_loader)

for j in range(num_epoch):
    model.train()
    st = time.time()
    total_loss, total_acc, total_num = 0, 0, 0
    for i, (input_ids, attention_mask, token_type_ids,
            labels) in enumerate(train_loader):
        labels = labels.to(device)
        out = model(input_ids=input_ids.to(device),
                    attention_mask=attention_mask.to(device),
                    token_type_ids=token_type_ids.to(device))
        loss = criterion(out, labels)  # 输出跟真实的labels计算loss
        loss.backward()   # 调用反向传播得到每个要更新参数的梯度
        optimizer.step()  # 每个参数根据上一步得到的梯度进行优化
        optimizer.zero_grad()  # 把上一步训练的每个参数的梯度清零
        total_loss += loss.item() * labels.size(0)
        total_num += labels.size(0)

        accuracy = top_k_accuracy(out, labels, 1).item()
        total_acc += accuracy * labels.size(0)
        if i % 10 == 0:
            print(f"\rIter {i}/{total_iter}: Train Loss {loss.item() :.4f}  Train Accuracy {accuracy :.4f}", end='')

    lr_scheduler.step()
    losses.append(total_loss / total_num)
    accuracies.append((total_acc / total_num))

    val_top1s, val_top3s = val(model, val_loader)
    val_top1, val_top3 = np.mean(val_top1s), np.mean(val_top3s)
    val_top1s.append(val_top1)
    val_top3s.append(val_top3)
    et = time.time()
    print(f"\rEpoch {j + 1}/{num_epoch} {et-st :.2f}sec: Train Loss {losses[j] :.4f}", end="")
    print(f"  Train Accuracy {accuracies[j] :.4f}  Val Top1 {val_top1 :.4f}, Val Top3 {val_top3 :.4f}")

print("[INFO] End training")

model_path = checkpoint_dir + "/bert_tuning.pth"
# model.save_pretrained(model_path)
torch.save(model.state_dict(), model_path)
print(f"[INFO] Success to save lora model to {model_path}")

test_top1s, test_top3s = val(model, test_loader)
test_top1 = np.mean(test_top1s)
test_top3 = np.mean(test_top3s)
print(f"Final Test: Top1 {test_top1 :.4f} Top3 {test_top3:.4f}")

