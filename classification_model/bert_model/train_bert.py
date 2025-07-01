import os

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

from transformers import get_scheduler
from tqdm import tqdm

from utils.load_config import load_config

base_model_name = "bert-base-chinese"
config = load_config()
save_models = config.get('save_models')
bert_dir = "bert_dir"


def train_and_save_bert_model(train_loader, model_name, device):
    tokenizer = BertTokenizer.from_pretrained(base_model_name)
    model = BertForSequenceClassification.from_pretrained(base_model_name, num_labels=2).to(device)

    # 优化器 & 学习率调度器
    optimizer = AdamW(model.parameters(), lr=2e-5)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer,
                                 num_warmup_steps=0,
                                 num_training_steps=len(train_loader) * 3)

    # 🔁 开始训练
    num_epochs = 16
    model.train()
    for epoch in range(num_epochs):
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for batch in loop:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["label"])

            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            loop.set_postfix(loss=loss.item())

    save_bert_full_dir = os.path.join(save_models, "embedding_dir")
    os.makedirs(save_bert_full_dir, exist_ok=True)

    # ✅ 保存模型和 tokenizer
    model_save_path = os.path.join(save_bert_full_dir, "bert", model_name)
    os.makedirs(model_save_path, exist_ok=True)

    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    print(f"模型和分词器已保存到: {model_save_path}")

    return tokenizer, model


def load_bert_model(model_name, device):
    save_bert_full_dir = os.path.join(save_models, "embedding_dir")
    os.makedirs(save_bert_full_dir, exist_ok=True)
    model_save_path = os.path.join(save_bert_full_dir, "bert", model_name)
    os.makedirs(model_save_path, exist_ok=True)
    # 拼接保存路径
    model_path = os.path.join(save_bert_full_dir, "bert", model_name)

    # 判断路径是否存在
    if not os.path.exists(model_path, device):
        return None, None

    # 加载 tokenizer 和 model
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path).to(device)

    print(f"模型和分词器已从 {model_path} 加载成功")
    return tokenizer, model
