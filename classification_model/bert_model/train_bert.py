import os
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm import tqdm
from datasets import load_dataset
from utils.load_config import load_config
from torch.utils.data import DataLoader

base_model_name = "bert-base-chinese"
config = load_config()
save_models = config.get('save_models')
bert_dir = "bert_dir"
tokenizer = BertTokenizer.from_pretrained(base_model_name)


def encode_bert_tensor(dataset_name):
    dataset = load_dataset(dataset_name)

    def tokenize_function(example):
        return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

    # 分词 + 数据集准备
    tokenized_ds = dataset.map(tokenize_function, batched=True)
    tokenized_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    # 构建 DataLoader
    train_loader = DataLoader(tokenized_ds["train"], batch_size=16, shuffle=True)
    test_loader = DataLoader(tokenized_ds["test"], batch_size=16)
    return train_loader, test_loader


def train_and_save_bert_model(train_loader, model_name, device):
    # tokenizer = BertTokenizer.from_pretrained(base_model_name)
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

    return model


def load_bert_model(model_name, device):
    save_bert_full_dir = os.path.join(save_models, "embedding_dir")
    model_path = os.path.join(save_bert_full_dir, "bert", model_name)

    # 如果路径不存在，直接返回 None（不要先创建）
    if not os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
        return None

    # 加载模型
    model = BertForSequenceClassification.from_pretrained(model_path).to(device)
    print(f"模型已从 {model_path} 加载成功")
    return model
