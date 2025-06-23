import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import os
import torch
import torch.nn.functional as F

from utils.load_config import load_config


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim1=128, hidden_dim2=64, output_dim=2):
        super(MLPClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, output_dim),
            nn.Softmax(dim=1)  # ✅ 添加 Softmax 层
        )

    def forward(self, x):
        return self.net(x)


config = load_config()
save_models = config.get('save_models')
embedding_dir = "embeddings"


def train_and_save_pytorch_model(X_train, y_train, model_name, device):
    print("Training PyTorch MLP model...")

    # 数据准备
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    input_dim = X_train_tensor.shape[1]
    model = MLPClassifier(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        model.train()
        total_loss = 0
        for xb, yb in dataloader:
            optimizer.zero_grad()
            output = model(xb)
            loss = F.cross_entropy(output, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1} Loss: {total_loss:.4f}")

    # 保存模型
    save_path = os.path.join(save_models, embedding_dir, f"{model_name}.pt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存到: {save_path}")
    return model
