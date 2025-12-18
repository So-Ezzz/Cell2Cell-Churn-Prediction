import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class DeepANN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 250),
            nn.ReLU(),
            nn.Linear(250, 250),
            nn.ReLU(),
            nn.Linear(250, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
class ANNWrapper:
    """
    Wrap PyTorch ANN to behave like sklearn model
    """

    def __init__(self, model, scaler, device=None):
        self.model = model
        self.scaler = scaler
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def predict_proba(self, X):
        """
        返回 shape = (n_samples, 2)
        [:, 1] 是 churn=1 的概率
        """
        X_scaled = self.scaler.transform(X)
        X_t = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            probs_1 = self.model(X_t).cpu().numpy().reshape(-1)

        probs_0 = 1.0 - probs_1
        return np.column_stack([probs_0, probs_1])

def train_deep_ann(
    X_train,
    y_train,
    X_valid,
    y_valid,
    epochs=300,
    batch_size=32,
    lr=1e-3,
    patience=10,
    device=None
):
    """
    Deep-BP-ANN using PyTorch (paper-aligned)
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ===== 1. MinMaxScaler（论文一致）=====
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)

    # ===== 2. 转为 Tensor =====
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)

    X_valid_t = torch.tensor(X_valid, dtype=torch.float32)
    y_valid_t = torch.tensor(y_valid.values, dtype=torch.float32).unsqueeze(1)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True
    )

    # ===== 3. 初始化模型 =====
    model = DeepANN(input_dim=X_train.shape[1]).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ===== 4. Early Stopping =====
    best_val_loss = float("inf")
    patience_counter = 0

    # ===== 5. 训练循环 =====
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # ===== 验证 =====
        model.eval()
        with torch.no_grad():
            val_preds = model(X_valid_t.to(device))
            val_loss = criterion(val_preds, y_valid_t.to(device)).item()

        # ===== Early stopping =====
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    # 恢复最佳权重
    model.load_state_dict(best_state)

    model_wrapper = ANNWrapper(model, scaler, device=device)

    return model_wrapper