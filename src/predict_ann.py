import torch
import numpy as np

def predict_ann(
    model,
    scaler,
    X,
    threshold=0.5,
    device=None
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    X_scaled = scaler.transform(X)
    X_t = torch.tensor(X_scaled, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        y_proba = model(X_t).cpu().numpy().ravel()

    y_pred = (y_proba >= threshold).astype(int)
    return y_proba, y_pred