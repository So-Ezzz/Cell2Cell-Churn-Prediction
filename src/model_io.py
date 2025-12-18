import joblib
import torch
import json
from pathlib import Path
from datetime import datetime
from src.train_ann import DeepANN, ANNWrapper


def load_best_model(model_dir="models", feature_dim=None, device=None):
    """
    Load best model saved in models/
    Automatically detect model type via meta.json

    Returns:
        model: sklearn / xgboost model OR ANNWrapper
        meta: dict (model info)
    """
    model_dir = Path(model_dir)

    # ===== 1. 读取 meta =====
    meta_path = model_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError("meta.json not found in models/")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    model_name = meta["model"]

    # ===== 2. 加载模型 =====
    if model_name in {"XGBoost", "Random Forest", "Logistic Regression"}:
        model_path = model_dir / "best_model.pkl"
        model = joblib.load(model_path)

    elif model_name == "Deep ANN":
        if feature_dim is None:
            raise ValueError("feature_dim must be provided for Deep ANN")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # 加载 scaler
        scaler = joblib.load(model_dir / "best_model_scaler.pkl")

        # 重建网络
        ann = DeepANN(input_dim=feature_dim).to(device)
        ann.load_state_dict(
            torch.load(model_dir / "best_model.pt", map_location=device)
        )

        model = ANNWrapper(ann, scaler, device=device)

    else:
        raise ValueError(f"Unknown model type: {model_name}")

    return model, meta
    
def save_best_model(
    model,
    model_name,
    feature_cols,
    best_threshold,
    cv_results=None,
    output_dir="models"
):
    """
    Save best model + meta information for reproducibility
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # ===== 1. 保存模型本体 =====
    if model_name in {"XGBoost", "Random Forest", "Logistic Regression"}:
        model_path = output_dir / "best_model.pkl"
        joblib.dump(model, model_path)

    elif model_name == "Deep ANN":
        # model 是 ANNWrapper
        torch.save(model.model.state_dict(), output_dir / "best_model.pt")
        joblib.dump(model.scaler, output_dir / "best_model_scaler.pkl")

    else:
        raise ValueError(f"Unknown model type: {model_name}")

    # ===== 2. 组织 meta 信息 =====
    meta = {
        "model": model_name,
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "feature_cols": feature_cols,
        "best_threshold": float(best_threshold),
    }

    if cv_results is not None:
        meta["cv"] = {
            "f1_mean": float(cv_results["F1"]),
            "f1_std": float(cv_results.get("F1_std", 0.0)),
        }

    # ===== 3. 写 meta.json =====
    meta_path = output_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"💾 Saved model to {output_dir}")
    print(f"🧾 Saved meta info to {meta_path}")