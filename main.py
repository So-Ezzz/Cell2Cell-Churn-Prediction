from src.train import train_xgboost, train_logistic, train_random_forest
from src.evaluate import evaluate_models_cv, find_best_threshold
from src.data_loader import load_train, load_holdout
from src.preprocess import preprocess_data
from sklearn.model_selection import train_test_split
from src.predict import predict_and_save
from src.train_ann import train_deep_ann
from pathlib import Path
from datetime import datetime

target = "Churn"
id_col = "CustomerID"

df = preprocess_data(load_train(),is_train=True)

X = df.drop(columns=[target, id_col])
y = df[target]

feature_cols = X.columns.tolist()

model_builders = {
    "Logistic Regression": train_logistic,
    "Random Forest": train_random_forest,
    "XGBoost": train_xgboost,
    "Deep ANN": train_deep_ann
}

results = evaluate_models_cv(
    model_builders,
    X,
    y,
    n_splits=5
)

# ===== 1. 选出最优模型名称（来自 CV）=====
best_model_name = max(results, key=lambda k: results[k]["F1"])
best_f1 = results[best_model_name]["F1"]

print(f"\n🏆 Best model selected by CV: {best_model_name}")

# ===== 2. 用全量训练数据重新训练最终模型 =====
if best_model_name == "XGBoost":
    best_model = train_xgboost(X, y)
elif best_model_name == "Random Forest":
    best_model = train_random_forest(X, y)
elif best_model_name == "Logistic Regression":
    best_model = train_logistic(X, y)
elif best_model_name == "Deep ANN":
    best_model = train_deep_ann(X, y, X, y)  # 用全量（不早停也可）
else:
    raise ValueError("Unknown model")

# ===== 3. 在训练集上重新搜索 threshold（最终版）=====
best_threshold, _ = find_best_threshold(best_model, X, y)

print(f"🎯 Final threshold selected: {best_threshold:.2f}")

# ===== 4. 处理 holdout 数据 =====
df_holdout = preprocess_data(load_holdout(), is_train=False)

# ===== 5. 预测并保存 =====
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

output_file = (
    Path("results")
    / f"prediction_{timestamp}_{best_model_name.replace(' ', '_')}_F1_{best_f1:.4f}.csv"
)

predict_and_save(
    model=best_model,
    df_test=df_holdout,
    feature_cols=feature_cols,
    id_col=id_col,
    output_path=output_file,
    threshold=best_threshold
)

print(f"📁 Prediction saved to: {output_file}")