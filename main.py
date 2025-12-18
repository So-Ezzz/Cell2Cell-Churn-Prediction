from src.train import train_xgboost, train_logistic, train_random_forest
from src.evaluate import evaluate_models
from src.data_loader import load_train, load_holdout
from src.preprocess import preprocess_data
from sklearn.model_selection import train_test_split
from src.predict import predict_and_save
from pathlib import Path
from datetime import datetime

target = "Churn"
id_col = "CustomerID"

df = preprocess_data(load_train(),is_train=True)

X = df.drop(columns=[target, id_col])
y = df[target]

feature_cols = X.columns.tolist()

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Logistic Regression": train_logistic(X_train, y_train),
    "Random Forest": train_random_forest(X_train, y_train),
    "XGBoost": train_xgboost(X_train, y_train)
}

results = evaluate_models(models, X_valid, y_valid)

best_model_name = max(results, key=lambda k: results[k]["F1"])
best_model = models[best_model_name]
best_f1 = results[best_model_name]["F1"]
best_threshold = results[best_model_name]["BestThreshold"]

print(f"\n🏆 Best model selected: {best_model_name}")


df_holdout = preprocess_data(load_holdout(), is_train=False)


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