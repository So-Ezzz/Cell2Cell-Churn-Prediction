from src.train import train_xgboost, train_logistic, train_random_forest
from src.evaluate import evaluate_models
from src.data_loader import load_train
from src.preprocess import preprocess_data
from sklearn.model_selection import train_test_split

df = load_train()
df = preprocess_data(df)

target = "是否流失"
X = df.drop(columns=[target, "客户唯一标识", "服务区域"])
y = df[target]

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Logistic Regression": train_logistic(X_train, y_train),
    "Random Forest": train_random_forest(X_train, y_train),
    "XGBoost": train_xgboost(X_train, y_train)
}

results = evaluate_models(models, X_valid, y_valid)