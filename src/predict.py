import joblib
import pandas as pd
from src.config import MODEL_DIR, RESULT_DIR
from src.data_loader import load_holdout

def predict():
    model = joblib.load(MODEL_DIR / "lgbm_model.pkl")
    df = load_holdout()
    X = df.drop(columns=["CustomerID"])

    proba = model.predict_proba(X)[:, 1]
    df["Churn_Prob"] = proba

    RESULT_DIR.mkdir(exist_ok=True)
    df[["CustomerID", "Churn_Prob"]].to_csv(
        RESULT_DIR / "holdout_predictions.csv",
        index=False
    )

if __name__ == "__main__":
    predict()