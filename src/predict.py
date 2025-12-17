import pandas as pd
from pathlib import Path

def predict_and_save(
    model,
    df_test: pd.DataFrame,
    feature_cols: list,
    id_col: str,
    output_path: Path,
    threshold: float = 0.5
):
    """
    使用训练好的模型进行预测并保存结果
    """
    X_test = df_test[feature_cols]

    # 预测概率
    y_proba = model.predict_proba(X_test)[:, 1]

    # 二值化
    y_pred = (y_proba >= threshold).astype(int)

    result_df = pd.DataFrame({
        id_col: df_test[id_col],
        "Churn": y_pred
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)

    return result_df