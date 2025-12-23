import pandas as pd
from pathlib import Path
import numpy as np

# 这里依据置信度进行预测并保存结果(提交更有把握的前60%部分)
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

    # ===== 2. 计算置信度 =====
    confidence = np.abs(y_proba - threshold)

    # ===== 3. 构建结果表 =====
    result_df = pd.DataFrame({
        id_col: df_test[id_col],
        "proba": y_proba,
        "confidence": confidence
    })

    # ===== 4. 按置信度排序，取前 0.6 =====
    result_df = result_df.sort_values(
        by="confidence",
        ascending=False
    )

    top_k = int(len(result_df) * 0.6)
    result_df = result_df.iloc[:top_k]

    # ===== 5. 二值化预测 =====
    result_df["Churn"] = (result_df["proba"] >= threshold).astype(int)

    # 只保留需要的列
    result_df = result_df[[id_col, "Churn"]]

    # ===== 2. 映射为 Yes / No =====
    result_df["Churn"] = result_df["Churn"].map({
        1: "Yes",
        0: "No"
    })
    result_df = result_df.sample(frac=1, random_state=42).reset_index(drop=True)

    half = len(result_df) // 2
    test1 = result_df.iloc[:half]
    test2 = result_df.iloc[half:half * 2]

    # # ===== 6. 保存 =====
    # output_path.parent.mkdir(parents=True, exist_ok=True)
    # result_df.to_csv(output_path, index=False)
    
    test1.to_csv("./results/第3组-test1.csv", index=False)
    test2.to_csv("./results/第3组-test2.csv", index=False)

    return result_df