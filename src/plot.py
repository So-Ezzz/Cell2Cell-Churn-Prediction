import shap
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
matplotlib.rcParams['font.family'] = ['STFangsong']
plt.rcParams['axes.unicode_minus'] = False 

def plot_shap_summary(
    model,
    X,
    max_display_bar=15,
    max_display_beeswarm=30,
    model_output="raw"
):
    """
    Plot SHAP summary plots (bar + beeswarm) for tree-based models.

    Parameters
    ----------
    model : trained tree-based model (e.g. XGBoost)
    X : pandas.DataFrame
        Data used for SHAP explanation (e.g. validation set)
    max_display_bar : int
        Number of features in bar plot
    max_display_beeswarm : int
        Number of features in beeswarm plot
    model_output : str
        SHAP model_output, default="raw" (log-odds)
    """

    # ===== 1. 创建 SHAP explainer =====
    explainer = shap.TreeExplainer(
        model,
        model_output=model_output
    )

    # ===== 2. 计算 SHAP values =====
    shap_values = explainer.shap_values(X)

    # ===== 3. Bar summary =====
    plt.figure()
    shap.summary_plot(
        shap_values,
        X,
        plot_type="bar",
        max_display=max_display_bar,
        show=False
    )
    plt.title("SHAP Feature Importance (Mean |SHAP|)")
    plt.tight_layout()
    plt.show()

    # ===== 4. Beeswarm =====
    shap.summary_plot(
        shap_values,
        X,
        max_display=max_display_beeswarm
    )


def get_permutation_importance(
    model,
    X_valid,
    y_valid,
    scoring="f1",
    n_repeats=5,
    random_state=42
):
    perm = permutation_importance(
        model,
        X_valid,
        y_valid,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )

    perm_df = (
        pd.DataFrame({
            "feature": X_valid.columns,
            "importance": perm.importances_mean
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return perm_df



def get_xgb_gain_importance(model, feature_cols):
    gain_raw = model.get_booster().get_score(importance_type="gain")

    if all(k.startswith("f") for k in gain_raw):
        gain_importance = {
            feature_cols[int(k[1:])]: v
            for k, v in gain_raw.items()
        }
    else:
        gain_importance = gain_raw

    gain_df = (
        pd.DataFrame(
            gain_importance.items(),
            columns=["feature", "gain"]
        )
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )

    return gain_df
def plot_gain_importance(
    gain_df,
    top_k=15,
    title="XGBoost Feature Importance (Gain)",
    figsize=(6, 5),
    save_path=None
):
    """
    绘制 XGBoost Gain Feature Importance 条形图

    Parameters
    ----------
    gain_df : pd.DataFrame
        包含两列：["feature", "gain"]，已按 gain 降序排序
    top_k : int
        显示前多少个特征
    title : str
        图标题
    figsize : tuple
        图像尺寸
    save_path : str or None
        若提供路径，则保存图片
    """

    plt.figure(figsize=figsize)
    plt.barh(
        gain_df["feature"].head(top_k)[::-1],
        gain_df["gain"].head(top_k)[::-1]
    )
    plt.xlabel("Gain Importance")
    plt.title(title)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()