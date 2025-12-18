from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from rich.console import Console
from rich.table import Table
import numpy as np

def print_metrics_table(results: dict):
    console = Console()

    # 找出 F1 最优模型
    best_model = max(results, key=lambda k: results[k]["F1"])

    table = Table(
        title="📊 Model Evaluation Results",
        show_header=True,
        header_style="bold cyan"
    )

    table.add_column("Model", style="bold")
    table.add_column("F1", justify="right")
    table.add_column("Accuracy", justify="right")
    table.add_column("AUC", justify="right")
    table.add_column("Threshold", justify="center")

    for model_name, metrics in results.items():
        threshold = metrics.get("BestThreshold", 0.5)

        table.add_row(
            model_name,
            f"{metrics['F1']:.4f}",
            f"{metrics['Accuracy']:.4f}",
            f"{metrics['AUC']:.4f}",
            f"{threshold:.2f}"
        )

    console.print(table)

    # 表格外单独强调 Best F1
    console.print(
        f"\n✅ Best model based on F1-score: "
        f"[bold green]{best_model}[/bold green] "
        f"(F1 = {results[best_model]['F1']:.4f}, "
        f"threshold = {results[best_model]['BestThreshold']:.2f})"
    )

def find_best_threshold(model, X_valid, y_valid):
    """
    在验证集上搜索使 F1-score 最大的分类阈值
    """
    y_proba = model.predict_proba(X_valid)[:, 1]
    thresholds = np.linspace(0.1, 0.9, 81)

    best_f1, best_t = 0.0, 0.5
    for t in thresholds:
        f1 = f1_score(y_valid, (y_proba >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, t

    return best_t, best_f1

def evaluate_model(model, X, y, threshold=0.5):
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "F1": f1_score(y, y_pred),
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_proba)
    }

def evaluate_models(models: dict, X_valid, y_valid, optimize_threshold=True):
    """
    对多个模型进行评估
    若 optimize_threshold=True，则为每个模型自动搜索最优 threshold
    """
    results = {}

    for name, model in models.items():
        if optimize_threshold:
            best_t, best_f1 = find_best_threshold(model, X_valid, y_valid)
            metrics = evaluate_model(model, X_valid, y_valid, threshold=best_t)
            metrics["BestThreshold"] = best_t
        else:
            metrics = evaluate_model(model, X_valid, y_valid)
            metrics["BestThreshold"] = 0.5

        results[name] = metrics

    print_metrics_table(results)
    return results