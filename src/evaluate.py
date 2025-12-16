from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from rich.console import Console
from rich.table import Table

def print_metrics_table(results: dict):
    """
    results 示例：
    {
        "Logistic Regression": {"F1": 0.45, "Accuracy": 0.59, "AUC": 0.61},
        "XGBoost": {"F1": 0.51, "Accuracy": 0.70, "AUC": 0.68}
    }
    """
    console = Console()

    table = Table(
        title="📊 Model Evaluation Results",
        show_header=True,
        header_style="bold cyan"
    )

    table.add_column("Model", style="bold")
    table.add_column("F1", justify="right")
    table.add_column("Accuracy", justify="right")
    table.add_column("AUC", justify="right")

    for model_name, metrics in results.items():
        table.add_row(
            model_name,
            f"{metrics['F1']:.4f}",
            f"{metrics['Accuracy']:.4f}",
            f"{metrics['AUC']:.4f}",
        )

    console.print(table)

def evaluate_model(model, X, y, threshold=0.5):
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "F1": f1_score(y, y_pred),
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_proba)
    }

def evaluate_models(models: dict, X_valid, y_valid, threshold=0.5):
    """
    对多个模型进行统一评估并使用 rich 输出结果表
    """
    results = {}

    for name, model in models.items():
        results[name] = evaluate_model(
            model,
            X_valid,
            y_valid,
            threshold=threshold
        )

    print_metrics_table(results)
    return results