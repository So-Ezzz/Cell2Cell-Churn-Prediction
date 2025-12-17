from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from rich.console import Console
from rich.table import Table

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
    table.add_column("Note", justify="center")

    for model_name, metrics in results.items():
        is_best = model_name == best_model

        table.add_row(
            f"[bold yellow]{model_name}[/bold yellow]" if is_best else model_name,
            f"{metrics['F1']:.4f}",
            f"{metrics['Accuracy']:.4f}",
            f"{metrics['AUC']:.4f}",
            "🏆 Best F1" if is_best else ""
        )

    console.print(table)

    console.print(
        f"\n✅ Best model based on F1-score: "
        f"[bold green]{best_model}[/bold green] "
        f"(F1 = {results[best_model]['F1']:.4f})"
    )

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