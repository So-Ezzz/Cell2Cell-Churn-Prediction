from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from rich.console import Console
from rich.table import Table
import numpy as np

def print_metrics_table(results: dict):
    console = Console()

    # 找出 F1 最优模型（基于 CV mean）
    best_model = max(results, key=lambda k: results[k]["F1"])

    table = Table(
        title="📊 Cross-Validation Model Evaluation Results",
        show_header=True,
        header_style="bold cyan"
    )

    table.add_column("Model (CV=5)", style="bold")
    table.add_column("F1 (mean)", justify="right")
    table.add_column("F1 (std)", justify="right")
    table.add_column("Accuracy (mean)", justify="right")
    table.add_column("AUC (mean)", justify="right")

    for model_name, metrics in results.items():
        is_best = model_name == best_model

        table.add_row(
            f"[bold yellow]{model_name}[/bold yellow]" if is_best else model_name,
            f"{metrics['F1']:.4f}",
            f"{metrics['F1_std']:.4f}",
            f"{metrics['Accuracy']:.4f}",
            f"{metrics['AUC']:.4f}",
        )

    console.print(table)

    console.print(
        f"\n✅ Best model based on CV mean F1-score: "
        f"[bold green]{best_model}[/bold green] "
        f"(F1 = {results[best_model]['F1']:.4f} ± "
        f"{results[best_model]['F1_std']:.4f})"
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

def evaluate_models_cv(
    model_builders: dict,
    X,
    y,
    n_splits=6,
    optimize_threshold=True,
    random_state=42
):
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    results = {}

    for name, train_fn in model_builders.items():
        f1_scores = []
        accuracy_scores = []
        auc_scores = []

        for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

            # ===== 训练模型 =====
            if name == "Deep ANN":
                model = train_fn(X_train, y_train, X_valid, y_valid)
            else:
                model = train_fn(X_train, y_train)

            # ===== 阈值搜索 =====
            if optimize_threshold:
                best_t, _ = find_best_threshold(model, X_valid, y_valid)
            else:
                best_t = 0.5

            metrics = evaluate_model(model, X_valid, y_valid, threshold=best_t)

            f1_scores.append(metrics["F1"])
            accuracy_scores.append(metrics["Accuracy"])
            auc_scores.append(metrics["AUC"])


        results[name] = {
            "F1": np.mean(f1_scores),
            "Accuracy": np.mean(accuracy_scores),
            "AUC": np.mean(auc_scores),
            "F1_std": np.std(f1_scores)
        }

    print_metrics_table(results)
    return results