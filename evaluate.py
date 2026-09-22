"""
evaluate.py
===========
Loads the fitted pipelines and the held-out test split saved by train.py,
computes accuracy / precision / recall / F1 / ROC-AUC / confusion matrix for
both models on the TEST set (never seen during training or CV), and writes:

  reports/evaluation.md   - markdown metrics table + written analysis +
                             Random Forest top-5 feature importances
  reports/roc_curve.png   - ROC curves for both models

Run:
    python evaluate.py
"""

import os
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
)

MODELS_DIR = "models"
REPORTS_DIR = "reports"

# Categorical palette slots (dataviz skill default, validated for CVD /
# contrast), used consistently for the two model series.
COLOR_LOGREG = "#2a78d6"   # slot 1 - blue
COLOR_RF = "#eb6834"       # slot 2 - orange
COLOR_DIAGONAL = "#9a9a94"  # neutral gray reference line


def load_artifacts():
    logreg = joblib.load(os.path.join(MODELS_DIR, "logistic_regression.joblib"))
    rf = joblib.load(os.path.join(MODELS_DIR, "random_forest.joblib"))
    X_test, y_test = joblib.load(os.path.join(MODELS_DIR, "test_split.joblib"))
    return {"Logistic Regression": logreg, "Random Forest": rf}, X_test, y_test


def compute_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    cm = confusion_matrix(y_test, y_pred)
    return metrics, cm, y_proba


def get_feature_names(pipeline):
    preprocess = pipeline.named_steps["preprocess"]
    return preprocess.get_feature_names_out()


def top_feature_importances(rf_pipeline, n=5):
    importances = rf_pipeline.named_steps["model"].feature_importances_
    names = get_feature_names(rf_pipeline)
    order = np.argsort(importances)[::-1][:n]
    return [(names[i], importances[i]) for i in order]


def plot_roc_curves(results, y_test, out_path):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)

    colors = {"Logistic Regression": COLOR_LOGREG, "Random Forest": COLOR_RF}
    for name, (metrics, cm, y_proba) in results.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        ax.plot(
            fpr, tpr,
            color=colors[name],
            linewidth=2,
            label=f"{name} (AUC = {metrics['roc_auc']:.3f})",
        )

    ax.plot([0, 1], [0, 1], linestyle="--", color=COLOR_DIAGONAL, linewidth=1.5,
             label="Random chance (AUC = 0.500)")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Churn Prediction Models (held-out test set)")
    ax.legend(loc="lower right", frameon=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_report(results, rf_pipeline, out_path, winner, winner_reason):
    lines = []
    lines.append("# Churn Model Evaluation Report\n")
    lines.append(
        "Metrics below are computed on a held-out test set (20% of the data, "
        "stratified split, never used for training or cross-validation).\n"
    )

    lines.append("## Metrics\n")
    lines.append("| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |")
    lines.append("|---|---|---|---|---|---|")
    for name, (metrics, cm, y_proba) in results.items():
        lines.append(
            f"| {name} | {metrics['accuracy']:.3f} | {metrics['precision']:.3f} "
            f"| {metrics['recall']:.3f} | {metrics['f1']:.3f} | {metrics['roc_auc']:.3f} |"
        )
    lines.append("")

    lines.append("## Confusion Matrices\n")
    for name, (metrics, cm, y_proba) in results.items():
        lines.append(f"**{name}**  (rows = actual [no-churn, churn], columns = predicted [no-churn, churn])\n")
        lines.append("```")
        lines.append(f"[[{cm[0][0]:>5} {cm[0][1]:>5}]")
        lines.append(f" [{cm[1][0]:>5} {cm[1][1]:>5}]]")
        lines.append("```\n")

    lines.append("## ROC Curve\n")
    lines.append("![ROC curve](roc_curve.png)\n")

    lines.append("## Analysis\n")
    lines.append(f"**Winner: {winner}**\n")
    lines.append(winner_reason + "\n")

    lines.append("## Random Forest — Top 5 Feature Importances\n")
    lines.append("| Feature | Importance |")
    lines.append("|---|---|")
    for feat, imp in top_feature_importances(rf_pipeline, n=5):
        lines.append(f"| {feat} | {imp:.4f} |")
    lines.append("")

    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def main():
    models, X_test, y_test = load_artifacts()

    results = {}
    for name, model in models.items():
        metrics, cm, y_proba = compute_metrics(model, X_test, y_test)
        results[name] = (metrics, cm, y_proba)
        print(f"\n[{name}]")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        print(f"  confusion_matrix:\n{cm}")

    os.makedirs(REPORTS_DIR, exist_ok=True)
    roc_path = os.path.join(REPORTS_DIR, "roc_curve.png")
    plot_roc_curves(results, y_test, roc_path)
    print(f"\nSaved ROC curve plot -> {roc_path}")

    # Determine winner by test-set ROC-AUC and write a short, honest reason.
    winner = max(results, key=lambda n: results[n][0]["roc_auc"])
    loser = [n for n in results if n != winner][0]
    winner_auc = results[winner][0]["roc_auc"]
    loser_auc = results[loser][0]["roc_auc"]

    if winner == "Random Forest":
        reason = (
            f"Random Forest achieves the higher ROC-AUC ({winner_auc:.3f} vs. "
            f"{loser_auc:.3f} for Logistic Regression). A plausible reason: the "
            "churn label is generated from a logistic function of the raw "
            "features, but that function is applied to feature combinations "
            "(e.g. support tickets interacting with contract type) that a "
            "linear model captures only through the features actually offered "
            "to it, while a tree ensemble can carve out non-linear regions and "
            "interactions among tenure, usage, and support tickets without "
            "those interactions being explicitly engineered."
        )
    else:
        reason = (
            f"Logistic Regression achieves the higher ROC-AUC ({winner_auc:.3f} "
            f"vs. {loser_auc:.3f} for Random Forest). This is plausible because "
            "the synthetic label is generated directly from a linear "
            "combination of the (encoded) features passed through a sigmoid — "
            "i.e. the true data-generating process is close to logistic — so a "
            "correctly-specified logistic model can match that structure "
            "almost exactly, while the Random Forest has more variance and can "
            "slightly overfit non-existent interactions on a dataset of this "
            "size."
        )

    report_path = os.path.join(REPORTS_DIR, "evaluation.md")
    write_report(results, models["Random Forest"], report_path, winner, reason)
    print(f"\nWrote report -> {report_path}")
    print(f"Winner (by test ROC-AUC): {winner}")


if __name__ == "__main__":
    main()
