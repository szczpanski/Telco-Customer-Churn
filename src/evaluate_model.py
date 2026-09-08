from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

TARGET = "Churn"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/best_model.joblib")
    parser.add_argument("--test-data", default="reports/test_sample.csv")
    parser.add_argument("--metrics", default="reports/metrics.json")
    parser.add_argument("--figures-dir", default="reports/figures")
    args = parser.parse_args()

    model = joblib.load(args.model)
    test_df = pd.read_csv(args.test_data)
    with open(args.metrics, encoding="utf-8") as f:
        metrics = json.load(f)

    threshold = float(metrics.get("selected_threshold", 0.50))
    X = test_df.drop(columns=[TARGET])
    y = test_df[TARGET].astype(int)
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    figures = Path(args.figures_dir)
    figures.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    RocCurveDisplay.from_predictions(y, y_prob, ax=ax)
    ax.set_title("ROC Curve — Holdout Test")
    fig.tight_layout(); fig.savefig(figures / "roc_curve.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    PrecisionRecallDisplay.from_predictions(y, y_prob, ax=ax)
    ax.set_title("Precision-Recall Curve — Holdout Test")
    fig.tight_layout(); fig.savefig(figures / "precision_recall_curve.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y, y_pred, display_labels=["Não Churn", "Churn"], ax=ax)
    ax.set_title(f"Confusion Matrix — threshold={threshold:.2f}")
    fig.tight_layout(); fig.savefig(figures / "confusion_matrix.png", dpi=160); plt.close(fig)

    prep = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]
    sample = X.sample(min(500, len(X)), random_state=42)
    Xt = prep.transform(sample)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    feature_names = prep.get_feature_names_out()
    model_name = estimator.__class__.__name__.lower()
    if "xgb" in model_name or "forest" in model_name:
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(Xt)
        if isinstance(shap_values, list):
            shap_values = shap_values[-1]
    else:
        explainer = shap.LinearExplainer(estimator, Xt)
        shap_values = explainer.shap_values(Xt)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, Xt, feature_names=feature_names, show=False, max_display=20)
    plt.tight_layout(); plt.savefig(figures / "shap_summary.png", dpi=160, bbox_inches="tight"); plt.close()

    print(f"Artefatos salvos em: {figures}")


if __name__ == "__main__":
    main()
