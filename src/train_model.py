from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from data_preprocessing import build_preprocessor, load_telco_data, split_xy
from metrics import classification_metrics, ranking_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    reports_dir = Path(args.reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = load_telco_data(args.input)
    X, y = split_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=args.random_state
    )

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    spw = neg / max(pos, 1)
    preprocessor = build_preprocessor(X_train)

    models = {
        "logistic_regression": LogisticRegression(max_iter=2000, solver="liblinear", random_state=args.random_state),
        "logistic_regression_balanced": LogisticRegression(max_iter=2000, solver="liblinear", class_weight="balanced", random_state=args.random_state),
        "random_forest": RandomForestClassifier(n_estimators=500, min_samples_leaf=2, random_state=args.random_state, n_jobs=-1),
        "xgboost": XGBClassifier(n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, eval_metric="logloss", random_state=args.random_state, n_jobs=-1),
        "xgboost_balanced": XGBClassifier(n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, scale_pos_weight=spw, eval_metric="logloss", random_state=args.random_state, n_jobs=-1),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)
    scoring = {"roc_auc": "roc_auc", "pr_auc": "average_precision", "precision": "precision", "recall": "recall", "f1": "f1"}
    comparison = {}

    for name, estimator in models.items():
        pipe = Pipeline([("preprocessor", clone(preprocessor)), ("model", estimator)])
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
        comparison[name] = {
            metric: {"mean": float(np.mean(scores[f"test_{metric}"])), "std": float(np.std(scores[f"test_{metric}"]))}
            for metric in scoring
        }

    champion_name = max(comparison, key=lambda n: comparison[n]["roc_auc"]["mean"])
    champion = Pipeline([("preprocessor", clone(preprocessor)), ("model", models[champion_name])])

    oof_prob = cross_val_predict(champion, X_train, y_train, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    thresholds = np.arange(0.20, 0.81, 0.01)
    f1s = [f1_score(y_train, (oof_prob >= t).astype(int), zero_division=0) for t in thresholds]
    selected_threshold = float(thresholds[int(np.argmax(f1s))])

    champion.fit(X_train, y_train)
    y_prob = champion.predict_proba(X_test)[:, 1]
    ranking = ranking_metrics(y_test, y_prob)

    artifact = {
        "winner": champion_name,
        "selection_metric": "5-fold CV ROC-AUC",
        "cv_comparison": comparison,
        "selected_threshold": selected_threshold,
        "test_metrics_threshold_0_50": classification_metrics(y_test, y_prob, 0.50),
        "test_metrics_selected_threshold": classification_metrics(y_test, y_prob, selected_threshold),
        "ranking_metrics": ranking.to_dict(orient="records"),
        "dataset": {
            "rows": int(len(df)),
            "churn_rate": float(y.mean()),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        },
    }

    joblib.dump(champion, output_dir / "best_model.joblib")
    X_test.assign(Churn=y_test.values).to_csv(reports_dir / "test_sample.csv", index=False)
    with open(reports_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=False)

    print(json.dumps(artifact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
