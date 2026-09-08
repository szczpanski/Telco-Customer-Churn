from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
    }


def ranking_metrics(y_true, y_prob, ks=(0.05, 0.10, 0.20, 0.30, 0.40)) -> pd.DataFrame:
    temp = pd.DataFrame({"y_true": np.asarray(y_true), "y_prob": np.asarray(y_prob)})
    temp = temp.sort_values("y_prob", ascending=False).reset_index(drop=True)
    total_churners = temp["y_true"].sum()
    base_rate = temp["y_true"].mean()
    rows = []
    for k in ks:
        n = max(1, int(np.ceil(len(temp) * k)))
        top = temp.head(n)
        rate = top["y_true"].mean()
        captured = top["y_true"].sum()
        rows.append({
            "top_pct": float(k),
            "clientes": int(n),
            "churn_rate_top": float(rate),
            "lift": float(rate / base_rate) if base_rate > 0 else float("nan"),
            "capture_rate": float(captured / total_churners) if total_churners > 0 else float("nan"),
        })
    return pd.DataFrame(rows)
