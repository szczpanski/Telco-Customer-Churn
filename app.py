from pathlib import Path
import json
import streamlit as st

st.set_page_config(page_title="Telco Churn", layout="wide")
st.title("Telco Customer Churn — Model Dashboard")

metrics_path = Path("reports/metrics.json")
if not metrics_path.exists():
    st.warning("Rode `make train` antes de abrir o dashboard.")
    st.stop()

with open(metrics_path, encoding="utf-8") as f:
    metrics = json.load(f)

selected = metrics["test_metrics_selected_threshold"]
threshold = metrics["selected_threshold"]
ranking = metrics["ranking_metrics"]
lift10 = next((r["lift"] for r in ranking if abs(r["top_pct"] - 0.10) < 1e-9), None)

st.subheader("Modelo vencedor")
st.write(metrics["winner"])

cols = st.columns(5)
cols[0].metric("ROC-AUC", f'{selected["roc_auc"]:.3f}')
cols[1].metric("PR-AUC", f'{selected["pr_auc"]:.3f}')
cols[2].metric("Recall", f'{selected["recall"]:.3f}')
cols[3].metric("Threshold", f'{threshold:.2f}')
cols[4].metric("Lift @ 10%", f'{lift10:.2f}x' if lift10 is not None else "N/A")

st.subheader("Artefatos de avaliação")
for img in [
    "reports/figures/roc_curve.png",
    "reports/figures/precision_recall_curve.png",
    "reports/figures/confusion_matrix.png",
    "reports/figures/shap_summary.png",
]:
    if Path(img).exists():
        st.image(img, caption=Path(img).stem.replace("_", " ").title(), use_container_width=True)

st.caption("O threshold é selecionado com previsões out-of-fold no treino; o holdout é reservado para avaliação final.")
