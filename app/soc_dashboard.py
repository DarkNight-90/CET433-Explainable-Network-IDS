import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import shap

# Streamlit can launch this script with only app/ on the import path.
project_dir = str(Path(__file__).resolve().parents[1])
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
from ids_utils import prepare_features, read_flow_csv
from model_artifact import load_model_artifact

st.set_page_config(page_title="SOC Network IDS", page_icon="🛡️", layout="wide")
st.title("🛡️ SOC Network Intrusion Detection Dashboard")
st.caption("CET433 — Project 2 | Explainable Network IDS")

try:
    artifact = load_model_artifact()
    pipe = artifact["pipeline"]
    threshold = float(artifact["threshold"])
    feature_cols = artifact["feature_cols"]
    model_name = artifact["model_name"]
except FileNotFoundError as exc:
    st.error(str(exc))
    st.info("Run the M01R notebook through its save-model cell.")
    st.stop()
except Exception as exc:
    st.error(f"IDS model artifact could not be loaded or validated: {exc}")
    st.info("Use a trusted artifact exported by the M01R notebook and its locked environment.")
    st.stop()

st.sidebar.header("Model information")
st.sidebar.write(f"**Model:** {model_name}")
st.sidebar.write(f"**Decision threshold:** {threshold:.3f}")
st.sidebar.write(f"**Features expected:** {len(feature_cols)}")

uploaded = st.file_uploader("Upload a CSV file containing network flows", type=["csv"])
if uploaded is None:
    st.info("Upload a CSV file to score network flows. The CSV may include id/label/attack_cat columns; they are excluded from model inputs.")
    st.stop()

try:
    raw = read_flow_csv(uploaded)
    X = prepare_features(raw, feature_cols)
    probs = pipe.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)
    out = raw.copy()
    out["attack_probability"] = probs
    out["prediction"] = np.where(preds == 1, "Attack", "Normal")
    out["alert_level"] = np.select([probs >= max(threshold, 0.8), preds == 1],
                                   ["High", "Review"], default="Normal")
except Exception as exc:
    st.error(f"Could not process this CSV: {exc}")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Flows analyzed", f"{len(out):,}")
c2.metric("Attack alerts", f"{int((preds == 1).sum()):,}")
c3.metric("Normal flows", f"{int((preds == 0).sum()):,}")
c4.metric("Alert rate", f"{(preds.mean() * 100):.1f}%")

st.subheader("Alert overview")
counts = out["prediction"].value_counts().rename_axis("prediction").reset_index(name="flows")
st.bar_chart(counts.set_index("prediction"))
st.subheader("Scored network flows")
st.dataframe(out, use_container_width=True, hide_index=True)
st.download_button("Download scored results (CSV)", out.to_csv(index=False).encode("utf-8"),
                   file_name="ids_scored_alerts.csv", mime="text/csv")

st.subheader("Explain an individual alert (SHAP)")
alert_rows = np.flatnonzero(preds == 1)
if len(alert_rows) == 0:
    st.warning("No rows crossed the current threshold. Choose a row below to inspect its score.")
choices = list(range(len(out)))
chosen = st.selectbox("Select flow row", choices, format_func=lambda i: f"Row {i} — {out.iloc[i]['prediction']} ({probs[i]:.3f})")
try:
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    transformed = pre.transform(X.iloc[[chosen]])
    names = list(pre.get_feature_names_out())
    explainer = shap.TreeExplainer(clf)
    vals = explainer.shap_values(transformed)
    if isinstance(vals, list):
        vals = vals[1]
    vals = np.asarray(vals)
    if vals.ndim == 3:
        vals = vals[:, :, 1]
    contrib = vals[0]
    top = np.argsort(np.abs(contrib))[::-1][:10]
    reasons = pd.DataFrame({"feature": [names[j] for j in top],
                            "SHAP contribution": [float(contrib[j]) for j in top]})
    st.write(f"**Prediction:** {out.iloc[chosen]['prediction']} | **Attack probability:** {probs[chosen]:.3f}")
    st.caption("Positive SHAP values push the model toward Attack; negative values push it toward Normal.")
    st.dataframe(reasons, use_container_width=True, hide_index=True)
except Exception as exc:
    st.warning(f"SHAP explanation could not be generated for this model/row: {exc}")

st.caption("Decision support only: SOC analysts must verify alerts. Model output is not proof of malicious activity.")
