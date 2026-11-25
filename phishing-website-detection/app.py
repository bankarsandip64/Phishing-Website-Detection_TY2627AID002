"""Streamlit decision-support prototype for Phishing Website Detection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
import streamlit as st

from src.data_loader import FEATURE_METADATA, get_feature_names, load_dataset
from src.train_evaluate import ARTIFACTS_DIR, MODELS_DIR, PLOTS_DIR, run_training_pipeline

st.set_page_config(
    page_title="Phishing Website Detection",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_resource
def load_models_and_summary():
    """Load trained models and evaluation summary, training if artifacts are missing."""
    all_models_path = MODELS_DIR / "all_models.joblib"
    summary_path = ARTIFACTS_DIR / "metrics_summary.json"

    if not all_models_path.exists() or not summary_path.exists():
        run_training_pipeline(save_artifacts=True)

    models = joblib.load(all_models_path)
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    return models, summary


def get_presets(feature_names: list[str]) -> Dict[str, Dict[str, int]]:
    """Return sample feature presets for quick demonstration."""
    legitimate = {f: 1 for f in feature_names}

    phishing = {f: 1 for f in feature_names}
    phishing.update(
        {
            "having_IP_Address": -1,
            "URL_Length": -1,
            "Shortining_Service": -1,
            "Prefix_Suffix": -1,
            "having_Sub_Domain": -1,
            "SSLfinal_State": -1,
            "Domain_registeration_length": -1,
            "Request_URL": -1,
            "URL_of_Anchor": -1,
            "Links_in_tags": -1,
            "SFH": -1,
            "age_of_domain": -1,
            "DNSRecord": -1,
            "web_traffic": -1,
            "Statistical_report": -1,
        }
    )

    suspicious = {f: 1 for f in feature_names}
    suspicious.update(
        {
            "URL_Length": 0,
            "having_Sub_Domain": 0,
            "SSLfinal_State": 1,
            "Domain_registeration_length": -1,
            "URL_of_Anchor": 0,
            "Links_in_tags": 0,
            "SFH": 0,
            "age_of_domain": -1,
            "web_traffic": 0,
        }
    )

    return {
        "Legitimate Trusted Site": legitimate,
        "High-Risk Phishing Site": phishing,
        "Borderline / Suspicious Site": suspicious,
    }


def main() -> None:
    st.title("🛡️ Phishing Website Detection")
    st.caption(
        "Machine learning decision-support prototype using URL, webpage, and domain characteristics."
    )

    models, summary = load_models_and_summary()
    feature_names = get_feature_names()
    presets = get_presets(feature_names)
    best_name = summary["best_model_name"]

    # Sidebar controls
    with st.sidebar:
        st.header("Configuration")
        model_names = list(models.keys())
        default_idx = model_names.index(best_name) if best_name in model_names else 0
        selected_model_name = st.selectbox(
            "Select Model",
            options=model_names,
            index=default_idx,
            help=f"Best validation model: {best_name}",
        )

        st.subheader("Sample Presets")
        preset_choice = st.radio(
            "Load a preset profile:",
            options=list(presets.keys()),
            index=0,
        )
        if st.button("Apply Preset", use_container_width=True):
            for feat, val in presets[preset_choice].items():
                st.session_state[f"feat_{feat}"] = val

        st.divider()
        st.warning(
            "**Decision-Support Disclaimer:** This application is an educational "
            "ML prototype trained on the 2015 UCI Phishing Websites dataset. "
            "Predictions are probabilistic risk indicators—not definitive security "
            "verdicts—and should be verified by security analysts before blocking domains."
        )

    # Initialize session state for features
    for feat in feature_names:
        key = f"feat_{feat}"
        if key not in st.session_state:
            st.session_state[key] = presets["Legitimate Trusted Site"][feat]

    tab_predict, tab_eval = st.tabs(["🔍 Website Risk Predictor", "📊 Model Comparison & Error Analysis"])

    with tab_predict:
        st.subheader("Interactive Feature Input")
        st.markdown(
            "Adjust the website characteristics below (`-1` = Phishing indicator, "
            "`0` = Suspicious, `1` = Legitimate indicator)."
        )

        groups = ["URL / Address Bar", "Webpage Features", "Domain Information"]
        cols = st.columns(3)

        for col, group_name in zip(cols, groups):
            with col:
                st.markdown(f"#### {group_name}")
                group_feats = [
                    f for f in feature_names if FEATURE_METADATA[f]["group"] == group_name
                ]
                for feat in group_feats:
                    meta = FEATURE_METADATA[feat]
                    options = meta["values"]
                    st.select_slider(
                        label=str(meta["label"]),
                        options=options,
                        key=f"feat_{feat}",
                        help=str(meta["description"]),
                    )

        input_row = {feat: st.session_state[f"feat_{feat}"] for feat in feature_names}
        input_df = pd.DataFrame([input_row])

        model = models[selected_model_name]
        phishing_prob = float(model.predict_proba(input_df)[0, 1])
        pred_label = "Phishing" if phishing_prob >= 0.5 else "Legitimate"

        st.divider()
        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            st.metric("Selected Model", selected_model_name)
        with res_col2:
            st.metric("Phishing Risk Probability", f"{phishing_prob * 100:.1f}%")
        with res_col3:
            if pred_label == "Phishing":
                st.error("⚠️ Prediction: **PHISHING (High Risk)**")
            else:
                st.success("✅ Prediction: **LEGITIMATE (Low Risk)**")

        st.progress(min(max(phishing_prob, 0.0), 1.0))

        risk_signals = [
            FEATURE_METADATA[f]["label"] for f, v in input_row.items() if v == -1
        ]
        suspicious_signals = [
            FEATURE_METADATA[f]["label"] for f, v in input_row.items() if v == 0
        ]
        sig_col1, sig_col2 = st.columns(2)
        with sig_col1:
            st.markdown(f"**High-Risk Indicators (`-1`):** {len(risk_signals)}")
            if risk_signals:
                st.write(", ".join(str(s) for s in risk_signals))
        with sig_col2:
            st.markdown(f"**Suspicious Indicators (`0`):** {len(suspicious_signals)}")
            if suspicious_signals:
                st.write(", ".join(str(s) for s in suspicious_signals))

    with tab_eval:
        st.subheader("Held-Out Test Set Performance (20% Split)")
        test_rows = []
        for m_name, metrics in summary["test_results"].items():
            test_rows.append(
                {
                    "Model": m_name,
                    "Precision": round(metrics["precision"], 4),
                    "Recall": round(metrics["recall"], 4),
                    "F1-Score": round(metrics["f1"], 4),
                    "ROC-AUC": round(metrics["roc_auc"], 4),
                    "PR-AUC": round(metrics["pr_auc"], 4),
                    "Accuracy": round(metrics["accuracy"], 4),
                }
            )
        st.dataframe(pd.DataFrame(test_rows), use_container_width=True, hide_index=True)

        p1 = PLOTS_DIR / "model_comparison.png"
        p2 = PLOTS_DIR / "roc_pr_curves.png"
        p3 = PLOTS_DIR / "confusion_matrices.png"
        p4 = PLOTS_DIR / "error_analysis_and_importance.png"

        c1, c2 = st.columns(2)
        if p1.exists():
            c1.image(str(p1), caption="Model Comparison (Test Set)")
        if p2.exists():
            c2.image(str(p2), caption="ROC & Precision-Recall Curves")

        c3, c4 = st.columns(2)
        if p3.exists():
            c3.image(str(p3), caption="Confusion Matrices")
        if p4.exists():
            c4.image(str(p4), caption="Feature Importance & Error Confidence")

        err = summary["error_analysis"]
        st.markdown(
            f"**Error Analysis ({err['model']} on {err['test_total']} Test Websites):** "
            f"{err['false_positives']} False Positives (FPR={err['false_positive_rate']:.2%}), "
            f"{err['false_negatives']} False Negatives (FNR={err['false_negative_rate']:.2%})."
        )


if __name__ == "__main__":
    main()
