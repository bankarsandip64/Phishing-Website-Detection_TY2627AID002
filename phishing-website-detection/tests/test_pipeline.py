"""End-to-end tests for data loading, preprocessing, training, evaluation, and app presets."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from app import get_presets
from src.data_loader import TARGET_COLUMN, get_dataset_summary, get_feature_names, load_dataset
from src.preprocessing import build_preprocessor, encode_target, split_dataset
from src.train_evaluate import MODELS_DIR, PLOTS_DIR, run_training_pipeline


def test_dataset_loads_and_schema():
    df = load_dataset()
    assert df.shape == (11055, 31)
    assert TARGET_COLUMN in df.columns
    assert set(df[TARGET_COLUMN].unique()) == {-1, 1}
    assert df.isna().sum().sum() == 0

    summary = get_dataset_summary(df)
    assert summary["n_rows"] == 11055
    assert summary["n_features"] == 30
    assert summary["phishing_count"] == 4898
    assert summary["legitimate_count"] == 6157


def test_split_ratios_and_no_dropped_real_features():
    df = load_dataset()
    splits = split_dataset(df, random_state=42)

    assert len(splits.X_train) == 6633
    assert len(splits.X_val) == 2211
    assert len(splits.X_test) == 2211
    assert len(splits.feature_names) == 30
    assert splits.dropped_columns == []

    # Verify stratification (~44.3% phishing across all splits)
    for y_split in (splits.y_train, splits.y_val, splits.y_test):
        rate = y_split.mean()
        assert 0.43 <= rate <= 0.46


def test_id_and_constant_columns_are_removed():
    df = load_dataset().head(100).copy()
    df["id"] = range(len(df))
    df["constant_col"] = 7

    splits = split_dataset(df, random_state=42)
    assert "id" in splits.dropped_columns
    assert "constant_col" in splits.dropped_columns
    assert "id" not in splits.feature_names
    assert "constant_col" not in splits.feature_names


def test_leak_free_preprocessor_fit_only_on_train():
    df = load_dataset()
    splits = split_dataset(df, random_state=42)
    prep = build_preprocessor(scale_features=True)

    X_tr = prep.fit_transform(splits.X_train)
    X_val = prep.transform(splits.X_val)
    X_te = prep.transform(splits.X_test)

    assert X_tr.shape == (6633, 30)
    assert X_val.shape == (2211, 30)
    assert X_te.shape == (2211, 30)
    assert np.isfinite(X_tr).all()
    assert np.isfinite(X_val).all()
    assert np.isfinite(X_te).all()


def test_training_pipeline_and_saved_artifacts():
    report = run_training_pipeline(random_state=42, save_artifacts=False)
    assert set(report["test_results"].keys()) == {
        "Logistic Regression",
        "SVM",
        "Decision Tree",
        "Naive Bayes",
    }
    assert report["best_model_name"] in report["test_results"]

    for model_name, metrics in report["test_results"].items():
        assert metrics["roc_auc"] > 0.90, f"{model_name} ROC-AUC unexpectedly low"
        assert metrics["pr_auc"] > 0.90, f"{model_name} PR-AUC unexpectedly low"

    # Verify saved artifacts exist from full pipeline run
    assert (MODELS_DIR / "best_model.joblib").exists()
    assert (MODELS_DIR / "all_models.joblib").exists()
    for plot_name in (
        "model_comparison.png",
        "roc_pr_curves.png",
        "confusion_matrices.png",
        "error_analysis_and_importance.png",
    ):
        assert (PLOTS_DIR / plot_name).exists()


def test_app_presets_predictions():
    feature_names = get_feature_names()
    presets = get_presets(feature_names)
    best_model = joblib.load(MODELS_DIR / "best_model.joblib")

    legit_df = pd.DataFrame([presets["Legitimate Trusted Site"]])
    phish_df = pd.DataFrame([presets["High-Risk Phishing Site"]])

    legit_prob = float(best_model.predict_proba(legit_df)[0, 1])
    phish_prob = float(best_model.predict_proba(phish_df)[0, 1])

    assert legit_prob < 0.30
    assert phish_prob > 0.70
