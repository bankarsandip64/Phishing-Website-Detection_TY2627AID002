"""Model training, validation tuning, test evaluation, error analysis, and plotting."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.data_loader import PROJECT_ROOT, get_dataset_summary, load_dataset
from src.preprocessing import (
    DatasetSplits,
    build_preprocessor,
    get_selected_feature_names,
    split_dataset,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
MODELS_DIR = ARTIFACTS_DIR / "models"


def get_model_search_spaces(random_state: int = 42) -> Dict[str, Dict[str, Any]]:
    """Define candidate models and hyperparameter grids for validation tuning."""
    return {
        "Logistic Regression": {
            "scale": True,
            "builder": lambda params: LogisticRegression(
                max_iter=1000, random_state=random_state, **params
            ),
            "grid": [
                {"C": c, "class_weight": cw}
                for c in (0.01, 0.1, 1.0, 10.0)
                for cw in (None, "balanced")
            ],
        },
        "SVM": {
            "scale": True,
            "builder": lambda params: SVC(
                probability=True, random_state=random_state, **params
            ),
            "grid": [
                {"C": c, "kernel": k, "gamma": "scale"}
                for c in (0.1, 1.0, 10.0)
                for k in ("rbf", "linear")
            ],
        },
        "Decision Tree": {
            "scale": False,
            "builder": lambda params: DecisionTreeClassifier(
                random_state=random_state, **params
            ),
            "grid": [
                {"max_depth": md, "min_samples_leaf": msl, "criterion": crit}
                for md in (4, 8, 12, 16, None)
                for msl in (1, 5, 10)
                for crit in ("gini", "entropy")
            ],
        },
        "Naive Bayes": {
            "scale": True,
            "builder": lambda params: GaussianNB(**params),
            "grid": [{"var_smoothing": vs} for vs in (1e-9, 1e-7, 1e-5, 1e-3)],
        },
    }


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, Any]:
    """Compute imbalanced-classification metrics for binary phishing detection."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(x) for x in cm.ravel())
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def tune_on_validation(
    splits: DatasetSplits,
    random_state: int = 42,
) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, Any]]]:
    """Fit candidate hyperparameters on Train and select best per model on Validation."""
    spaces = get_model_search_spaces(random_state=random_state)
    best_pipelines: Dict[str, Pipeline] = {}
    tuning_records: Dict[str, Dict[str, Any]] = {}

    for model_name, spec in spaces.items():
        best_val_f1 = -1.0
        best_val_roc = -1.0
        best_pipeline: Pipeline | None = None
        best_params: Dict[str, Any] = {}
        best_val_metrics: Dict[str, Any] = {}
        best_train_time = 0.0

        for params in spec["grid"]:
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", build_preprocessor(scale_features=spec["scale"])),
                    ("classifier", spec["builder"](params)),
                ]
            )
            t0 = time.perf_counter()
            pipeline.fit(splits.X_train, splits.y_train)
            fit_time = time.perf_counter() - t0

            val_prob = pipeline.predict_proba(splits.X_val)[:, 1]
            val_pred = (val_prob >= 0.5).astype(int)
            val_metrics = compute_classification_metrics(
                splits.y_val.to_numpy(), val_pred, val_prob
            )

            # Select by Validation F1, breaking ties with Validation ROC-AUC
            if (val_metrics["f1"], val_metrics["roc_auc"]) > (
                best_val_f1,
                best_val_roc,
            ):
                best_val_f1 = val_metrics["f1"]
                best_val_roc = val_metrics["roc_auc"]
                best_pipeline = pipeline
                best_params = params
                best_val_metrics = val_metrics
                best_train_time = fit_time

        assert best_pipeline is not None
        best_pipelines[model_name] = best_pipeline
        tuning_records[model_name] = {
            "best_params": best_params,
            "train_time_sec": round(best_train_time, 4),
            "validation_metrics": best_val_metrics,
        }

    return best_pipelines, tuning_records


def analyze_errors(
    best_model_name: str,
    best_pipeline: Pipeline,
    splits: DatasetSplits,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Perform error analysis on held-out Test predictions for the best model."""
    y_true = splits.y_test.to_numpy()
    y_prob = best_pipeline.predict_proba(splits.X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    fp_mask = (y_true == 0) & (y_pred == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)

    test_df = splits.X_test.copy()
    test_df["true_label"] = y_true
    test_df["pred_label"] = y_pred
    test_df["phishing_prob"] = y_prob

    # Worst false positives: legitimate sites predicted with highest phishing probability
    worst_fp = (
        test_df[fp_mask]
        .sort_values("phishing_prob", ascending=False)
        .head(top_k)
        .reset_index()
    )
    # Worst false negatives: phishing sites predicted with lowest phishing probability
    worst_fn = (
        test_df[fn_mask]
        .sort_values("phishing_prob", ascending=True)
        .head(top_k)
        .reset_index()
    )

    # Compare mean feature values for FP vs TN and FN vs TP
    tn_mask = (y_true == 0) & (y_pred == 0)
    tp_mask = (y_true == 1) & (y_pred == 1)

    feature_cols = splits.feature_names
    fp_diff = (
        (splits.X_test[fp_mask].mean() - splits.X_test[tn_mask].mean())
        .abs()
        .sort_values(ascending=False)
        .head(5)
    )
    fn_diff = (
        (splits.X_test[fn_mask].mean() - splits.X_test[tp_mask].mean())
        .abs()
        .sort_values(ascending=False)
        .head(5)
    )

    return {
        "model": best_model_name,
        "test_total": int(len(y_true)),
        "false_positives": int(fp_mask.sum()),
        "false_negatives": int(fn_mask.sum()),
        "false_positive_rate": round(float(fp_mask.sum() / max((y_true == 0).sum(), 1)), 4),
        "false_negative_rate": round(float(fn_mask.sum() / max((y_true == 1).sum(), 1)), 4),
        "top_fp_distinguishing_features": {
            k: round(float(v), 4) for k, v in fp_diff.items()
        },
        "top_fn_distinguishing_features": {
            k: round(float(v), 4) for k, v in fn_diff.items()
        },
        "worst_false_positives": worst_fp[
            ["index", "phishing_prob", "SSLfinal_State", "URL_of_Anchor", "web_traffic"]
        ].to_dict(orient="records"),
        "worst_false_negatives": worst_fn[
            ["index", "phishing_prob", "SSLfinal_State", "URL_of_Anchor", "web_traffic"]
        ].to_dict(orient="records"),
    }


def _extract_feature_importance(
    pipeline: Pipeline,
    feature_names: List[str],
) -> pd.Series | None:
    """Extract feature importance or absolute coefficients when available."""
    selected = get_selected_feature_names(pipeline.named_steps["preprocessor"], feature_names)
    clf = pipeline.named_steps["classifier"]
    if hasattr(clf, "feature_importances_"):
        return pd.Series(clf.feature_importances_, index=selected).sort_values(ascending=False)
    if hasattr(clf, "coef_"):
        return pd.Series(np.abs(clf.coef_[0]), index=selected).sort_values(ascending=False)
    return None


def generate_plots(
    pipelines: Dict[str, Pipeline],
    test_results: Dict[str, Dict[str, Any]],
    splits: DatasetSplits,
    best_model_name: str,
    plots_dir: Path = PLOTS_DIR,
) -> List[Path]:
    """Generate and save model comparison, ROC/PR, confusion matrix, and error plots."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    saved_paths: List[Path] = []

    y_test = splits.y_test.to_numpy()
    model_names = list(pipelines.keys())

    # 1. Model Comparison Bar Chart
    metrics_to_plot = ["precision", "recall", "f1", "roc_auc", "pr_auc"]
    rows = []
    for name in model_names:
        for m in metrics_to_plot:
            rows.append(
                {
                    "Model": name,
                    "Metric": m.upper().replace("_", "-"),
                    "Score": test_results[name][m],
                }
            )
    comp_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(data=comp_df, x="Metric", y="Score", hue="Model", ax=ax, palette="viridis")
    ax.set_ylim(0.5, 1.02)
    ax.set_title("Held-Out Test Set Performance Across Models", fontsize=13, fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.legend(title="Model", loc="lower right")
    fig.tight_layout()
    p1 = plots_dir / "model_comparison.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    saved_paths.append(p1)

    # 2. ROC & Precision-Recall Curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, pipe in pipelines.items():
        prob = pipe.predict_proba(splits.X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, prob)
        prec, rec, _ = precision_recall_curve(y_test, prob)
        axes[0].plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{name} (AUC={test_results[name]['roc_auc']:.3f})",
        )
        axes[1].plot(
            rec,
            prec,
            linewidth=2,
            label=f"{name} (AP={test_results[name]['pr_auc']:.3f})",
        )
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[0].set_title("ROC Curves (Test Set)", fontweight="bold")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate (Recall)")
    axes[0].legend(loc="lower right", fontsize=8.5)

    axes[1].set_title("Precision-Recall Curves (Test Set)", fontweight="bold")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="lower left", fontsize=8.5)

    fig.tight_layout()
    p2 = plots_dir / "roc_pr_curves.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    saved_paths.append(p2)

    # 3. Confusion Matrices Grid
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 8))
    for ax, name in zip(axes.ravel(), model_names):
        cm_dict = test_results[name]["confusion_matrix"]
        cm_arr = np.array([[cm_dict["tn"], cm_dict["fp"]], [cm_dict["fn"], cm_dict["tp"]]])
        sns.heatmap(
            cm_arr,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Legit (0)", "Phishing (1)"],
            yticklabels=["Legit (0)", "Phishing (1)"],
        )
        ax.set_title(f"{name} (F1={test_results[name]['f1']:.3f})", fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.suptitle("Confusion Matrices on Held-Out Test Set", fontsize=13, fontweight="bold")
    fig.tight_layout()
    p3 = plots_dir / "confusion_matrices.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    saved_paths.append(p3)

    # 4. Feature Importance & Error Confidence Distribution (Best Model)
    best_pipe = pipelines[best_model_name]
    y_prob = best_pipe.predict_proba(splits.X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    fp_probs = y_prob[(y_test == 0) & (y_pred == 1)]
    fn_probs = y_prob[(y_test == 1) & (y_pred == 0)]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    imp = _extract_feature_importance(best_pipe, splits.feature_names)
    if imp is not None:
        top_imp = imp.head(12).iloc[::-1]
        top_imp.plot(kind="barh", ax=axes[0], color="#2a7b88")
        axes[0].set_title(f"Top 12 Features ({best_model_name})", fontweight="bold")
        axes[0].set_xlabel("Importance / |Coefficient|")
    else:
        axes[0].text(0.5, 0.5, "Feature importance not available", ha="center")

    if len(fp_probs) > 0 or len(fn_probs) > 0:
        if len(fp_probs) > 0:
            axes[1].hist(fp_probs, bins=10, alpha=0.7, label=f"False Positives (n={len(fp_probs)})", color="#e07a5f")
        if len(fn_probs) > 0:
            axes[1].hist(fn_probs, bins=10, alpha=0.7, label=f"False Negatives (n={len(fn_probs)})", color="#3d405b")
        axes[1].axvline(0.5, color="black", linestyle="--", linewidth=1, label="Decision Threshold (0.5)")
        axes[1].set_xlim(0, 1)
        axes[1].set_title(f"Error Confidence Distribution ({best_model_name})", fontweight="bold")
        axes[1].set_xlabel("Predicted Phishing Probability")
        axes[1].set_ylabel("Count")
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "Zero errors on test set!", ha="center")

    fig.tight_layout()
    p4 = plots_dir / "error_analysis_and_importance.png"
    fig.savefig(p4, dpi=150)
    plt.close(fig)
    saved_paths.append(p4)

    return saved_paths


def run_training_pipeline(
    random_state: int = 42,
    save_artifacts: bool = True,
) -> Dict[str, Any]:
    """Run full end-to-end loading, splitting, tuning, evaluation, and artifact saving."""
    df = load_dataset()
    dataset_summary = get_dataset_summary(df)
    splits = split_dataset(df, random_state=random_state)

    pipelines, tuning_records = tune_on_validation(splits, random_state=random_state)

    test_results: Dict[str, Dict[str, Any]] = {}
    for name, pipe in pipelines.items():
        test_prob = pipe.predict_proba(splits.X_test)[:, 1]
        test_pred = (test_prob >= 0.5).astype(int)
        test_results[name] = compute_classification_metrics(
            splits.y_test.to_numpy(), test_pred, test_prob
        )

    # Pick best model strictly using Validation F1 + Validation ROC-AUC (no test leakage!)
    best_model_name = max(
        tuning_records.keys(),
        key=lambda n: (
            tuning_records[n]["validation_metrics"]["f1"],
            tuning_records[n]["validation_metrics"]["roc_auc"],
        ),
    )

    error_report = analyze_errors(
        best_model_name=best_model_name,
        best_pipeline=pipelines[best_model_name],
        splits=splits,
    )

    report = {
        "dataset_summary": dataset_summary,
        "split_sizes": {
            "train": int(len(splits.X_train)),
            "validation": int(len(splits.X_val)),
            "test": int(len(splits.X_test)),
        },
        "dropped_columns": splits.dropped_columns,
        "best_model_name": best_model_name,
        "tuning_records": tuning_records,
        "test_results": test_results,
        "error_analysis": error_report,
    }

    if save_artifacts:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipelines[best_model_name], MODELS_DIR / "best_model.joblib")
        joblib.dump(pipelines, MODELS_DIR / "all_models.joblib")
        with open(ARTIFACTS_DIR / "metrics_summary.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        generate_plots(
            pipelines=pipelines,
            test_results=test_results,
            splits=splits,
            best_model_name=best_model_name,
            plots_dir=PLOTS_DIR,
        )

    return report


if __name__ == "__main__":
    summary = run_training_pipeline()
    print(f"Best model (selected on Validation): {summary['best_model_name']}")
    print("Split sizes:", summary["split_sizes"])
    for model_name, metrics in summary["test_results"].items():
        print(
            f"  {model_name:20s} | "
            f"Prec={metrics['precision']:.4f} | "
            f"Rec={metrics['recall']:.4f} | "
            f"F1={metrics['f1']:.4f} | "
            f"ROC-AUC={metrics['roc_auc']:.4f} | "
            f"PR-AUC={metrics['pr_auc']:.4f}"
        )
