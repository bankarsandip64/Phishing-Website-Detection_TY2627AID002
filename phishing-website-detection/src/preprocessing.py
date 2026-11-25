"""Leak-free preprocessing and 60/20/20 train/validation/test splitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_loader import TARGET_COLUMN

# Columns that should never be used as model inputs if present
EXACT_EXCLUDED_COLUMNS = {"id", "index", "url_id", "row_id", "timestamp", "email", "ip_raw"}


@dataclass
class DatasetSplits:
    """Container for 60/20/20 train, validation, and test splits."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    feature_names: List[str]
    dropped_columns: List[str]


def encode_target(series: pd.Series) -> pd.Series:
    """Map raw UCI target (-1=Phishing, 1=Legitimate) to binary (1=Phishing, 0=Legitimate)."""
    unique_vals = set(series.unique())
    if unique_vals <= {-1, 1}:
        return (series == -1).astype(int)
    if unique_vals <= {0, 1}:
        return series.astype(int)
    raise ValueError(f"Unexpected target values: {sorted(unique_vals)}")


def identify_removable_columns(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
) -> List[str]:
    """Identify ID-like, constant, or unnecessary columns to exclude from model inputs."""
    dropped: List[str] = []
    for col in df.columns:
        if col == target_col:
            continue
        col_lower = col.lower()
        if col_lower in EXACT_EXCLUDED_COLUMNS:
            dropped.append(col)
            continue
        if df[col].nunique(dropna=False) <= 1:
            dropped.append(col)
    return dropped


def split_dataset(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    random_state: int = 42,
) -> DatasetSplits:
    """Create a stratified 60% / 20% / 20% Train / Validation / Test split."""
    dropped_cols = identify_removable_columns(df, target_col=target_col)
    feature_cols = [c for c in df.columns if c != target_col and c not in dropped_cols]

    X = df[feature_cols].copy()
    y = encode_target(df[target_col])

    # 60% train, 40% temp (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.40,
        stratify=y,
        random_state=random_state,
    )

    # Split 40% temp equally into 20% validation and 20% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=random_state,
    )

    return DatasetSplits(
        X_train=X_train.reset_index(drop=True),
        X_val=X_val.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_val=y_val.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        feature_names=feature_cols,
        dropped_columns=dropped_cols,
    )


def build_preprocessor(scale_features: bool = True) -> Pipeline:
    """Build a leak-free preprocessing pipeline to be fit ONLY on training data."""
    steps: List[Tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("variance_filter", VarianceThreshold(threshold=0.0)),
    ]
    if scale_features:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps=steps)


def get_selected_feature_names(
    preprocessor: Pipeline,
    input_features: Sequence[str],
) -> List[str]:
    """Return feature names surviving the variance filter."""
    variance_filter = preprocessor.named_steps.get("variance_filter")
    if variance_filter is None:
        return list(input_features)
    mask = variance_filter.get_support()
    return [name for name, keep in zip(input_features, mask) if keep]
