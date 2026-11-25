"""Shared training and inference helpers, including the serialized log_clip symbol."""

from pathlib import Path

import numpy as np
import pandas as pd


TARGET_COLS = ["label", "attack_cat"]
CATEGORICAL_COLS = ("proto", "service", "state")


def log_clip(a):
    """Apply log1p after clipping negative values to zero."""
    return np.log1p(np.clip(a, 0, None))


def _validate_frame(df):
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Expected a non-empty flow table.")
    if not df.columns.is_unique:
        raise ValueError("Flow table contains duplicate column names.")


def read_flow_csv(source):
    """Read a CSV path or seekable upload and reject ambiguous headers."""
    stream = not isinstance(source, (str, Path))
    if stream:
        source.seek(0)
    try:
        header = pd.read_csv(source, header=None, nrows=1, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("Expected a CSV with a header and at least one flow.") from exc
    if header.empty or header.iloc[0].str.strip().eq("").any() or header.iloc[0].duplicated().any():
        raise ValueError("CSV feature headers must be non-blank and unique.")
    if stream:
        source.seek(0)
    df = pd.read_csv(source)
    _validate_frame(df)
    if not isinstance(df.index, pd.RangeIndex):
        raise ValueError("CSV contains more fields than its header declares.")
    return df


def clean(df):
    """Validate labeled flows and apply the existing training-time cleaning."""
    _validate_frame(df)
    features = df.drop(columns=["id"] + TARGET_COLS, errors="ignore").columns.tolist()
    missing = [name for name in (*TARGET_COLS, *CATEGORICAL_COLS) if name not in df]
    if missing:
        raise ValueError(f"Missing required training columns: {missing}")
    data = prepare_features(df, features)
    labels = df["label"]
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("label must contain only 0 (Normal) or 1 (Attack).")
    categories = df["attack_cat"]
    if categories.isna().any() or not categories.map(lambda value: isinstance(value, str)).all():
        raise ValueError("attack_cat must contain non-missing text labels.")
    categories = categories.str.strip()
    if categories.eq("").any() or not categories.eq("Normal").eq(labels.eq(0)).all():
        raise ValueError("attack_cat must be non-blank and agree with the binary label.")
    data["label"] = labels.astype(int)
    data["attack_cat"] = categories
    return data.drop_duplicates().reset_index(drop=True)


def prepare_features(df, feature_cols):
    """Return a validated, ordered feature table; preserve missing feature values."""
    _validate_frame(df)
    columns = list(feature_cols)
    if not columns or len(columns) != len(set(columns)) or set(columns) & {"id", *TARGET_COLS}:
        raise ValueError("Expected a non-empty, unique list of model feature columns.")
    missing = [name for name in columns if name not in df]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    data = df.loc[:, columns].copy()
    for name in columns:
        values = data[name]
        if name in CATEGORICAL_COLS:
            if not values.dropna().map(lambda value: isinstance(value, str)).all():
                raise ValueError(f"Feature {name!r} must contain text or missing values.")
            data[name] = values.astype(object).where(values.notna(), np.nan)
        else:
            try:
                data[name] = pd.to_numeric(values, errors="raise").replace([np.inf, -np.inf], np.nan)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Feature {name!r} must contain numbers or missing values.") from exc
    return data


def make_preprocessor(features):
    """Build the approved, unfitted preprocessing pipeline for the ordered schema."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

    categorical = [name for name in features if name in CATEGORICAL_COLS]
    numeric = [name for name in features if name not in CATEGORICAL_COLS]
    return ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("log", FunctionTransformer(log_clip, feature_names_out="one-to-one")),
            ("sc", StandardScaler()),
        ]), numeric),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.005,
                              sparse_output=False), categorical),
    ])
