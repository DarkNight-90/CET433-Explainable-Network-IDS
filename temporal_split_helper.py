"""Chronological split helper for timestamped network-flow datasets.

This deliberately refuses to invent time values. Supply a real timestamp column
from the source dataset and document its meaning.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

def random_split(df, target="label", test_size=0.2, seed=42):
    return train_test_split(df, test_size=test_size, stratify=df[target], random_state=seed)

def chronological_split(df, timestamp_col, target="label", train_fraction=0.7, val_fraction=0.15):
    if timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column {timestamp_col!r} is not present.")
    data = df.copy()
    data[timestamp_col] = pd.to_datetime(data[timestamp_col], errors="coerce", utc=True)
    if data[timestamp_col].isna().any():
        raise ValueError("Timestamp column contains missing/unparseable values; resolve them from source data.")
    data = data.sort_values(timestamp_col).reset_index(drop=True)
    n = len(data)
    a, b = int(n * train_fraction), int(n * (train_fraction + val_fraction))
    if a == 0 or b <= a or b >= n:
        raise ValueError("Invalid chronological split fractions.")
    return data.iloc[:a].copy(), data.iloc[a:b].copy(), data.iloc[b:].copy()

if __name__ == "__main__":
    print("Use chronological_split(df, timestamp_col='<REAL_TIMESTAMP_COLUMN>') only when your dataset contains a real timestamp.")
