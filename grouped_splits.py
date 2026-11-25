"""Versioned feature fingerprints and training-only split assignments."""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import PredefinedSplit, StratifiedGroupKFold

from colab_preflight import verify_partition
from ids_utils import CATEGORICAL_COLS, read_flow_csv


POLICY = "cet433-exact-groups-v1"
N_FOLDS = 5


def feature_groups(features, ordered_columns):
    """Return stable SHA-256 group IDs for the supplied ordered feature table."""
    columns = list(ordered_columns)
    if (list(features.columns) != columns or not columns or not features.columns.is_unique
            or set(columns) & {"id", "label", "attack_cat"}):
        raise ValueError("Feature columns must match the ordered feature schema exactly.")
    normalized = []
    for name in columns:
        series = features[name]
        if name in CATEGORICAL_COLS:
            values = series.astype(object).where(series.notna(), None).tolist()
            if any(value is not None and not isinstance(value, str) for value in values):
                raise ValueError(f"{name}: expected text or a missing value.")
        else:
            numbers = pd.to_numeric(series, errors="raise")
            if pd.api.types.is_complex_dtype(numbers) or ((numbers > 2**53) | (numbers < -(2**53))).any():
                raise ValueError(f"{name}: outside the safe numeric normalization range.")
            values = []
            for value in numbers.to_numpy(dtype="float64", na_value=np.nan):
                if np.isinf(value):
                    raise ValueError(f"{name}: infinite value.")
                values.append(None if np.isnan(value) else float(0 if value == 0 else value).hex())
        normalized.append(values)

    # Exact tuple comparison resolves equality first. The reverse map rejects hash collisions.
    tuple_to_group = {}
    group_to_tuple = {}
    for key in zip(*normalized):
        if key not in tuple_to_group:
            payload = json.dumps([POLICY, columns, key], ensure_ascii=True, separators=(",", ":"))
            group = hashlib.sha256(payload.encode("ascii")).hexdigest()
            if group in group_to_tuple and group_to_tuple[group] != key:
                raise ValueError("Feature fingerprint collision.")
            tuple_to_group[key] = group
            group_to_tuple[group] = key
    return np.array([tuple_to_group[key] for key in zip(*normalized)], dtype=object)


@dataclass
class SplitPlan:
    groups: np.ndarray
    assignments: str

    @property
    def fit(self):
        return np.fromiter((i for i, code in enumerate(self.assignments) if code != "V"), dtype=int)

    @property
    def validation(self):
        return np.fromiter((i for i, code in enumerate(self.assignments) if code == "V"), dtype=int)

    def cv(self):
        """Supply the precomputed group-aware folds in fit-row order to sklearn."""
        return PredefinedSplit([int(code) for code in self.assignments if code != "V"])

    @property
    def assignment_sha256(self):
        return hashlib.sha256(self.assignments.encode("ascii")).hexdigest()


def build_split_plan(features, ordered_columns, categories, labels, *, seed=42):
    """Assign all official-training observations, with no test input or row sampling."""
    groups = feature_groups(features, ordered_columns)
    y = np.asarray(categories)
    binary = np.asarray(labels)
    if len(y) != len(groups) or len(binary) != len(groups):
        raise ValueError("Features and targets must have the same row count.")
    if pd.isna(y).any() or set(binary) != {0, 1} or not np.array_equal(y == "Normal", binary == 0):
        raise ValueError("Both binary classes and consistent nonmissing attack categories are required.")

    expected = set(y)
    counts = pd.Series(y).value_counts()

    def supported(indices):
        return set(binary[indices]) == {0, 1} and set(y[indices]) == expected

    # A held-out fold of five gives roughly 20%. Rank feasible candidates using training data only.
    candidates = []
    outer = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for fold, (fit, held) in enumerate(outer.split(binary, y, groups)):
        if not supported(fit) or not supported(held):
            continue
        share = pd.Series(y[held]).value_counts().div(counts)
        score = (abs(len(held) / len(y) - 0.2) + (share - 0.2).abs().mean(), fold)
        candidates.append((score, fit, held))

    for _, fit, held in sorted(candidates, key=lambda item: item[0]):
        assignments = np.full(len(y), "V", dtype="U1")
        inner = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for fold, (cv_train, cv_held) in enumerate(inner.split(binary[fit], y[fit], groups[fit])):
            if not supported(fit[cv_train]) or not supported(fit[cv_held]):
                break
            if not set(groups[fit[cv_train]]).isdisjoint(groups[fit[cv_held]]):
                raise ValueError("Feature group crosses a CV boundary.")
            assignments[fit[cv_held]] = str(fold)
        else:
            if not set(groups[fit]).isdisjoint(groups[held]) or np.any(assignments[fit] == "V"):
                raise ValueError("Incomplete assignments or feature group crossing validation.")
            return SplitPlan(groups, "".join(assignments))
    raise ValueError("Could not retain both binary classes and every attack category in all splits.")


def model_feature_columns(manifest):
    schema = manifest["schema"]
    excluded = {schema["identifier"], *schema["targets"]}
    columns = [name for name in schema["ordered_columns"] if name not in excluded]
    if excluded != {"id", "label", "attack_cat"} or len(columns) != 42:
        raise ValueError("Expected the official UNSW-NB15 schema.")
    return columns


def split_record(plan, source, training_sha256, *, seed=42):
    """Describe assignments and their support, without recording raw row data."""
    def support(indices):
        partition = source.iloc[indices]
        return {
            "rows": len(indices),
            "groups": len(set(plan.groups[indices])),
            "binary": partition.label.value_counts().sort_index().to_dict(),
            "attack_categories": partition.attack_cat.value_counts().sort_index().to_dict(),
        }

    fit = plan.fit
    return {
        "policy": POLICY,
        "training_sha256": training_sha256,
        "assignment_sha256": plan.assignment_sha256,
        "assignment_encoding": "One ASCII code per original CSV data row in order: V=validation, 0-4=CV held-out fold.",
        "seed": seed,
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__},
        "fit": support(fit),
        "validation": support(plan.validation),
        "cv": [{"fold": fold, "fit": support(fit[a]), "held_out": support(fit[b])}
               for fold, (a, b) in enumerate(plan.cv().split())],
    }


def write_assignment(path, plan, record):
    """Persist the exact positions and bind them to the input digest."""
    if record["assignment_sha256"] != plan.assignment_sha256:
        raise ValueError("Assignment record does not match its row codes.")
    with Path(path).open("x", encoding="utf-8") as out:
        json.dump({**record, "assignments": plan.assignments}, out, indent=2, sort_keys=True)
        out.write("\n")


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=root / "data/official")
    parser.add_argument("--assignments", type=Path, required=True, help="New JSON output path")
    args = parser.parse_args()
    manifest = json.loads((root / "data/dataset_manifest.json").read_text(encoding="utf-8"))
    schema = manifest["schema"]["ordered_columns"]
    reference = manifest["files"]["training"]
    source = args.data_dir / "UNSW_NB15_training-set.csv"
    verified = verify_partition(source, schema, reference)
    raw = read_flow_csv(source)
    columns = model_feature_columns(manifest)
    plan = build_split_plan(raw[columns], columns, raw.attack_cat, raw.label)
    record = split_record(plan, raw, verified["sha256"])
    write_assignment(args.assignments, plan, record)
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
