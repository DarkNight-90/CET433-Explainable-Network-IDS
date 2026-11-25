"""Verify frozen Phase 1 evidence without fitting or reading the official test CSV."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from colab_acceptance import load_verified_split
from stage_a import FN_COST, FP_COST

OFFICIAL = "Official UNSW-NB15 training CSV"
FIT = "M02.2 internal fit"
VALID = "M02.2 internal validation"
TRAINING = ROOT / "data/official/UNSW_NB15_training-set.csv"
ASSIGNMENT = ROOT / "artifacts/generated/m02.2-split-assignments.json"
CATEGORICAL = ("proto", "service", "state")


def frequencies(frame):
    rows = []
    for col in ("label", "attack_cat", *CATEGORICAL):
        counts = frame[col].value_counts(dropna=False)
        rows.extend((col, value, n, 100 * n / len(frame)) for value, n in counts.items())
    return pd.DataFrame(rows, columns=["column", "value", "rows", "percent"])


def profile(frame, groups):
    sizes = pd.Series(groups).value_counts()
    targets = frame[["label", "attack_cat"]].assign(group=groups).groupby("group").nunique()
    numbers = frame.select_dtypes(include="number")
    summary = {
        "observations": len(frame),
        "original columns": len(frame.columns),
        "integer columns": int(frame.dtypes.eq("int64").sum()),
        "floating-point columns": int(frame.dtypes.eq("float64").sum()),
        "text columns": len(frame.columns) - len(numbers.columns),
        "numeric model inputs": len(numbers.columns) - 2,
        "categorical model inputs": len(CATEGORICAL),
        "missing cells": int(frame.isna().to_numpy().sum()),
        "blank/whitespace text cells": sum(int(frame[c].str.fullmatch(r"\s*").sum())
                                          for c in frame.select_dtypes(exclude="number")),
        "infinite numeric cells": int(np.isinf(numbers).to_numpy().sum()),
        "extra duplicate rows (all columns)": int(frame.duplicated().sum()),
        "extra duplicate rows (excluding id)": int(frame.drop(columns="id").duplicated().sum()),
        "distinct model feature groups": len(sizes),
        "extra duplicate feature vectors": len(frame) - len(sizes),
        "repeated feature groups": int(sizes.gt(1).sum()),
        "rows in repeated feature groups": int(sizes[sizes.gt(1)].sum()),
        "largest feature group (rows)": int(sizes.max()),
    }
    for target, name in [("label", "binary label"), ("attack_cat", "family label")]:
        conflict = targets.index[targets[target].gt(1)]
        summary[f"groups with conflicting {name}"] = len(conflict)
        summary[f"rows in groups with conflicting {name}"] = int(sizes.loc[conflict].sum())
    return summary


def baseline_metrics(baseline, support):
    tp, fp, tn, fn = (baseline[key] for key in ("TP", "FP", "TN", "FN"))
    n = tp + fp + tn + fn
    if (n != support["rows"] or n != baseline["observations"]
            or {"0": tn + fp, "1": tp + fn} != support["binary"]):
        raise ValueError("Baseline counts do not match frozen validation support.")
    metrics = {
        "precision": tp / (tp + fp), "recall": tp / (tp + fn),
        "f1": 2 * tp / (2 * tp + fp + fn), "fpr": fp / (fp + tn),
    }
    for name, value in metrics.items():
        if round(value, 6) != baseline[name]:
            raise ValueError(f"Baseline {name} differs from the source.")
    return {**metrics, "normal recall": tn / (tn + fp), "accuracy": (tn + tp) / n,
            "cost minutes": FP_COST * fp + FN_COST * fn}


def figures(raw, freq, baseline, output):
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    blue, orange = "#126782", "#A64B28"

    def save(fig, name):
        for extension in ("png", "pdf"):
            fig.savefig(output / f"{name}.{extension}", dpi=240, bbox_inches="tight")
        plt.close(fig)

    def bars(ax, field, *, labels=None, limit=None):
        counts = freq.loc[freq.column.eq(field)].set_index("value")["rows"]
        if limit and len(counts) > limit:
            counts = pd.concat([counts.iloc[:limit], pd.Series({"Other (pooled)": counts.iloc[limit:].sum()})])
        if labels:
            counts = counts.rename(index=labels)
        counts = counts.iloc[::-1]
        rects = ax.barh(counts.index.astype(str), counts, color=blue)
        ax.bar_label(rects, labels=[f"{n:,}" for n in counts], padding=4, fontsize=9)
        ax.set_xlim(0, counts.max() * 1.32)
        ax.set_xlabel("Rows")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
    fig.suptitle(f"{OFFICIAL} | Label distributions (n={len(raw):,})", weight="bold")
    bars(axes[0], "label", labels={0: "Normal", 1: "Attack"})
    bars(axes[1], "attack_cat")
    axes[0].set_title("Binary label")
    axes[1].set_title("Attack family and Normal")
    save(fig, "labels")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), layout="constrained")
    fig.suptitle(f"{OFFICIAL} | Categorical inputs (n={len(raw):,})", weight="bold")
    for ax, field in zip(axes, CATEGORICAL):
        bars(ax, field, limit=10)
        ax.set_title(field)
    save(fig, "categorical")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    fig.suptitle(f"{OFFICIAL} | Selected numeric inputs (n={len(raw):,})", weight="bold")
    for ax, field in zip(axes.flat, ("dur", "sbytes", "dbytes", "rate")):
        if raw[field].lt(0).any():
            raise ValueError(f"Display log1p requires nonnegative values for {field}.")
        edges = np.histogram_bin_edges(np.log1p(raw[field]), bins=40)
        for label, name, color in [(0, "Normal", blue), (1, "Attack", orange)]:
            values = np.log1p(raw.loc[raw.label.eq(label), field])
            ax.hist(values, bins=edges, weights=np.full(len(values), 100 / len(values)),
                    histtype="step", linewidth=1.8, label=name, color=color)
        ax.set(xlabel=f"log1p({field})", ylabel="Within-class rows per bin (%)")
        ax.legend(frameon=False)
    save(fig, "numeric")

    counts = np.array([[baseline["TN"], baseline["FP"]], [baseline["FN"], baseline["TP"]]])
    proportions = counts / counts.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6, 4.9), layout="constrained")
    ax.imshow(proportions, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    for (i, j), count in np.ndenumerate(counts):
        percent = proportions[i, j]
        ax.text(j, i, f"{count:,}\n{percent:.2%}", ha="center", va="center",
                color="white" if percent > .5 else "#142538", fontsize=13, weight="bold")
    ax.set_xticks([0, 1], ["Normal", "Attack"])
    ax.set_yticks([0, 1], ["Normal", "Attack"])
    ax.set(xlabel="Predicted label", ylabel="Actual label",
           title=f"{VALID} (n={baseline['observations']:,})\nStage A Logistic Regression | Threshold {baseline['threshold']}")
    save(fig, "confusion-matrix")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional directory for regenerated evidence")
    args = parser.parse_args()
    if TRAINING.resolve() != TRAINING or ASSIGNMENT.resolve() != ASSIGNMENT:
        raise ValueError("Expected local training and assignment files at their verified paths.")
    raw, plan, split = load_verified_split(ROOT, TRAINING, ASSIGNMENT)
    baseline = json.loads((HERE / "baseline.json").read_text())
    metrics = baseline_metrics(baseline, split["support"]["validation"])
    print(f"{OFFICIAL}: {raw.shape}; SHA-256 {split['training_sha256']}")
    print(f"{FIT}: {len(plan.fit):,}; {VALID}: {len(plan.validation):,}")
    print(f"Exact-feature overlaps: fit/validation {split['fit_validation_group_overlap']}, "
          f"CV folds {split['cv_group_overlap']}")
    print(f"{baseline['partition']}: {metrics}; reported ROC-AUC {baseline['roc_auc']}")
    if args.output is None:
        return

    summaries, missing, numeric, freq_frames = {}, [], [], []
    for name, positions in [(OFFICIAL, slice(None)), (FIT, plan.fit), (VALID, plan.validation)]:
        frame = raw.iloc[positions]
        summaries[name] = profile(frame, plan.groups[positions])
        counts = frame.isna().sum().rename("missing_rows").to_frame()
        counts["percent"] = 100 * counts["missing_rows"] / len(frame)
        missing.append(counts.rename_axis("column").reset_index().assign(partition=name))
        desc = frame.select_dtypes(include="number").drop(columns=["id", "label"]).describe().T
        numeric.append(desc.rename_axis("feature").reset_index().assign(partition=name))
        freq_frames.append(frequencies(frame).assign(partition=name))

    output = args.output
    charts = output / "figures"
    charts.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).rename_axis("measure").to_csv(output / "profile.tsv", sep="\t")
    for filename, frames in [("missing", missing), ("numeric-summary", numeric),
                             ("distributions", freq_frames)]:
        pd.concat(frames, ignore_index=True).to_csv(output / f"{filename}.tsv", sep="\t", index=False)
    (output / "split-verification.json").write_text(json.dumps(split, indent=2) + "\n")
    figures(raw, freq_frames[0], baseline, charts)


if __name__ == "__main__":
    main()
