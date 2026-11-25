"""Fixed binary validation-only workflow for CET433 M03.1."""

from dataclasses import dataclass
from time import perf_counter
import warnings


SEED = 42
TOTAL_ROWS = 175_341
FIT_ROWS = 140_273
VALIDATION_ROWS = 35_068
FIT_BUDGET = 3
MODEL_ORDER = ("logistic_regression", "random_forest", "lightgbm")
RECALL_MIN_PERCENT = 95
FPR_STRICT_UPPER_PERCENT = 1
FP_COST = 5
FN_COST = 120


def model_pipelines(features):
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    from ids_utils import make_preprocessor

    common = {"class_weight": "balanced", "random_state": SEED}
    estimators = {
        "logistic_regression": LogisticRegression(solver="lbfgs", C=1.0, max_iter=1000, **common),
        "random_forest": RandomForestClassifier(n_estimators=100, max_depth=15,
                                                min_samples_leaf=2, n_jobs=2, **common),
        "lightgbm": LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31,
                                   min_child_samples=20, n_jobs=2, verbose=-1, **common),
    }
    return {
        name: Pipeline([("pre", make_preprocessor(features)), ("clf", estimator)])
        for name, estimator in estimators.items()
    }


@dataclass
class Partition:
    features: object
    labels: object


def partitions(raw, plan, feature_columns):
    from ids_utils import prepare_features

    fit, validation = plan.fit, plan.validation
    if (len(raw) != TOTAL_ROWS or len(plan.assignments) != TOTAL_ROWS
            or len(plan.groups) != TOTAL_ROWS or len(fit) != FIT_ROWS
            or len(validation) != VALIDATION_ROWS):
        raise ValueError("Stage A requires the complete frozen M02.2 split.")
    if not set(plan.groups[fit]).isdisjoint(plan.groups[validation]):
        raise ValueError("Exact feature groups overlap between fit and validation.")
    features = prepare_features(raw, feature_columns)
    labels = raw["label"]
    return (Partition(features.iloc[fit], labels.iloc[fit]),
            Partition(features.iloc[validation], labels.iloc[validation]))


def metrics(y, scores, threshold):
    positive = scores >= threshold
    tp = int((positive & y).sum())
    fp = int((positive & ~y).sum())
    fn = int(y.sum()) - tp
    tn = len(y) - tp - fp - fn
    return {
        "threshold": threshold, "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn), "f1": (2 * tp) / (2 * tp + fp + fn),
        "fpr": fp / (fp + tn), "cost_minutes": FP_COST * fp + FN_COST * fn,
    }


def validation_report(labels, scores):
    import numpy as np
    from sklearn.metrics import roc_auc_score

    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    if labels.ndim != 1 or scores.shape != labels.shape or set(labels) != {0, 1}:
        raise ValueError("Validation requires one probability per row and both binary classes.")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Validation probabilities must be finite and in [0, 1].")

    y = labels == 1
    order = np.argsort(scores)
    sorted_scores = scores[order]
    thresholds = np.unique(np.r_[0.0, sorted_scores, 1.0])
    starts = np.searchsorted(sorted_scores, thresholds, side="left")
    cumulative_positive = np.r_[0, np.cumsum(y[order])]
    positives = int(cumulative_positive[-1])
    negatives = len(y) - positives
    fn = cumulative_positive[starts]
    fp = negatives - (starts - fn)

    # Integer comparisons preserve the strict FPR boundary and inclusive recall boundary.
    feasible = ((positives - fn) * 100 >= RECALL_MIN_PERCENT * positives) & (
        fp * 100 < FPR_STRICT_UPPER_PERCENT * negatives)
    joint_feasible = bool(feasible.any())
    candidates = np.flatnonzero(feasible) if joint_feasible else np.arange(len(thresholds))
    costs = FP_COST * fp + FN_COST * fn
    rank = np.lexsort((-thresholds[candidates], fp[candidates], costs[candidates]))
    threshold = float(thresholds[candidates[rank[0]]])

    return {
        "support": {"rows": len(y), "normal": negatives, "attack": positives},
        "attack_prevalence": positives / len(y),
        "roc_auc": roc_auc_score(y, scores),
        "at_0_5": metrics(y, scores, 0.5),
        "threshold_selection": {
            "status": "FEASIBLE" if joint_feasible else "INFEASIBLE",
            "purpose": "feasible_operating_point" if joint_feasible else "descriptive_benchmark",
            **metrics(y, scores, threshold),
        },
    }


def run_models(pipelines, fit, validation, record, persist):
    """Fit each approved pipeline once, journaling the attempt before calling fit."""
    if tuple(pipelines) != MODEL_ORDER or len(pipelines) != FIT_BUDGET:
        raise ValueError("Stage A requires exactly the three approved binary candidates in order.")

    reports = {}
    for index, (name, pipe) in enumerate(pipelines.items(), start=1):
        prefix = f"{index:02d}-{name}"
        record(f"{prefix}.fit-started.json", {"model": name, "attempt": index})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            started = perf_counter()
            try:
                pipe.fit(fit.features, fit.labels)
            except BaseException as exc:
                record(f"{prefix}.fit-result.json", fit_result(started, caught, exc))
                raise
            else:
                result = fit_result(started, caught)
                record(f"{prefix}.fit-result.json", result)

        if list(pipe.classes_) != [0, 1]:
            raise ValueError("The fitted model must expose class probabilities in [0, 1] order.")
        report = validation_report(validation.labels, pipe.predict_proba(validation.features)[:, 1])
        artifact = persist(name, pipe, report["threshold_selection"]["threshold"])
        reports[name] = {"fit": result, "validation": report, "artifact": artifact}
        record(f"{prefix}.validation.json", reports[name])
    return reports


def fit_result(started, caught, failure=None):
    return {
        "status": ("complete" if failure is None else
                   "interrupted" if isinstance(failure, KeyboardInterrupt) else "failed"),
        "seconds": perf_counter() - started,
        "warnings": [{"category": item.category.__name__, "message": str(item.message)}
                     for item in caught],
        "failure": None if failure is None else {
            "type": type(failure).__name__, "message": str(failure),
        },
    }


def selection(reports):
    # max keeps the first candidate on an exact tie, as in the reviewed notebook.
    winner = max(MODEL_ORDER[1:], key=lambda name: reports[name]["validation"]["at_0_5"]["f1"])
    return {
        "model": winner,
        "rule": "tree validation F1 at 0.5; Random Forest first on ties",
        **reports[winner]["validation"]["threshold_selection"],
    }
