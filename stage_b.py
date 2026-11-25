"""Frozen two-tree search, with an explicit journaled fit per candidate and fold."""

import os
from time import perf_counter
import warnings

from stage_a import fit_result, selection, validation_report


MODELS = ("random_forest", "lightgbm")
CANDIDATES_PER_MODEL = 5
FOLDS = 5
CV_LIMIT = 50
REFIT_LIMIT = 2
SCORER = "f1"
THREADS = 2
THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS", "NUMEXPR_MAX_THREADS",
)
RF_KEYS = ("n_estimators", "max_depth", "min_samples_leaf")
LGBM_KEYS = ("n_estimators", "learning_rate", "num_leaves", "min_child_samples")

# Stage A settings first, followed by fixed contrasts within the approved ranges.
RF_CANDIDATES = ((100, 15, 2), (200, 15, 2), (200, 25, 1),
                 (100, 25, 5), (100, None, 2))
LGBM_CANDIDATES = ((200, 0.05, 31, 20), (400, 0.05, 31, 20),
                   (200, 0.1, 31, 20), (200, 0.05, 63, 20), (400, 0.05, 63, 50))


def configuration():
    candidates = {
        "random_forest": [dict(zip(RF_KEYS, values, strict=True)) for values in RF_CANDIDATES],
        "lightgbm": [dict(zip(LGBM_KEYS, values, strict=True)) for values in LGBM_CANDIDATES],
    }
    for entries in candidates.values():
        if (len(entries) != CANDIDATES_PER_MODEL
                or len({tuple(entry.items()) for entry in entries}) != CANDIDATES_PER_MODEL):
            raise ValueError("Stage B requires five distinct predeclared candidates per model.")
    return {
        "candidates": candidates, "cv_folds": FOLDS, "scoring": SCORER,
        "cv_workers": 1, "cv_refit": False, "estimator_threads": THREADS,
        "cv_fit_limit": CV_LIMIT, "final_refit_limit": REFIT_LIMIT,
        "cv_selection": "highest mean binary F1; first candidate on exact ties",
        "validation_selection": "tree validation F1 at 0.5; Random Forest first on ties",
    }


def verify_thread_environment():
    for name in THREAD_VARIABLES:
        if os.environ.get(name) not in {"1", "2"}:
            raise ValueError(f"Stage B requires {name}=1 or {name}=2 before Python starts.")


def frozen_folds(plan, labels):
    import numpy as np

    fit = plan.fit
    if len(fit) != len(labels):
        raise ValueError("CV labels must be in complete frozen fit-row order.")
    if len(plan.validation) == 0 or not set(plan.groups[fit]).isdisjoint(plan.groups[plan.validation]):
        raise ValueError("Frozen fit/validation exact feature groups overlap.")
    splits = tuple(plan.cv().split())
    if len(splits) != FOLDS:
        raise ValueError("Stage B requires exactly five frozen CV folds.")
    codes = np.array([int(plan.assignments[i]) for i in fit])
    groups = plan.groups[fit]
    labels = np.asarray(labels)
    for fold, (train, held) in enumerate(splits):
        if (not np.array_equal(train, np.flatnonzero(codes != fold))
                or not np.array_equal(held, np.flatnonzero(codes == fold))):
            raise ValueError("CV positions differ from frozen membership or fit-row order.")
        if not set(groups[train]).isdisjoint(groups[held]):
            raise ValueError("Exact feature group crosses a CV fold.")
        if set(labels[train]) != {0, 1} or set(labels[held]) != {0, 1}:
            raise ValueError("Both classes must appear on each side of every CV fold.")
    return splits


def pipelines(feature_columns, seed):
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline

    from ids_utils import make_preprocessor

    common = {"class_weight": "balanced", "random_state": seed, "n_jobs": THREADS}
    return {
        name: Pipeline([("pre", make_preprocessor(feature_columns)), ("clf", model)])
        for name, model in (
            ("random_forest", RandomForestClassifier(**common)),
            ("lightgbm", LGBMClassifier(verbose=-1, **common)),
        )
    }


class FitJournal:
    def __init__(self, record):
        self.record = record
        self.counts = dict.fromkeys(("cv", "refit"), 0)
        self.per_model = {name: dict(self.counts) for name in MODELS}

    def fit(self, pipe, features, labels, phase, report):
        limit = CV_LIMIT if phase == "cv" else REFIT_LIMIT
        model_counts = self.per_model[report["model"]]
        if self.counts[phase] >= limit or model_counts[phase] >= limit // len(MODELS):
            raise RuntimeError(f"Stage B {phase} fit budget exhausted.")
        prefix = f"{phase}-{self.counts[phase] + 1:02d}"
        self.record(f"{prefix}.fit-started.json", report)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            started = perf_counter()
            self.counts[phase] += 1
            model_counts[phase] += 1
            try:
                pipe.fit(features, labels)
            except BaseException as exc:
                self.record(f"{prefix}.fit-result.json", {**report, **fit_result(started, caught, exc)})
                raise
            result = fit_result(started, caught)
        self.record(f"{prefix}.fit-result.json", {**report, **result})
        return result


def candidate_pipeline(template, parameters):
    from sklearn.base import clone

    return clone(template).set_params(**{f"clf__{key}": value for key, value in parameters.items()})


def search(templates, fit, folds, journal, *, refit=False):
    """Run fixed CV serially so each actual fit and any failure can be recorded."""
    from statistics import mean, pstdev
    from sklearn.metrics import get_scorer

    if refit is not False:
        raise ValueError("Stage B CV requires refit=False.")
    if tuple(templates) != MODELS or len(folds) != FOLDS:
        raise ValueError("Unexpected Stage B model set or fold count.")
    scorer = get_scorer(SCORER)
    results = {}
    for name, candidates in configuration()["candidates"].items():
        scores = []
        for index, params in enumerate(candidates, start=1):
            fold_reports = []
            for fold, (training, held) in enumerate(folds):
                pipe = candidate_pipeline(templates[name], params)
                fit_info = journal.fit(pipe, fit.features.iloc[training], fit.labels.iloc[training],
                                       "cv", {"model": name, "candidate": index, "fold": fold,
                                              "parameters": params})
                score = float(scorer(pipe, fit.features.iloc[held], fit.labels.iloc[held]))
                fold_report = {"fold": fold, "f1": score, "fit": fit_info,
                               "fit_rows": len(training), "held_out_rows": len(held)}
                journal.record(f"cv-{journal.counts['cv']:02d}.score.json", fold_report)
                fold_reports.append(fold_report)
            fold_scores = [entry["f1"] for entry in fold_reports]
            scores.append({"candidate": index, "parameters": params, "folds": fold_reports,
                           "mean_f1": mean(fold_scores), "std_f1": pstdev(fold_scores)})
        winner = max(scores, key=lambda item: item["mean_f1"])
        results[name] = {"candidates": scores, "best_candidate": winner["candidate"],
                         "best_parameters": winner["parameters"], "best_mean_f1": winner["mean_f1"]}
    return results


def evaluate_models(templates, fit, validation, folds, journal, persist):
    from threadpoolctl import threadpool_info, threadpool_limits

    with threadpool_limits(limits=THREADS):
        pools = threadpool_info()
        if any(pool["num_threads"] > THREADS for pool in pools):
            raise RuntimeError("Detected a native thread pool above the Stage B cap.")
        journal.record("native_threadpools.json", pools)
        results = search(templates, fit, folds, journal, refit=False)
        if journal.counts != {"cv": CV_LIMIT, "refit": 0}:
            raise RuntimeError("Stage B must complete all 50 CV fits before final refits.")
        journal.record("cv_results.json", results)
        reports = {}
        for name in MODELS:
            best = results[name]
            pipe = candidate_pipeline(templates[name], best["best_parameters"])
            info = journal.fit(pipe, fit.features, fit.labels, "refit", {
                "model": name, "candidate": best["best_candidate"], "parameters": best["best_parameters"],
            })
            if list(pipe.classes_) != [0, 1]:
                raise ValueError("Unexpected class order for binary probabilities.")
            report = validation_report(validation.labels, pipe.predict_proba(validation.features)[:, 1])
            threshold = report["threshold_selection"]["threshold"]
            reports[name] = {
                "fit": info, "validation": report, "cv_selection": {
                    key: best[key] for key in ("best_candidate", "best_parameters", "best_mean_f1")
                },
                "model_parameters": pipe.named_steps["clf"].get_params(),
                "artifact": persist(name, pipe, threshold),
            }
            journal.record(f"{name}.validation.json", reports[name])
        return reports, selection(reports)
