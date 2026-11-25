"""Verified Colab launch route. Default action verifies inputs without fitting."""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

from colab_acceptance import load_verified_split, verify_model_guard, verify_snapshot
from colab_preflight import runtime_inventory, verify_partition


ROOT = Path(__file__).resolve().parent
NOTEBOOK = "CET433_Project2_IDS_M01R.ipynb"
DRIVE_MOUNT = Path("/content/drive")
DRIVE = DRIVE_MOUNT / "MyDrive/CET433_Explainable_IDS"
PYTHON = Path("/content/cet433-py313-env/bin/python")
LOCK_SHA256 = "6f5c33c4f2f84bdccb6c433aa11e3e73c9f6f3deccbd79339a5ba8f492414c68"
SELECTION = (
    "features", "frozen_split", "preprocessing", "sanity", "binary_baselines",
    "tuning", "comparison", "select_model", "threshold", "validation", "multiclass",
)
HOLDOUT = "official_holdout"
STAGE_A_SNAPSHOT = "CET433_colab_m03.1-stage-a.zip"
STAGE_B_SNAPSHOT = "CET433_colab_m03.4-stage-b.zip"


def verify_environment():
    if (Path(sys.executable) != PYTHON or Path(sys.prefix) != PYTHON.parent.parent
            or sys.prefix == sys.base_prefix or platform.python_version() != "3.13.18"):
        raise RuntimeError(f"Use the qualified isolated interpreter at {PYTHON}.")
    if os.environ.get("MPLBACKEND") != "Agg":
        raise RuntimeError("Set MPLBACKEND=Agg before launching the isolated interpreter.")


def verify_lock(path):
    with Path(path).open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if digest != LOCK_SHA256:
        raise ValueError("Dependency lock SHA-256 differs from the qualified lock.")
    return digest


def reviewed_cells(path):
    """Compile only explicitly selected M03 cells. No execution during preflight."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = (*SELECTION, HOLDOUT)
    marked = [cell for cell in document["cells"] if cell["metadata"].get("cet433_m03_role")]
    if tuple(cell["metadata"]["cet433_m03_role"] for cell in marked) != expected:
        raise ValueError("Unexpected set or order of reviewed M03 cells.")
    compiled = {}
    for role, cell in zip(expected, marked, strict=True):
        if cell["cell_type"] != "code":
            raise ValueError(f"{role} needs to be a code cell.")
        tree = ast.parse("".join(cell["source"]))
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        if role in SELECTION and names & {"TEST_PATH", "df_test", "X_test", "yb_test", "ym_test", "test_report"}:
            raise ValueError(f"Official test state is forbidden during selection ({role}).")
        if names & {"build_split_plan", "train_test_split", "StratifiedKFold", "MLPClassifier", "clean"}:
            raise ValueError(f"Legacy or non-group-aware code in reviewed cell {role}.")
        compiled[role] = compile(tree, f"{path}:{role}", "exec")
    return compiled


def verify_stage_a_paths(args):
    verify_controlled_paths(args, STAGE_A_SNAPSHOT)


def verify_controlled_paths(args, snapshot_name):
    if not DRIVE_MOUNT.is_mount() or DRIVE.resolve(strict=True) != DRIVE:
        raise ValueError("The controlled runner requires the mounted, canonical project Drive directory.")
    expected = (
        (args.snapshot, DRIVE / "snapshots" / snapshot_name),
        (args.lock, DRIVE / "environment/colab-py313-v1.lock"),
        (args.assignment, DRIVE / "splits/m02.2-split-assignments.json"),
    )
    for supplied, required in expected:
        if supplied != required or required.resolve(strict=True) != required:
            raise ValueError(f"The controlled runner requires the canonical input path {required}.")
    training = DRIVE / "data/official/UNSW_NB15_training-set.csv"
    if training.resolve(strict=True) != training:
        raise ValueError("The controlled runner requires the canonical official training input.")


def preflight(args):
    verify_environment()
    stage_a_enabled = getattr(args, "execute_stage_a", False)
    stage_a_snapshot = Path(args.snapshot).name == STAGE_A_SNAPSHOT
    stage_b_enabled = getattr(args, "execute_stage_b", False)
    stage_b_snapshot = Path(args.snapshot).name == STAGE_B_SNAPSHOT
    controlled = stage_a_enabled or stage_a_snapshot or stage_b_enabled or stage_b_snapshot
    if stage_b_enabled or stage_b_snapshot:
        import stage_b

        verify_controlled_paths(args, STAGE_B_SNAPSHOT)
        stage_b.verify_thread_environment()
    elif stage_a_enabled or stage_a_snapshot:
        verify_stage_a_paths(args)
    digest = verify_snapshot(ROOT, args.snapshot, args.snapshot_sha256)
    lock = verify_lock(args.lock)
    verify_model_guard(ROOT / NOTEBOOK, model_execution=False)
    # Each controlled stage has a dedicated Python path.
    cells = {} if controlled else reviewed_cells(ROOT / NOTEBOOK)
    smoke_command = [sys.executable, "-E", "-s", "-B", str(ROOT / "environment_smoke.py")]
    if controlled:
        smoke_command.append("--without-shap")
    result = subprocess.run(smoke_command, check=True, text=True, stdout=subprocess.PIPE)
    smoke = json.loads(result.stdout)
    if (smoke.get("matplotlib_backend") or "").lower() != "agg":
        raise RuntimeError("Strict smoke test did not observe Agg.")
    raw, plan, support = load_verified_split(
        ROOT, DRIVE / "data/official/UNSW_NB15_training-set.csv", args.assignment,
    )
    metadata = {
        "source_base_revision": args.source_revision,
        "source_identity": "Verified snapshot; Git base revision does not represent working-tree changes.",
        "snapshot_sha256": digest,
        "dependency_lock_sha256": lock,
        "interpreter": sys.executable,
        "sys_prefix": sys.prefix,
        "sys_base_prefix": sys.base_prefix,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "libc": platform.libc_ver(),
        "matplotlib_backend": "Agg",
        "environment_smoke": smoke,
        "dataset_access": {"training": "verified", "official_test": "not opened"},
        "frozen_assignment_file_sha256": hash_file(args.assignment),
        "split": support,
        "execution_enabled": args.execute_m03 or stage_a_enabled or stage_b_enabled,
        "mode": ("stage_b" if stage_b_enabled else "stage_a" if stage_a_enabled else
                 "m03" if args.execute_m03 else "preflight"),
        "status": "verified_no_fit_preflight",
    }
    if stage_b_enabled or stage_b_snapshot:
        from stage_a import FIT_ROWS, TOTAL_ROWS, VALIDATION_ROWS

        if (len(raw) != TOTAL_ROWS or len(plan.fit) != FIT_ROWS
                or len(plan.validation) != VALIDATION_ROWS):
            raise ValueError("Stage B requires the complete frozen M02.2 population.")
        folds = stage_b.frozen_folds(plan, raw.label.iloc[plan.fit])
        metadata["stage_b_plan"] = {
            **stage_b.configuration(),
            "verified_fold_sizes": [{"fold": i, "fit": len(a), "held_out": len(b)}
                                    for i, (a, b) in enumerate(folds)],
            "native_thread_environment": {name: os.environ[name] for name in stage_b.THREAD_VARIABLES},
        }
    return raw, plan, metadata, cells


def hash_file(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def model_namespace(raw, plan, seed):
    """Import the experiment stack only after M03 authorization and preflight."""
    from importlib.metadata import version
    import time

    import joblib
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from lightgbm import LGBMClassifier
    from sklearn.base import clone
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

    from grouped_splits import feature_groups, model_feature_columns
    from ids_utils import log_clip, prepare_features, read_flow_csv
    from sklearn.tree import DecisionTreeClassifier

    if plt.get_backend().lower() != "agg":
        raise RuntimeError("Matplotlib did not activate Agg.")
    np.random.seed(seed)
    sns.set_theme(style="whitegrid")
    namespace = locals().copy()
    namespace.update(CET433_CONTROLLED_RUN=True, PROJECT_DIR=ROOT, SEED=seed, df_raw=raw,
                     split_plan=plan, json=json,
                     display=print)
    return namespace


def write_json(path, value):
    def convert_scalar(item):
        if type(item).__module__.startswith("numpy") and hasattr(item, "item"):
            return item.item()
        raise TypeError(f"Unsupported metadata type {type(item).__name__}.")

    payload = json.dumps(value, indent=2, allow_nan=False, default=convert_scalar)
    with path.open("x", encoding="utf-8") as out:
        out.write(payload + "\n")


def select_models(cells, namespace):
    for role in SELECTION:
        exec(cells[role], namespace)
    return {
        "binary_model": namespace["final_name"],
        "binary_model_parameters": namespace["final_pipe"]["clf"].get_params(),
        "multiclass_model_parameters": namespace["multi_lgbm"]["clf"].get_params(),
        "search_configuration": {
            name: {"estimator": clf.get_params(), "space": grid}
            for name, (clf, grid) in namespace["search_space"].items()
        },
        "search": {"iterations": 6, "scoring": "f1", "folds": 5},
        "selected_parameters": namespace["best_params"],
        "seed": namespace["SEED"],
        "fit_and_validation_weighting": "raw_original_observations",
        "threshold": namespace["THRESHOLD"],
        "validation_joint_feasible": namespace["joint_feasible"],
        "threshold_rule": {"recall_min": 0.95, "fpr_strict_upper": 0.01,
                           "false_alarm_cost_minutes": namespace["COST_FP_MIN"],
                           "missed_attack_cost_minutes": namespace["COST_FN_MIN"]},
        "validation_metrics": namespace["final"],
    }


def open_official_holdout(namespace, cells):
    manifest = namespace["dataset_manifest"]
    path = DRIVE / "data/official" / Path(manifest["files"]["testing"]["path"]).name
    report = verify_partition(path, manifest["schema"]["ordered_columns"], manifest["files"]["testing"])
    namespace["TEST_PATH"] = path
    exec(cells[HOLDOUT], namespace)
    return report["sha256"]


def execute(args, raw, plan, metadata, cells):
    from model_artifact import ARTIFACT_VERSION, MODEL_PACKAGES

    run_root = DRIVE / "runs"
    run_root.mkdir(exist_ok=True)
    if run_root.resolve() != run_root:
        raise ValueError("Run output root must be the persistent project Drive directory.")
    run = run_root / args.run_id
    run.mkdir(exist_ok=False)
    paths = {name: run / filename for name, filename in {
        "start": "run_started.json", "assignments": "split_assignments.json",
        "selection": "selection.json", "binary": "ids_model.joblib",
        "multiclass": "attack_family_model.joblib", "results": "results.json",
        "manifest": "run_manifest.json",
    }.items()}
    metadata["artifacts"] = {key: str(value) for key, value in paths.items()}
    write_json(paths["start"], {**metadata, "status": "started"})
    with Path(args.assignment).open("rb") as source, paths["assignments"].open("xb") as out:
        shutil.copyfileobj(source, out)
    if hash_file(paths["assignments"]) != metadata["frozen_assignment_file_sha256"]:
        raise ValueError("Persisted assignment changed before model selection.")

    frozen = json.loads((ROOT / "data/m02.2_split_manifest.json").read_text(encoding="utf-8"))
    namespace = model_namespace(raw, plan, frozen["seed"])
    selection = select_models(cells, namespace)
    write_json(paths["selection"], selection)
    with paths["binary"].open("xb") as out:
        namespace["joblib"].dump({
            "pipeline": namespace["final_pipe"], "threshold": selection["threshold"],
            "feature_cols": namespace["feature_columns"], "model_name": selection["binary_model"],
            "metadata": {"artifact_version": ARTIFACT_VERSION, "python": platform.python_version(),
                         "packages": {name: namespace["version"](name) for name in MODEL_PACKAGES}},
        }, out)
    with paths["multiclass"].open("xb") as out:
        namespace["joblib"].dump(namespace["multi_lgbm"], out)

    # The threshold and model artifacts have been persisted. Only now open the test CSV.
    test_sha256 = open_official_holdout(namespace, cells)
    comparison = namespace["comp"]
    write_json(paths["results"], {
        "evaluation_policy": "cet433-eval-v1",
        "official_test_weighting": "raw_original_observations",
        "deduplicated_evaluation": "not performed",
        "selection": selection,
        "comparison": comparison.astype(object).where(comparison.notna(), None).to_dict(orient="index"),
        "validation_family_recall": namespace["rec"].to_dict(),
        "estimated_analyst_hours_saved_validation": namespace["hours_saved"],
        "estimated_analyst_hours_saved_per_100k": namespace["per_100k"],
        "official_test": namespace["test_report"],
    })
    write_json(paths["manifest"], {**metadata, "status": "complete", "selection": selection,
                                  "dataset_access": {"training": "verified", "official_test": "evaluated after selection"},
                                  "dataset_hashes": {"training": metadata["split"]["training_sha256"],
                                                     "testing": test_sha256}})


def execute_stage_a(args, raw, plan, metadata):
    import joblib

    from grouped_splits import model_feature_columns
    from model_artifact import ARTIFACT_VERSION, MODEL_PACKAGES
    import stage_a

    if not args.execute_stage_a:
        raise ValueError("Stage A requires explicit authorization via --execute-stage-a.")
    if not DRIVE_MOUNT.is_mount() or DRIVE.resolve(strict=True) != DRIVE:
        raise ValueError("Stage A requires the mounted, canonical project Drive directory.")
    run_root = DRIVE / "runs"
    run_root.mkdir(exist_ok=True)
    if run_root.resolve(strict=True) != run_root:
        raise ValueError("Stage A runs must stay in the persistent project Drive directory.")
    run = run_root / args.run_id
    run.mkdir(exist_ok=False)

    def record(name, value):
        write_json(run / name, value)

    metadata = {**metadata, "run_id": args.run_id, "stage": "A",
                "official_test": "not opened", "planned_fit_count": stage_a.FIT_BUDGET,
                "random_seed": stage_a.SEED}
    record("run_started.json", {**metadata, "status": "started"})
    try:
        assignment = run / "split_assignments.json"
        with args.assignment.open("rb") as source, assignment.open("xb") as out:
            shutil.copyfileobj(source, out)
        if hash_file(assignment) != metadata["frozen_assignment_file_sha256"]:
            raise ValueError("Persisted assignment differs from the verified input.")
        split_manifest = ROOT / "data/m02.2_split_manifest.json"
        dataset_manifest = ROOT / "data/dataset_manifest.json"
        metadata.update(
            frozen_split_manifest_sha256=hash_file(split_manifest),
            dataset_manifest_sha256=hash_file(dataset_manifest),
            training_dataset_sha256=metadata["split"]["training_sha256"],
            frozen_assignment_codes_sha256=plan.assignment_sha256,
        )
        if json.loads(split_manifest.read_text(encoding="utf-8"))["seed"] != stage_a.SEED:
            raise ValueError("Unexpected frozen Stage A seed.")
        features = model_feature_columns(json.loads(dataset_manifest.read_text(encoding="utf-8")))
        fit, validation = stage_a.partitions(raw, plan, features)
        pipelines = stage_a.model_pipelines(features)
        inventory = runtime_inventory(ROOT / "environment/colab-py313-v1.in")
        versions = {name: item["installed"] for name, item in inventory["packages"].items()}
        metadata.update(
            package_versions=versions,
            model_parameters={name: pipe.named_steps["clf"].get_params()
                              for name, pipe in pipelines.items()},
            validation_policy={
                "score_cutoff": 0.5, "recall_min_percent": stage_a.RECALL_MIN_PERCENT,
                "fpr_strict_upper_percent": stage_a.FPR_STRICT_UPPER_PERCENT,
                "false_positive_cost_minutes": stage_a.FP_COST,
                "false_negative_cost_minutes": stage_a.FN_COST,
            },
        )
        record("configuration.json", metadata)

        def persist(name, pipeline, threshold):
            artifact = run / f"{name}.joblib"
            with artifact.open("xb") as out:
                joblib.dump({
                    "pipeline": pipeline, "threshold": threshold, "feature_cols": features,
                    "model_name": name, "metadata": {
                        "artifact_version": ARTIFACT_VERSION, "python": platform.python_version(),
                        "packages": {name: versions[name] for name in MODEL_PACKAGES},
                    },
                }, out)
            return {"filename": artifact.name, "sha256": hash_file(artifact)}

        reports = stage_a.run_models(pipelines, fit, validation, record, persist)
        record("run_manifest.json", {
            **metadata, "status": "complete", "actual_fit_count": len(reports),
            "models": reports, "selection": stage_a.selection(reports),
        })
    except BaseException as exc:
        record("run_failure.json", {
            "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            "actual_fit_count": len(list(run.glob("*.fit-started.json"))),
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "official_test": "not opened",
        })
        raise


def execute_stage_b(args, raw, plan, metadata):
    import joblib

    from grouped_splits import model_feature_columns
    from model_artifact import ARTIFACT_VERSION, MODEL_PACKAGES
    import stage_a
    import stage_b

    if not args.execute_stage_b or metadata.get("mode") != "stage_b":
        raise ValueError("Stage B requires explicit authorization and verified Stage B preflight.")
    verify_controlled_paths(args, STAGE_B_SNAPSHOT)
    stage_b.verify_thread_environment()
    run_root = DRIVE / "runs"
    run_root.mkdir(exist_ok=True)
    if run_root.resolve(strict=True) != run_root:
        raise ValueError("Stage B runs must stay in the persistent project Drive directory.")
    run = run_root / args.run_id
    run.mkdir(exist_ok=False)

    def record(name, value):
        write_json(run / name, value)

    metadata = {**metadata, "run_id": args.run_id, "stage": "B", "official_test": "not opened",
                "random_seed": stage_a.SEED, "planned_fit_count": stage_b.CV_LIMIT + stage_b.REFIT_LIMIT}
    record("run_started.json", {**metadata, "status": "started"})
    journal = stage_b.FitJournal(record)
    try:
        assignment = run / "split_assignments.json"
        with args.assignment.open("rb") as source, assignment.open("xb") as out:
            shutil.copyfileobj(source, out)
        if hash_file(assignment) != metadata["frozen_assignment_file_sha256"]:
            raise ValueError("Persisted assignment differs from the verified input.")
        split_manifest = ROOT / "data/m02.2_split_manifest.json"
        dataset_manifest = ROOT / "data/dataset_manifest.json"
        if json.loads(split_manifest.read_text(encoding="utf-8"))["seed"] != stage_a.SEED:
            raise ValueError("Unexpected frozen Stage B seed.")
        features = model_feature_columns(json.loads(dataset_manifest.read_text(encoding="utf-8")))
        fit, validation = stage_a.partitions(raw, plan, features)
        folds = stage_b.frozen_folds(plan, fit.labels)
        templates = stage_b.pipelines(features, stage_a.SEED)
        inventory = runtime_inventory(ROOT / "environment/colab-py313-v1.in")
        versions = {name: item["installed"] for name, item in inventory["packages"].items()}
        metadata.update(
            frozen_split_manifest_sha256=hash_file(split_manifest),
            dataset_manifest_sha256=hash_file(dataset_manifest),
            training_dataset_sha256=metadata["split"]["training_sha256"],
            frozen_assignment_codes_sha256=plan.assignment_sha256,
            package_versions=versions,
            base_model_parameters={name: pipe.named_steps["clf"].get_params()
                                   for name, pipe in templates.items()},
            validation_policy={
                "score_cutoff": 0.5, "recall_min_percent": stage_a.RECALL_MIN_PERCENT,
                "fpr_strict_upper_percent": stage_a.FPR_STRICT_UPPER_PERCENT,
                "false_positive_cost_minutes": stage_a.FP_COST,
                "false_negative_cost_minutes": stage_a.FN_COST,
            },
        )
        record("configuration.json", metadata)

        def persist(name, pipeline, threshold):
            artifact = run / f"{name}.joblib"
            with artifact.open("xb") as out:
                joblib.dump({
                    "pipeline": pipeline, "threshold": threshold, "feature_cols": features,
                    "model_name": name, "metadata": {
                        "artifact_version": ARTIFACT_VERSION, "python": platform.python_version(),
                        "packages": {package: versions[package] for package in MODEL_PACKAGES},
                    },
                }, out)
            return {"filename": artifact.name, "sha256": hash_file(artifact)}

        reports, selection = stage_b.evaluate_models(templates, fit, validation, folds, journal, persist)
        if journal.counts != {"cv": stage_b.CV_LIMIT, "refit": stage_b.REFIT_LIMIT}:
            raise RuntimeError("Stage B did not complete the declared fit plan.")
        record("run_manifest.json", {
            **metadata, "status": "complete", "fit_counts": journal.counts,
            "fit_counts_by_model": journal.per_model,
            "actual_fit_count": sum(journal.counts.values()), "models": reports, "selection": selection,
            "cv_results": {"filename": "cv_results.json", "sha256": hash_file(run / "cv_results.json")},
        })
    except BaseException as exc:
        record("run_failure.json", {
            "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            "fit_counts": journal.counts, "actual_fit_count": sum(journal.counts.values()),
            "fit_counts_by_model": journal.per_model,
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "official_test": "not opened",
        })
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--snapshot-sha256", required=True)
    parser.add_argument("--source-revision", required=True, help="Git base revision of the reviewed snapshot")
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--assignment", type=Path, required=True)
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument("--execute-m03", action="store_true")
    execution.add_argument("--execute-stage-a", action="store_true")
    execution.add_argument("--execute-stage-b", action="store_true")
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    if not re.fullmatch("[0-9a-f]{40}", args.source_revision):
        parser.error("Supply the full Git source base revision.")
    enabled = args.execute_m03 or args.execute_stage_a or args.execute_stage_b
    if args.snapshot.name == STAGE_A_SNAPSHOT and (args.execute_m03 or args.execute_stage_b):
        parser.error("Use the Stage A snapshot only for training-only preflight or Stage A.")
    if args.snapshot.name == STAGE_B_SNAPSHOT and (args.execute_m03 or args.execute_stage_a):
        parser.error("Use the Stage B snapshot only for training-only preflight or Stage B.")
    if enabled and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.run_id or ""):
        parser.error("An enabled run needs a new safe --run-id.")
    if args.run_id and not enabled:
        parser.error("--run-id is only used for an explicitly enabled run.")
    raw, plan, metadata, cells = preflight(args)
    print(json.dumps(metadata, indent=2, allow_nan=False), flush=True)
    if args.execute_m03:
        execute(args, raw, plan, metadata, cells)
    elif args.execute_stage_a:
        execute_stage_a(args, raw, plan, metadata)
    elif args.execute_stage_b:
        execute_stage_b(args, raw, plan, metadata)


if __name__ == "__main__":
    main()
