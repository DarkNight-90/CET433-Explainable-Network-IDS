"""M03.3 inference-only audit of the completed CET433 Stage A run."""

import argparse
import hashlib
from importlib.metadata import version
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
from zipfile import ZipFile


AUDIT_VERSION = "M03.3-v1"
RUN_ID = "m03-stage-a-001"
MOUNT = Path("/content/drive")
DRIVE = MOUNT / "MyDrive/CET433_Explainable_IDS"
PYTHON = Path("/content/cet433-py313-env/bin/python")
SNAPSHOT = "CET433_colab_m03.1-stage-a.zip"
SOURCE_SHA = "bb457de2de25d2319b114717cf332f221c03e1881225c060394dfcef19aab342"
BASE_REVISION = "3a70e321c0845627ee16dc0fb9c27bb08021564b"
LOCK_SHA = "6f5c33c4f2f84bdccb6c433aa11e3e73c9f6f3deccbd79339a5ba8f492414c68"
MODELS = ("logistic_regression", "random_forest", "lightgbm")
POLICY = dict(score_cutoff=0.5, recall_min_percent=95, fpr_strict_upper_percent=1,
              false_positive_cost_minutes=5, false_negative_cost_minutes=120)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(path):
    require(path.is_absolute() and path.resolve(strict=True) == path, f"Noncanonical path: {path}")
    return path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    with canonical(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def read_json(path):
    return json.loads(canonical(path).read_bytes())


def verify_environment():
    require(Path(sys.executable) == PYTHON and Path(sys.prefix) == PYTHON.parent.parent
            and sys.prefix != sys.base_prefix and platform.python_version() == "3.13.18"
            and sys.flags.ignore_environment and sys.flags.no_user_site,
            "Use the isolated 3.13.18 interpreter with -E -s.")
    require(os.environ.get("MPLBACKEND") == "Agg", "Set MPLBACKEND=Agg.")
    require(MOUNT.is_mount(), "Colab Drive must be mounted.")
    canonical(DRIVE)


def stage_source(destination):
    """Extract the independently pinned source to a fresh, private directory."""
    payload = canonical(DRIVE / "snapshots" / SNAPSHOT).read_bytes()
    require(digest(payload) == SOURCE_SHA, "Stage A source archive digest mismatch.")
    with ZipFile(io.BytesIO(payload)) as archive:
        for member in archive.infolist():
            relative = Path(member.filename)
            require(not relative.is_absolute() and ".." not in relative.parts and not member.is_dir(),
                    "Unexpected source archive path.")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as out:
                out.write(archive.read(member))


def verify_run(run, trusted_manifest_sha):
    """Verify the manifest before using any run-controlled data."""
    require(re.fullmatch(r"[0-9a-f]{64}", trusted_manifest_sha), "Invalid trusted manifest SHA-256.")
    payload = canonical(run / "run_manifest.json").read_bytes()
    require(digest(payload) == trusted_manifest_sha, "Run manifest trust-anchor mismatch.")
    manifest = json.loads(payload)
    expected = {
        "run_id": RUN_ID, "stage": "A", "status": "complete", "mode": "stage_a",
        "planned_fit_count": 3, "actual_fit_count": 3, "random_seed": 42,
        "official_test": "not opened", "snapshot_sha256": SOURCE_SHA,
        "source_base_revision": BASE_REVISION, "dependency_lock_sha256": LOCK_SHA,
        "interpreter": str(PYTHON), "sys_prefix": str(PYTHON.parent.parent),
        "python": "3.13.18", "matplotlib_backend": "Agg", "execution_enabled": True,
        "dataset_access": {"training": "verified", "official_test": "not opened"},
        "validation_policy": POLICY,
    }
    for key, value in expected.items():
        require(manifest.get(key) == value, f"Unexpected manifest field: {key}")
    require(manifest["sys_prefix"] != manifest["sys_base_prefix"]
            and bool(manifest.get("source_identity")), "Missing isolation or source provenance.")
    require(not os.path.lexists(run / "run_failure.json"), "Run contains a failure record.")
    start = read_json(run / "run_started.json")
    config = read_json(run / "configuration.json")
    require(start.get("status") == "started" and config["status"] == "verified_no_fit_preflight",
            "Unexpected start/configuration stage.")
    for key, value in config.items():
        if key != "status":
            require(manifest.get(key) == value, f"Manifest/configuration disagreement: {key}")
    for key, value in start.items():
        if key != "status":
            require(manifest.get(key) == value, f"Manifest/start disagreement: {key}")

    reports = manifest["models"]
    require(set(reports) == set(MODELS) == set(manifest["model_parameters"]),
            "Unexpected model set.")
    expected_journals = set()
    for index, name in enumerate(MODELS, start=1):
        prefix = f"{index:02d}-{name}"
        paths = {suffix: run / f"{prefix}.{suffix}.json"
                 for suffix in ("fit-started", "fit-result", "validation")}
        expected_journals.update(path.name for path in paths.values())
        report = reports[name]
        require(read_json(paths["fit-started"]) == {"model": name, "attempt": index},
                f"Invalid fit attempt: {name}")
        fit = read_json(paths["fit-result"])
        require(fit == report["fit"] and fit["status"] == "complete" and fit["failure"] is None
                and math.isfinite(fit["seconds"]) and fit["seconds"] >= 0
                and isinstance(fit["warnings"], list), f"Invalid fit result: {name}")
        require(read_json(paths["validation"]) == report, f"Validation report mismatch: {name}")
        artifact = report["artifact"]
        require(artifact["filename"] == f"{name}.joblib"
                and file_digest(run / f"{name}.joblib") == artifact["sha256"],
                f"Model artifact mismatch: {name}")
    for glob in ("*.fit-*.json", "*.validation.json"):
        require({p.name for p in run.glob(glob)} == {name for name in expected_journals
                if (".fit-" in name if glob == "*.fit-*.json" else name.endswith(".validation.json"))},
                "Unexpected journal/report files.")

    winner = max(MODELS[1:], key=lambda name: reports[name]["validation"]["at_0_5"]["f1"])
    require(manifest["selection"] == {
        "model": winner, "rule": "tree validation F1 at 0.5; Random Forest first on ties",
        **reports[winner]["validation"]["threshold_selection"],
    }, "Saved provisional selection is inconsistent.")
    return manifest


def reconstruct(source, run, manifest):
    from colab_acceptance import load_verified_split
    from grouped_splits import model_feature_columns
    from ids_utils import prepare_features

    lock = DRIVE / "environment/colab-py313-v1.lock"
    require(file_digest(lock) == LOCK_SHA, "Qualified dependency lock mismatch.")
    smoke = subprocess.run(
        [str(PYTHON), "-E", "-s", "-B", str(source / "environment_smoke.py"), "--without-shap"],
        check=True, capture_output=True, text=True,
    )
    require(json.loads(smoke.stdout)["matplotlib_backend"].lower() == "agg", "Agg smoke check failed.")
    pins = manifest["package_versions"]
    require(pins and all(version(name) == pinned for name, pinned in pins.items()),
            "Installed versions differ from Stage A.")

    dataset = source / "data/dataset_manifest.json"
    split = source / "data/m02.2_split_manifest.json"
    assignment = run / "split_assignments.json"
    for path, key in ((dataset, "dataset_manifest_sha256"), (split, "frozen_split_manifest_sha256"),
                      (assignment, "frozen_assignment_file_sha256")):
        require(file_digest(path) == manifest[key], f"Run input digest mismatch: {path.name}")
    training = canonical(DRIVE / "data/official/UNSW_NB15_training-set.csv")
    raw, plan, report = load_verified_split(source, training, assignment)
    require(report["training_sha256"] == manifest["training_dataset_sha256"]
            and report["assignment_sha256"] == manifest["frozen_assignment_codes_sha256"]
            and report["fit_validation_group_overlap"] == 0
            and report["cv_group_overlap"] == [0] * 5, "Verified split disagrees with run provenance.")
    for key in ("rows", "groups", "support", "training_sha256", "assignment_sha256",
                "fit_validation_group_overlap", "cv_group_overlap"):
        require(report[key] == manifest["split"][key], f"Run split report mismatch: {key}")

    positions = plan.validation
    val = raw.iloc[positions]
    features = prepare_features(val, model_feature_columns(read_json(dataset)))
    labels = val.label.to_numpy()
    groups = plan.groups[positions]
    require(len(raw) == 175_341 and len(plan.fit) == 140_273 and len(val) == 35_068
            and int(labels.sum()) == 23_868, "Unexpected frozen validation population.")
    identity = hashlib.sha256()
    for pos, group, label in zip(positions, groups, labels, strict=True):
        identity.update(f"{pos},{group},{label}\n".encode("ascii"))
    return features, val, groups, report, identity.hexdigest()


def operating_point(labels, scores, threshold):
    attack = scores >= threshold
    tp = int((attack & labels).sum())
    fp = int((attack & ~labels).sum())
    fn = int(labels.sum()) - tp
    tn = len(labels) - tp - fp - fn
    return dict(threshold=threshold, TP=tp, FP=fp, TN=tn, FN=fn,
                precision=tp / (tp + fp) if tp + fp else 0.0, recall=tp / (tp + fn),
                f1=2 * tp / (2 * tp + fp + fn), fpr=fp / (fp + tn),
                cost_minutes=5 * fp + 120 * fn)


def boundary_audit(labels, scores, saved):
    """Use a descending histogram independent of the Stage A prefix-search code."""
    import numpy as np

    buckets, indexes = np.unique(np.r_[scores, 0.0, 1.0], return_inverse=True)
    attacks = np.bincount(indexes[:-2][labels], minlength=len(buckets))[::-1].cumsum()
    normal = np.bincount(indexes[:-2][~labels], minlength=len(buckets))[::-1].cumsum()
    p, n = int(labels.sum()), int((~labels).sum())
    fpr_ok = normal * 100 < n
    recall_ok = attacks * 100 >= 95 * p
    feasible = fpr_ok & recall_ok
    missed = p - attacks
    states = list(zip(5 * normal + 120 * missed, normal, -buckets[::-1], strict=True))
    candidates = np.flatnonzero(feasible) if feasible.any() else range(len(states))
    optimum_key = min(states[i] for i in candidates)
    saved_key = (saved["cost_minutes"], saved["FP"], -saved["threshold"])
    return {
        "boundaries_checked": len(buckets),
        "fp_cap": (n - 1) // 100, "fn_cap": p * 5 // 100,
        "jointly_feasible": bool(feasible.any()),
        "maximum_recall_under_fpr_cap": float(attacks[fpr_ok].max() / p) if fpr_ok.any() else None,
        "minimum_fpr_at_recall_floor": float(normal[recall_ok].min() / n) if recall_ok.any() else None,
        "saved_threshold_matches_cost_and_tie_break": bool(saved_key == optimum_key),
    }


def compare_metrics(reported, observed):
    mismatches = []
    for key, value in observed.items():
        actual = reported.get(key)
        if isinstance(value, float):
            equal = type(actual) in (int, float) and math.isclose(
                actual, value, rel_tol=0, abs_tol=1e-12)
        else:
            equal = actual == value
        if not equal:
            mismatches.append(key)
    return mismatches


def error_analysis(val, groups, alerts):
    import pandas as pd

    data = pd.DataFrame({"label": val.label.to_numpy(), "family": val.attack_cat.to_numpy(),
                         "group": groups, "alert": alerts})
    data["FP"] = (data.label == 0) & data.alert
    data["FN"] = (data.label == 1) & ~data.alert
    normal = data[data.label == 0]
    fp_by_category = {}
    for column in ("proto", "service", "state"):
        counts = normal.groupby(val[column].to_numpy()[data.label == 0], dropna=False)["FP"].agg(
            ["size", "sum"])
        fp_by_category[column] = [
            {"category": None if pd.isna(category) else str(category),
             "normal_rows": int(total), "false_positives": int(fp), "fpr": float(fp / total)}
            for category, total, fp in counts.itertuples()
        ]
    attacks = data[data.label == 1]
    fn_by_family = [
        {"family": family, "attack_rows": int(total), "false_negatives": int(fn),
         "recall": float(1 - fn / total)}
        for family, total, fn in attacks.groupby("family")["FN"].agg(["size", "sum"]).itertuples()
    ]
    grouped = data.groupby("group").agg(
        rows=("label", "size"), attacks=("label", "sum"), FP=("FP", "sum"), FN=("FN", "sum"))
    mixed = (grouped.attacks > 0) & (grouped.attacks < grouped.rows)

    def summary(mask):
        selected = grouped[mask]
        return {
            "groups": len(selected), "rows": int(selected.rows.sum()),
            "attack_rows": int(selected.attacks.sum()),
            "normal_rows": int((selected.rows - selected.attacks).sum()),
            "false_positives": int(selected.FP.sum()), "false_negatives": int(selected.FN.sum()),
        }

    size = grouped.rows
    return {
        "false_positives_by_category": fp_by_category,
        "false_negatives_by_family": fn_by_family,
        "groups_by_size": {"1": summary(size == 1), "2–5": summary(size.between(2, 5)),
                           "6–20": summary(size.between(6, 20)), "21+": summary(size >= 21)},
        "mixed_binary_label_groups": summary(mixed),
        "non_mixed_binary_label_groups": summary(~mixed),
        "error_concentration": {
            field: {"affected_groups": int((grouped[field] > 0).sum()),
                    "rows_in_repeated_groups": int(grouped.loc[size > 1, field].sum()),
                    "top_10_groups_error_share":
                        float(grouped[field].nlargest(10).sum() / grouped[field].sum())
                        if grouped[field].sum() else None}
            for field in ("FP", "FN")
        },
    }


def preprocessing_analysis(preprocessor, features, groups):
    import numpy as np

    encoder = preprocessor.named_transformers_["cat"]
    coverage = {}
    for column, known, rare in zip(("proto", "service", "state"), encoder.categories_,
                                   encoder.infrequent_categories_, strict=True):
        values = features[column]
        unfamiliar = ~values.isin(known)
        infrequent = values.isin([] if rare is None else rare)
        coverage[column] = {
            "missing_rows": int(values.isna().sum()),
            "unknown_rows": int(unfamiliar.sum()), "known_infrequent_rows": int(infrequent.sum()),
            "learned_categories": len(known), "learned_infrequent_categories": 0 if rare is None else len(rare),
            "unknown_encoding": "infrequent_bucket" if
            encoder.handle_unknown == "infrequent_if_exist" and rare is not None else "all_zero"
            if encoder.handle_unknown in ("ignore", "infrequent_if_exist") else encoder.handle_unknown,
        }

    # One representative per exact raw group. Compare full transformed byte vectors,
    # normalizing signed zeros and NaNs. Report counts without row or group identifiers.
    _, indexes = np.unique(groups, return_index=True)
    transformed = np.asarray(preprocessor.transform(features.iloc[indexes]), dtype=np.float64)
    transformed[transformed == 0] = 0
    transformed[np.isnan(transformed)] = np.nan
    signatures = {}
    for row in transformed:
        key = row.tobytes()
        signatures[key] = signatures.get(key, 0) + 1
    collisions = [count for count in signatures.values() if count > 1]
    return {
        "encoder_handle_unknown": encoder.handle_unknown, "encoder_coverage": coverage,
        "transformed_exact_collisions": {
            "raw_groups_examined": len(indexes), "transformed_vectors": len(signatures),
            "vectors_combining_distinct_raw_groups": len(collisions),
            "raw_groups_in_collisions": sum(collisions), "largest_collision": max(collisions, default=0),
        },
    }


def load_verified_model(run, name, record):
    """Hash the exact bytes passed to pickle after the manifest trust check."""
    import joblib

    blob = canonical(run / f"{name}.joblib").read_bytes()
    require(digest(blob) == record["sha256"], f"Model changed before deserialization: {name}")
    return joblib.load(io.BytesIO(blob))


def audit_model(artifact, name, features, val, groups, saved_report, manifest):
    import numpy as np
    from sklearn.metrics import roc_auc_score

    from model_artifact import ARTIFACT_VERSION, MODEL_PACKAGES

    saved = saved_report["validation"]
    require(artifact["model_name"] == name and artifact["feature_cols"] == list(features.columns)
            and artifact["threshold"] == saved["threshold_selection"]["threshold"],
            f"Artifact configuration mismatch: {name}")
    meta = artifact["metadata"]
    require(meta["artifact_version"] == ARTIFACT_VERSION and meta["python"] == manifest["python"]
            and meta["packages"] == {package: manifest["package_versions"][package]
                                    for package in MODEL_PACKAGES}, f"Artifact environment mismatch: {name}")
    pipeline = artifact["pipeline"]
    require(list(pipeline.feature_names_in_) == list(features.columns)
            and list(pipeline.classes_) == [0, 1]
            and tuple(pipeline.named_steps) == ("pre", "clf")
            and pipeline.named_steps["clf"].__class__.__name__ == {
                "logistic_regression": "LogisticRegression", "random_forest": "RandomForestClassifier",
                "lightgbm": "LGBMClassifier",
            }[name]
            and pipeline.named_steps["clf"].get_params() == manifest["model_parameters"][name],
            f"Unexpected fitted pipeline: {name}")

    labels = val.label.to_numpy() == 1
    probas = np.asarray(pipeline.predict_proba(features))
    require(probas.shape == (len(labels), 2) and np.isfinite(probas).all()
            and ((probas >= 0) & (probas <= 1)).all(), f"Invalid probabilities: {name}")
    scores = probas[:, 1]
    threshold = artifact["threshold"]
    require(type(threshold) in (int, float) and 0 <= threshold <= 1, "Invalid saved threshold.")
    at_default = operating_point(labels, scores, 0.5)
    at_saved = operating_point(labels, scores, threshold)
    boundaries = boundary_audit(labels, scores, at_saved)
    require((boundaries["fp_cap"], boundaries["fn_cap"]) == (111, 1193),
            "Unexpected validation gate caps.")
    positives = int(labels.sum())
    summary = {"support": {"rows": len(labels), "normal": len(labels) - positives, "attack": positives},
               "attack_prevalence": positives / len(labels),
               "roc_auc": float(roc_auc_score(labels, scores))}
    feasible = boundaries["jointly_feasible"]
    saved_status = "FEASIBLE" if feasible else "INFEASIBLE"
    purpose = "feasible_operating_point" if feasible else "descriptive_benchmark"
    mismatches = compare_metrics(saved, summary)
    mismatches.extend("at_0_5." + key for key in compare_metrics(saved["at_0_5"], at_default))
    mismatches.extend("threshold_selection." + key for key in compare_metrics(
        saved["threshold_selection"], {**at_saved, "status": saved_status, "purpose": purpose}))
    if not boundaries["saved_threshold_matches_cost_and_tie_break"]:
        mismatches.append("threshold_selection.cost_or_tie_break")
    return {
        **summary, "verified_fit_result": saved_report["fit"],
        "artifact_sha256": saved_report["artifact"]["sha256"],
        "at_0_5": at_default, "at_saved_threshold": at_saved,
        "threshold_feasibility": {**boundaries, "recorded_status": saved["threshold_selection"]["status"]},
        "record_mismatches": mismatches,
        "errors_at_0_5": error_analysis(val, groups, scores >= 0.5),
        "errors_at_saved_threshold": error_analysis(val, groups, scores >= threshold),
        "preprocessing": preprocessing_analysis(pipeline.named_steps["pre"], features, groups),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trusted-run-manifest-sha256", required=True,
                        help="Digest authenticated separately from the Drive run directory")
    parser.add_argument("--audit-id", required=True, help="Unused directory under the project audits folder")
    args = parser.parse_args(argv)
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.audit_id), "Invalid audit ID.")
    verify_environment()
    run = canonical(DRIVE / "runs" / RUN_ID)
    output_root = DRIVE / "audits"
    out = output_root / args.audit_id
    require(not os.path.lexists(out), "Audit directory already exists.")
    require(not os.path.lexists(output_root) or canonical(output_root).is_dir(), "Invalid audit root.")
    manifest = verify_run(run, args.trusted_run_manifest_sha256)

    with tempfile.TemporaryDirectory(prefix="cet433-audit-source-") as folder:
        source = Path(folder)
        stage_source(source)
        sys.path.insert(0, str(source))
        try:
            features, val, groups, split, identity = reconstruct(source, run, manifest)
            output_root.mkdir(exist_ok=True)
            canonical(output_root)
            out.mkdir(exist_ok=False)
            reports = {}
            for name in MODELS:
                record = manifest["models"][name]
                artifact = load_verified_model(run, name, record["artifact"])
                reports[name] = audit_model(artifact, name, features, val, groups, record, manifest)
                del artifact
        finally:
            sys.path.remove(str(source))

    matches = not any(report["record_mismatches"] for report in reports.values())
    report = {
        "schema_version": 1, "audit_version": AUDIT_VERSION, "audit_id": args.audit_id,
        "audit_code_sha256": file_digest(Path(__file__).absolute()),
        "run_id": RUN_ID, "authenticated_run_manifest_sha256": args.trusted_run_manifest_sha256,
        "stage_a_snapshot_sha256": SOURCE_SHA, "source_base_revision": BASE_REVISION,
        "stage_a_source_identity": manifest["source_identity"],
        "dependency_lock_sha256": LOCK_SHA, "fits_recorded": manifest["actual_fit_count"],
        "official_test": "not opened", "recorded_stage_a_official_test": manifest["official_test"],
        "validation": {
            **split["support"]["validation"], "training_sha256": split["training_sha256"],
            "assignment_codes_sha256": split["assignment_sha256"],
            "assignment_file_sha256": manifest["frozen_assignment_file_sha256"],
            "split_manifest_sha256": manifest["frozen_split_manifest_sha256"],
            "dataset_manifest_sha256": manifest["dataset_manifest_sha256"],
            "ordered_position_group_label_sha256": identity,
        },
        "record_matches": matches,
        "any_existing_model_jointly_feasible": any(model["threshold_feasibility"]["jointly_feasible"]
                                                   for model in reports.values()),
        "models": reports,
    }
    dest = out / "inference_audit.json"
    with dest.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    print(dest)
    return 0 if matches else 1


if __name__ == "__main__":
    raise SystemExit(main())
