"""Read-only verification of the Colab snapshot and frozen grouped split."""

import ast
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

from colab_preflight import verify_partition
from colab_snapshot import FILES, MAX_SNAPSHOT_BYTES


def verify_snapshot(root, snapshot, expected_sha256):
    root, snapshot = Path(root), Path(snapshot)
    if snapshot.stat().st_size > MAX_SNAPSHOT_BYTES:
        raise ValueError("Source snapshot exceeds its size limit.")
    with snapshot.open("rb") as source:
        observed = hashlib.file_digest(source, "sha256").hexdigest()
        if observed != expected_sha256.strip().lower():
            raise ValueError("Source snapshot SHA-256 mismatch.")
        source.seek(0)
        with ZipFile(source) as archive:
            if tuple(archive.namelist()) != FILES:
                raise ValueError("Unexpected source snapshot contents or order.")
            for name in FILES:
                if archive.read(name) != (root / name).read_bytes():
                    raise ValueError(f"Runtime source differs from the verified snapshot: {name}")
    return observed


def verify_model_guard(notebook_path, *, model_execution):
    cells = json.loads(Path(notebook_path).read_text(encoding="utf-8"))["cells"]
    guards = [
        node.value
        for cell in cells if cell["cell_type"] == "code"
        for node in ast.parse("".join(cell["source"])).body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "ALLOW_MODEL_EXECUTION"
                for target in node.targets)
    ]
    if (model_execution is not False or len(guards) != 1
            or not isinstance(guards[0], ast.Constant) or guards[0].value is not False):
        raise ValueError("Model execution must remain disabled for this check.")


def load_assignment(path, manifest, expected_rows):
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest["assignment_file_sha256"]:
        raise ValueError("Assignment file SHA-256 mismatch.")
    record = json.loads(payload)
    codes = record.pop("assignments")
    if record != {key: value for key, value in manifest.items() if key != "assignment_file_sha256"}:
        raise ValueError("Assignment metadata differs from the split manifest.")
    if len(codes) != expected_rows or set(codes) != set("V01234"):
        raise ValueError("Expected one valid assignment per training row and all six split codes.")
    if hashlib.sha256(codes.encode("ascii")).hexdigest() != manifest["assignment_sha256"]:
        raise ValueError("Assignment code digest mismatch.")
    return codes


def load_verified_split(root, training_path, assignment_path):
    """Check frozen membership against exact features, without regenerating or fitting."""
    from grouped_splits import (
        POLICY, SplitPlan, feature_groups, model_feature_columns, split_record,
    )
    from ids_utils import read_flow_csv

    root, training_path = Path(root), Path(training_path)
    dataset = json.loads((root / "data/dataset_manifest.json").read_text(encoding="utf-8"))
    frozen = json.loads((root / "data/m02.2_split_manifest.json").read_text(encoding="utf-8"))
    source = dataset["files"]["training"]
    if training_path.name != Path(source["path"]).name or frozen["policy"] != POLICY:
        raise ValueError("Expected the official training filename and grouped split policy.")
    verified = verify_partition(training_path, dataset["schema"]["ordered_columns"], source)
    if verified["sha256"] != frozen["training_sha256"]:
        raise ValueError("Split row order is not bound to this training file.")
    codes = load_assignment(assignment_path, frozen, verified["rows"])
    raw = read_flow_csv(training_path)
    columns = model_feature_columns(dataset)
    groups = feature_groups(raw[columns], columns)
    total_groups = frozen["fit"]["groups"] + frozen["validation"]["groups"]
    if len(raw) != len(codes) or len(set(groups)) != total_groups:
        raise ValueError("Training rows or exact feature groups differ from the frozen split.")
    seen = {}
    for group, code in zip(groups, codes, strict=True):
        if seen.setdefault(group, code) != code:
            raise ValueError("A feature group crosses validation or CV boundaries.")

    plan = SplitPlan(groups, codes)
    observed = split_record(plan, raw, verified["sha256"], seed=frozen["seed"])
    # JSON normalizes integer class keys in the same form as the stored manifest.
    observed = json.loads(json.dumps(observed, allow_nan=False))
    for key in ("fit", "validation", "cv"):
        if observed[key] != frozen[key]:
            raise ValueError(f"Split support differs from the frozen manifest: {key}")
    report = {
        "training_sha256": verified["sha256"],
        "rows": len(raw),
        "groups": total_groups,
        "assignment_sha256": plan.assignment_sha256,
        "fit_validation_group_overlap": 0,
        "cv_group_overlap": [0] * len(observed["cv"]),
        "assignment_generation_versions": frozen["versions"],
        "acceptance_versions": observed["versions"],
        "support": {key: observed[key] for key in ("fit", "validation", "cv")},
    }

    return raw, plan, report


def verify_grouped_assignment(root, training_path, assignment_path):
    return load_verified_split(root, training_path, assignment_path)[2]
