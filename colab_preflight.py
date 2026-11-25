"""Read-only runtime inventory and UNSW-NB15 partition checks."""

import csv
import hashlib
from importlib.metadata import PackageNotFoundError, version
import io
import json
import platform
from pathlib import Path
import sys


DRIVE_PROJECT = Path("/content/drive/MyDrive/CET433_Explainable_IDS")
PARTITIONS = ("training", "testing")
FAMILIES = {
    "Normal", "Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers",
    "Generic", "Reconnaissance", "Shellcode", "Worms",
}


def dataset_directory(project_root, *, cloud=False, override=None):
    """Use one explicit directory for both files."""
    if override:
        path = Path(override).expanduser()
        if not path.is_absolute():
            raise ValueError("CET433_DATA_DIR must be an absolute path.")
        return path
    return (DRIVE_PROJECT if cloud else Path(project_root)) / "data" / "official"


def runtime_inventory(requirements):
    """Compare installed distribution metadata against the direct project pins."""
    packages = {}
    for line in Path(requirements).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, expected = line.split("==", 1)
        try:
            observed = version(name)
        except PackageNotFoundError:
            observed = None
        packages[name] = {"pinned": expected, "installed": observed, "matches": observed == expected}
    return {
        "python": platform.python_version(),
        "python_3_12": sys.version_info[:2] == (3, 12),
        "packages": packages,
    }


def verify_partition(path, columns, reference):
    """Hash and validate a single partition against the recovered dataset manifest."""
    expected = reference["validation"]
    with Path(path).open("rb") as raw:
        if raw.seek(0, 2) != reference["size_bytes"]:
            raise ValueError(f"{path}: byte count differs from the manifest.")
        raw.seek(0)
        digest = hashlib.file_digest(raw, "sha256").hexdigest()
        if digest != reference["sha256"]:
            raise ValueError(f"{path}: SHA-256 differs from the manifest.")
        raw.seek(0)
        with io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as text:
            source = csv.reader(text, strict=True)
            if next(source, None) != columns or len(columns) != expected["columns"]:
                raise ValueError(f"{path}: ordered CSV schema differs from the manifest.")
            label_col, category_col = columns.index("label"), columns.index("attack_cat")
            binary = dict.fromkeys(("0", "1"), 0)
            categories = dict.fromkeys(FAMILIES, 0)
            rows = 0
            for rows, row in enumerate(source, 1):
                if len(row) != len(columns):
                    raise ValueError(f"{path}: invalid field count at data row {rows}.")
                label, category = row[label_col], row[category_col]
                if label not in binary or category not in categories or (label == "0") != (category == "Normal"):
                    raise ValueError(f"{path}: invalid or inconsistent targets at data row {rows}.")
                binary[label] += 1
                categories[category] += 1
    categories = {key: count for key, count in sorted(categories.items()) if count}
    if rows != expected["rows"] or binary != expected["binary_labels"] or categories != expected["attack_categories"]:
        raise ValueError(f"{path}: dimensions or label distributions differ from the manifest.")
    return {"path": str(path), "sha256": digest, "rows": rows, "columns": len(columns),
            "binary_labels": binary, "attack_categories": categories}


def verify_datasets(dataset_dir, manifest_path):
    """Validate the exact paired filenames without recursively searching other locations."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    columns = manifest["schema"]["ordered_columns"]
    return {
        split: verify_partition(
            Path(dataset_dir) / f"UNSW_NB15_{split}-set.csv",
            columns, manifest["files"][split],
        )
        for split in PARTITIONS
    }
