"""Build a deterministic, source-only CET433 snapshot for the Colab runtime."""

import argparse
import hashlib
import io
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile, ZipInfo


MAX_SNAPSHOT_BYTES = 1024 * 1024
FILES = (
    "CET433_Project2_IDS_M01R.ipynb",
    "colab_acceptance.py",
    "colab_preflight.py",
    "colab_snapshot.py",
    "data/dataset_manifest.json",
    "data/m02.2_split_manifest.json",
    "docs/m02.2-split-protocol.md",
    "docs/m02.3-colab-qualification.md",
    "docs/m02.4-m03-execution.md",
    "environment/colab-py313-v1.in",
    "environment/local-jupyter-py313-v1.in",
    "environment_smoke.py",
    "grouped_splits.py",
    "ids_utils.py",
    "model_artifact.py",
    "m03_runner.py",
    "requirements.txt",
    "stage_a.py",
    "stage_b.py",
)


def build_snapshot(root, destination):
    """Package current reviewed bytes and return the digest to use in the notebook."""
    payload = io.BytesIO()
    with ZipFile(payload, "w") as archive:
        for name in FILES:
            with (root / name).open("rb") as source:
                member = source.read(MAX_SNAPSHOT_BYTES + 1)
            if payload.tell() + len(member) > MAX_SNAPSHOT_BYTES:
                raise ValueError("Source snapshot exceeds its 1 MiB limit.")
            archive.writestr(ZipInfo(name), member, compress_type=ZIP_STORED)
    data = payload.getvalue()
    if len(data) > MAX_SNAPSHOT_BYTES:
        raise ValueError("Source snapshot plus ZIP metadata exceeds 1 MiB.")
    with destination.open("xb") as out:
        out.write(data)
    return hashlib.sha256(data).hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(build_snapshot(Path(__file__).resolve().parent, args.destination))
