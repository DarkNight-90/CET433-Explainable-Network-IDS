"""Location and validation contract for trusted, locally generated IDS artifacts."""

import math
from numbers import Real
from pathlib import Path

import joblib


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "ids_model.joblib"
ARTIFACT_VERSION = 1
MODEL_PACKAGES = ("numpy", "pandas", "scikit-learn", "lightgbm", "joblib", "shap")


def load_model_artifact(path=MODEL_PATH):
    """Load a trusted joblib artifact created by this project's notebook."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"IDS model artifact missing at {path}. Generate it with the project notebook.")
    artifact = joblib.load(path)
    if not isinstance(artifact, dict):
        raise ValueError("Expected an IDS artifact dictionary.")
    metadata = artifact.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError("Missing or unsupported IDS artifact metadata. Re-export from the current notebook.")
    versions = metadata.get("packages")
    if (
        not isinstance(metadata.get("python"), str)
        or not metadata["python"].strip()
        or not isinstance(versions, dict)
        or any(not isinstance(versions.get(name), str) or not versions[name].strip() for name in MODEL_PACKAGES)
    ):
        raise ValueError("IDS artifact metadata must record Python and model package versions.")
    columns = artifact.get("feature_cols")
    if (
        not isinstance(columns, list)
        or not columns
        or any(not isinstance(name, str) or not name.strip() for name in columns)
        or len(columns) != len(set(columns))
        or set(columns) & {"id", "label", "attack_cat"}
    ):
        raise ValueError("IDS artifact must have an ordered, unique feature schema.")
    threshold = artifact.get("threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, Real) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("IDS artifact threshold must be a finite number between zero and one.")
    name = artifact.get("model_name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("IDS artifact must identify its model.")
    pipeline = artifact.get("pipeline")
    if (
        not callable(getattr(pipeline, "predict_proba", None))
        or not {"pre", "clf"} <= set(getattr(pipeline, "named_steps", {}))
        or list(getattr(pipeline, "classes_", [])) != [0, 1]
        or list(getattr(pipeline, "feature_names_in_", [])) != columns
    ):
        raise ValueError("IDS artifact must contain a fitted binary pipeline matching its feature schema.")
    return artifact
