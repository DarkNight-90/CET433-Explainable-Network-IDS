import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

from ids_utils import clean, log_clip, prepare_features, read_flow_csv
from model_artifact import ARTIFACT_VERSION, MODEL_PACKAGES, MODEL_PATH, load_model_artifact


ROOT = Path(__file__).resolve().parents[1]


class FlowSchemaTests(unittest.TestCase):
    def setUp(self):
        self.flows = pd.DataFrame({
            "id": [100, 101], "dur": [0.5, np.inf], "proto": ["tcp", None],
            "service": ["-", "http"], "state": ["FIN", "INT"],
            "label": [0, 1], "attack_cat": [" Normal ", "Generic"],
        })

    def test_clean_and_prepare_keep_order_and_missing_features(self):
        cleaned = clean(self.flows)
        self.assertEqual(cleaned["attack_cat"].tolist(), ["Normal", "Generic"])
        features = prepare_features(self.flows, ["proto", "dur"])
        self.assertEqual(features.columns.tolist(), ["proto", "dur"])
        self.assertTrue(features.iloc[1].isna().all())
        self.assertTrue(np.isinf(self.flows.loc[1, "dur"]))

    def test_invalid_labels(self):
        for name, value in [("label", 2), ("label", None), ("attack_cat", None),
                            ("attack_cat", " "), ("attack_cat", "Generic")]:
            with self.subTest(name=name, value=value):
                bad = self.flows.copy()
                bad.loc[0, name] = value
                with self.assertRaises(ValueError):
                    clean(bad)

    def test_missing_training_columns(self):
        for name in ("label", "attack_cat", "proto"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Missing required training columns"):
                clean(self.flows.drop(columns=name))

    def test_invalid_input_schema(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            prepare_features(self.flows.iloc[:0], ["dur"])
        with self.assertRaisesRegex(ValueError, "Missing feature columns"):
            prepare_features(self.flows, ["rate"])
        with self.assertRaisesRegex(ValueError, "unique list"):
            prepare_features(self.flows, ["label"])
        with self.assertRaisesRegex(ValueError, "numbers"):
            prepare_features(self.flows.assign(dur=["invalid", "1"]), ["dur"])
        with self.assertRaisesRegex(ValueError, "text"):
            prepare_features(self.flows.assign(proto=[1, 2]), ["proto"])

    def test_csv_header_and_empty_inputs(self):
        for content in (b"", b"dur,proto\n", b"dur,dur\n1,2\n", b"dur,\n1,2\n", b"dur,proto\n1,tcp,extra\n"):
            with self.subTest(content=content), self.assertRaises(ValueError):
                read_flow_csv(io.BytesIO(content))
        self.assertEqual(read_flow_csv(io.BytesIO(b"dur,proto\n1,tcp\n")).iloc[0]["proto"], "tcp")


class ImportAndArtifactTests(unittest.TestCase):
    def test_clean_import_from_other_directory_without_writes(self):
        with tempfile.TemporaryDirectory() as work:
            subprocess.run(
                [sys.executable, "-B", "-c",
                 "from ids_utils import clean, log_clip, prepare_features; "
                 "from temporal_split_helper import chronological_split; "
                 "from model_artifact import MODEL_PATH; "
                 "import sys; from pathlib import Path; "
                 "assert MODEL_PATH == Path(sys.argv[1]) / 'models' / 'ids_model.joblib'",
                 str(ROOT)],
                cwd=work, env={**os.environ, "PYTHONPATH": str(ROOT)}, check=True,
            )
            self.assertEqual(list(Path(work).iterdir()), [])

    def test_serialized_transformer_preserves_legacy_symbol(self):
        stream = io.BytesIO()
        joblib.dump(FunctionTransformer(log_clip), stream)
        stream.seek(0)
        transformer = joblib.load(stream)
        self.assertIs(transformer.func, log_clip)
        np.testing.assert_allclose(transformer.transform(np.array([-1, 1])), [0, np.log(2)])

    def test_missing_artifact_reports_resolved_location(self):
        with tempfile.TemporaryDirectory() as work:
            absent = Path(work) / MODEL_PATH.name
            with self.assertRaises(FileNotFoundError) as failure:
                load_model_artifact(absent)
            self.assertIn(str(absent), str(failure.exception))

    def test_metadata_required_before_use(self):
        with patch("model_artifact.joblib.load", return_value={"pipeline": None}):
            with self.assertRaisesRegex(ValueError, "metadata"):
                load_model_artifact(ROOT / "requirements.txt")
        invalid = {"metadata": {"artifact_version": ARTIFACT_VERSION, "python": "3.12", "packages": {}}}
        with patch("model_artifact.joblib.load", return_value=invalid):
            with self.assertRaisesRegex(ValueError, "package versions"):
                load_model_artifact(ROOT / "requirements.txt")

    def test_artifact_schema_matches_pipeline_contract(self):
        pipeline = SimpleNamespace(
            predict_proba=lambda x: None,
            named_steps={"pre": object(), "clf": object()},
            classes_=[0, 1],
            feature_names_in_=["dur"],
        )
        artifact = {
            "metadata": {"artifact_version": ARTIFACT_VERSION, "python": "3.12",
                         "packages": dict.fromkeys(MODEL_PACKAGES, "test-version")},
            "feature_cols": ["dur"],
            "threshold": 0.5,
            "model_name": "Test contract",
            "pipeline": pipeline,
        }
        with patch("model_artifact.joblib.load", return_value=artifact):
            self.assertIs(load_model_artifact(ROOT / "requirements.txt"), artifact)
            artifact["threshold"] = float("nan")
            with self.assertRaisesRegex(ValueError, "threshold"):
                load_model_artifact(ROOT / "requirements.txt")
            artifact["threshold"] = 0.5
            artifact["feature_cols"] = ["rate"]
            with self.assertRaisesRegex(ValueError, "matching its feature schema"):
                load_model_artifact(ROOT / "requirements.txt")

    def test_notebook_entry_has_no_bootstrap_or_runtime_install(self):
        notebook = json.loads((ROOT / "CET433_Project2_IDS_M01R.ipynb").read_text(encoding="utf-8"))
        source = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
        self.assertNotIn("UTILS_SOURCE", source)
        self.assertNotIn("pip install", source)
        self.assertIn("from ids_utils import", source)


if __name__ == "__main__":
    unittest.main()
