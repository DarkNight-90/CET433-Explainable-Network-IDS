import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import m03_runner as runner


class LaunchTests(unittest.TestCase):
    def test_default_cli_never_executes_models(self):
        with patch.object(runner, "preflight", return_value=(None, None, {}, {})) as check,                 patch.object(runner, "execute") as execute, patch("builtins.print"):
            runner.main(["--snapshot", "s.zip", "--snapshot-sha256", "a" * 64,
                         "--source-revision", "a" * 40, "--lock", "l", "--assignment", "a"])
        self.assertFalse(check.call_args.args[0].execute_m03)
        execute.assert_not_called()

    def test_isolated_interpreter_and_agg_are_required(self):
        with patch.object(runner.platform, "python_version", return_value="3.13.18"),                 patch.object(runner.sys, "executable", "/usr/bin/python3"),                 patch.dict(runner.os.environ, {"MPLBACKEND": "Agg"}):
            with self.assertRaisesRegex(RuntimeError, "isolated interpreter"):
                runner.verify_environment()
            with patch.object(runner.sys, "executable", str(runner.PYTHON)),                     patch.object(runner.sys, "prefix", str(runner.PYTHON.parent.parent)),                     patch.object(runner.sys, "base_prefix", "/usr"):
                runner.verify_environment()
                with patch.dict(runner.os.environ, {"MPLBACKEND": "inline"}):
                    with self.assertRaisesRegex(RuntimeError, "MPLBACKEND"):
                        runner.verify_environment()

    def test_reviewed_selection_rejects_official_test_state(self):
        source = runner.ROOT / runner.NOTEBOOK
        self.assertEqual(tuple(runner.reviewed_cells(source)), (*runner.SELECTION, runner.HOLDOUT))
        notebook = json.loads(source.read_text())
        selection = next(cell for cell in notebook["cells"]
                         if cell["metadata"].get("cet433_m03_role") == "threshold")
        selection["source"].append("\nread_flow_csv(TEST_PATH)\n")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mutated.ipynb"
            path.write_text(json.dumps(notebook))
            with self.assertRaisesRegex(ValueError, "forbidden during selection"):
                runner.reviewed_cells(path)

    def test_preflight_loads_only_training_and_executes_no_model_cells(self):
        args = SimpleNamespace(snapshot="s", snapshot_sha256="s", lock="l", assignment="a",
                               execute_m03=False, source_revision="r")
        smoke = SimpleNamespace(stdout='{"matplotlib_backend":"Agg"}')
        with (patch.object(runner, "verify_environment"),
              patch.object(runner, "verify_snapshot", return_value="s"),
              patch.object(runner, "verify_lock", return_value="l"),
              patch.object(runner.subprocess, "run", return_value=smoke),
              patch.object(runner, "hash_file", return_value="a"),
              patch.object(runner, "load_verified_split", return_value=(None, None, {})) as load,
              patch.object(runner, "select_models") as select,
              patch.object(runner, "open_official_holdout") as holdout):
            _, _, meta, _ = runner.preflight(args)
        self.assertEqual(load.call_args.args[1].name, "UNSW_NB15_training-set.csv")
        self.assertEqual(meta["dataset_access"]["official_test"], "not opened")
        self.assertFalse(meta["execution_enabled"])
        select.assert_not_called()
        holdout.assert_not_called()


class PersistenceTests(unittest.TestCase):
    def test_run_id_reuse_refused_before_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "runs" / "existing").mkdir(parents=True)
            with patch.object(runner, "DRIVE", root), patch.object(runner, "select_models") as select:
                with self.assertRaises(FileExistsError):
                    runner.execute(SimpleNamespace(run_id="existing"), None, None, {}, {})
            select.assert_not_called()

    def test_selection_and_models_are_persisted_before_holdout(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            assignment = root / "assignment"
            assignment.write_text("test fixture")
            run = root / "runs" / "fixture"
            args = SimpleNamespace(run_id="fixture", assignment=assignment)
            metadata = {"frozen_assignment_file_sha256": runner.hash_file(assignment)}
            names = {"final_pipe": None, "multi_lgbm": None, "feature_columns": ["dur"],
                     "version": lambda name: "fixture",
                     "joblib": SimpleNamespace(dump=lambda obj, out: out.write(b"fixture"))}
            selection = {"threshold": 0.5, "binary_model": "fixture"}

            def inspect_holdout(*_):
                self.assertEqual(json.loads((run / "selection.json").read_text()), selection)
                self.assertTrue((run / "ids_model.joblib").is_file())
                self.assertTrue((run / "attack_family_model.joblib").is_file())
                raise RuntimeError("stop before test access")

            with (patch.object(runner, "DRIVE", root),
                  patch.object(runner, "model_namespace", return_value=names),
                  patch.object(runner, "select_models", return_value=selection),
                  patch.object(runner, "open_official_holdout", side_effect=inspect_holdout)):
                with self.assertRaisesRegex(RuntimeError, "stop before test access"):
                    runner.execute(args, None, None, metadata, {})
            self.assertFalse((run / "run_manifest.json").exists())
