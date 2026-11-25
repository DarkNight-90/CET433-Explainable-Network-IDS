import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from grouped_splits import SplitPlan
import m03_runner as runner
import stage_a


class StageATests(unittest.TestCase):
    def setUp(self):
        self.raw = pd.DataFrame({
            "dur": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
            "proto": ["tcp", "udp", "tcp", "validation", "validation", "udp"],
            "label": [1, 0, 1, 0, 1, 0],
        })
        self.plan = SplitPlan(np.array(list("abcdeZ")), "01VV23")
        with patch.multiple(stage_a, TOTAL_ROWS=6, FIT_ROWS=4, VALIDATION_ROWS=2):
            self.fit, self.validation = stage_a.partitions(self.raw, self.plan, ["dur", "proto"])

    def pipelines(self):
        return {name: Mock(classes_=[0, 1], **{
            "predict_proba.return_value": np.array([[0.9, 0.1], [0.1, 0.9]]),
        }) for name in stage_a.MODEL_ORDER}

    def test_frozen_positions_and_preprocessing_boundaries(self):
        self.assertEqual(list(self.fit.features.index), [0, 1, 4, 5])
        self.assertEqual(list(self.validation.features.index), [2, 3])
        # Use the real pipeline builder with only the LightGBM constructor mocked.
        with patch.dict(sys.modules, {"lightgbm": SimpleNamespace(LGBMClassifier=Mock())}):
            pipes = stage_a.model_pipelines(self.fit.features.columns)
        prep = [pipe.named_steps["pre"] for pipe in pipes.values()]
        self.assertEqual(len({id(item) for item in prep}), 3)
        for item in prep:
            self.assertFalse(hasattr(item, "transformers_"))
            self.assertEqual(item.transformers[0][1].named_steps["imp"].strategy, "median")
            self.assertEqual(item.transformers[1][1].min_frequency, 0.005)

        pipes = self.pipelines()
        for pipe in pipes.values():
            pipe.fit.side_effect = lambda *_: warnings.warn("synthetic", ConvergenceWarning)
        reports = stage_a.run_models(pipes, self.fit, self.validation, Mock(), Mock())
        for name, pipe in pipes.items():
            pipe.fit.assert_called_once_with(self.fit.features, self.fit.labels)
            pipe.predict_proba.assert_called_once_with(self.validation.features)
            self.assertEqual(reports[name]["fit"]["warnings"][0]["category"], "ConvergenceWarning")
        self.assertEqual(stage_a.selection(reports)["model"], "random_forest")

    def test_group_overlap_is_rejected(self):
        overlapping = SplitPlan(np.array(list("aabdeZ")), "0V1V23")
        with patch.multiple(stage_a, TOTAL_ROWS=6, FIT_ROWS=4, VALIDATION_ROWS=2):
            with self.assertRaisesRegex(ValueError, "overlap"):
                stage_a.partitions(self.raw, overlapping, ["dur", "proto"])

    def test_budget_rejects_missing_extra_and_reordered_candidates(self):
        pipes = self.pipelines()
        for invalid in ({**pipes, "knn": Mock()}, dict(list(pipes.items())[:2]),
                        dict(reversed(list(pipes.items())))):
            with self.assertRaisesRegex(ValueError, "exactly"):
                stage_a.run_models(invalid, self.fit, self.validation, Mock(), Mock())
        for pipe in pipes.values():
            pipe.fit.assert_not_called()

    def test_configuration_is_deterministic(self):
        lightgbm = Mock()
        with patch.dict(sys.modules, {"lightgbm": SimpleNamespace(LGBMClassifier=lightgbm)}):
            models = stage_a.model_pipelines(["dur", "proto"])
            repeat = stage_a.model_pipelines(["dur", "proto"])
        self.assertEqual(tuple(models), ("logistic_regression", "random_forest", "lightgbm"))
        for name, required in {
            "logistic_regression": {"solver": "lbfgs", "C": 1.0, "max_iter": 1000},
            "random_forest": {"n_estimators": 100, "max_depth": 15, "min_samples_leaf": 2, "n_jobs": 2},
        }.items():
            actual = models[name].named_steps["clf"].get_params()
            self.assertEqual(actual, repeat[name].named_steps["clf"].get_params())
            for key, value in {**required, "class_weight": "balanced", "random_state": 42}.items():
                self.assertEqual(actual[key], value)
        lightgbm.assert_called_with(n_estimators=200, learning_rate=0.05, num_leaves=31,
                                    min_child_samples=20, n_jobs=2, verbose=-1,
                                    class_weight="balanced", random_state=42)
        self.assertEqual(lightgbm.call_args_list[0], lightgbm.call_args_list[1])

    def test_threshold_gate_uses_strict_fpr_and_higher_threshold_for_ties(self):
        labels = [0] * 100 + [1] * 20
        scores = [0.8] + [0.1] * 99 + [0.8] * 19 + [0.1]
        result = stage_a.validation_report(labels, scores)
        self.assertEqual(result["threshold_selection"]["status"], "INFEASIBLE")
        self.assertEqual(result["at_0_5"]["fpr"], 0.01)
        scores[0] = 0.1
        result = stage_a.validation_report(labels, scores)["threshold_selection"]
        self.assertEqual((result["status"], result["threshold"], result["recall"], result["FN"]),
                         ("FEASIBLE", 0.8, 0.95, 1))
        perfect = stage_a.validation_report([0, 1], [0.1, 1.0])["threshold_selection"]
        self.assertEqual(perfect["threshold"], 1.0)


class StageALaunchTests(unittest.TestCase):
    def test_flag_dispatch_is_explicit_and_exclusive(self):
        flags = ["--snapshot", runner.STAGE_A_SNAPSHOT, "--snapshot-sha256", "a" * 64,
                 "--source-revision", "a" * 40, "--lock", "l", "--assignment", "a"]
        with (patch.object(runner, "preflight", return_value=(None, None, {}, {})),
              patch.object(runner, "execute_stage_a") as stage,
              patch.object(runner, "execute") as legacy, patch("builtins.print")):
            runner.main(flags)
            stage.assert_not_called()
            runner.main([*flags, "--execute-stage-a", "--run-id", "fixture"])
            stage.assert_called_once()
            for invalid in (["--run-id", "fixture"], ["--execute-stage-a"],
                            ["--execute-m03", "--execute-stage-a", "--run-id", "fixture"],
                            ["--execute-m03", "--run-id", "fixture"]):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    runner.main([*flags, *invalid])
            legacy.assert_not_called()

    def test_preflight_rejects_test_paths_and_symlink_before_reading(self):
        with tempfile.TemporaryDirectory() as folder:
            drive = Path(folder)
            args = SimpleNamespace(snapshot=drive / "snapshots" / runner.STAGE_A_SNAPSHOT,
                                   assignment=drive / "splits/m02.2-split-assignments.json",
                                   lock=drive / "environment/colab-py313-v1.lock")
            test = drive / "data/official/UNSW_NB15_testing-set.csv"
            training = test.with_name("UNSW_NB15_training-set.csv")
            for path in (args.snapshot, args.assignment, args.lock, training, test):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            with (patch.object(runner, "DRIVE", drive),
                  patch.object(runner, "DRIVE_MOUNT", Mock(is_mount=Mock(return_value=True)))):
                runner.verify_stage_a_paths(args)
                with patch.object(args, "lock", test), self.assertRaisesRegex(ValueError, "canonical"):
                    runner.verify_stage_a_paths(args)
                training.unlink()
                training.symlink_to(test)
                with self.assertRaisesRegex(ValueError, "canonical"):
                    runner.verify_stage_a_paths(args)

    def test_stage_a_preflight_does_not_select_or_load_holdout_cells(self):
        args = SimpleNamespace(snapshot=runner.STAGE_A_SNAPSHOT, snapshot_sha256="s",
                               lock="l", assignment="a", source_revision="r", execute_m03=False,
                               execute_stage_a=False)
        smoke = SimpleNamespace(stdout='{"matplotlib_backend":"Agg"}')
        with (patch.object(runner, "verify_environment"),
              patch.object(runner, "verify_stage_a_paths"),
              patch.object(runner, "verify_snapshot", return_value="s"),
              patch.object(runner, "verify_lock", return_value="l"),
              patch.object(runner.subprocess, "run", return_value=smoke),
              patch.object(runner, "hash_file", return_value="a"),
              patch.object(runner, "load_verified_split", return_value=(None, None, {})) as split,
              patch.object(runner, "reviewed_cells") as cells,
              patch.object(runner, "open_official_holdout") as holdout):
            _, _, meta, _ = runner.preflight(args)
        self.assertEqual(split.call_args.args[1].name, "UNSW_NB15_training-set.csv")
        self.assertFalse(meta["execution_enabled"])
        self.assertEqual(meta["dataset_access"]["official_test"], "not opened")
        cells.assert_not_called()
        holdout.assert_not_called()

    def test_failed_mock_fit_is_journaled_and_existing_id_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            drive = Path(folder)
            assignment = drive / "assignment.json"
            assignment.write_text("{}")
            args = SimpleNamespace(assignment=assignment, run_id="fixture", execute_stage_a=True)
            meta = {"frozen_assignment_file_sha256": runner.hash_file(assignment),
                    "split": {"training_sha256": "fixture"}}
            plan = SimpleNamespace(assignment_sha256="fixture")
            run = drive / "runs/fixture"
            pipelines = {name: Mock() for name in stage_a.MODEL_ORDER}
            for pipe in pipelines.values():
                pipe.named_steps = {"clf": Mock(get_params=Mock(return_value={}))}
            first = pipelines[stage_a.MODEL_ORDER[0]]
            first.fit.side_effect = RuntimeError("mock fit failed")
            partition = stage_a.Partition("features", "labels")
            with (patch.object(runner, "DRIVE", drive),
                  patch.object(runner, "DRIVE_MOUNT", Mock(is_mount=Mock(return_value=True))),
                  patch.object(stage_a, "partitions", return_value=(partition, partition)),
                  patch.object(stage_a, "model_pipelines", return_value=pipelines),
                  patch.object(runner, "open_official_holdout") as holdout):
                with self.assertRaisesRegex(RuntimeError, "mock fit failed"):
                    runner.execute_stage_a(args, None, plan, meta)
                with self.assertRaises(FileExistsError):
                    runner.execute_stage_a(args, None, plan, meta)
            first.fit.assert_called_once()
            pipelines["random_forest"].fit.assert_not_called()
            pipelines["lightgbm"].fit.assert_not_called()
            holdout.assert_not_called()
            self.assertEqual(json.loads((run / "run_failure.json").read_text())["status"], "failed")
            self.assertEqual(json.loads((run / "01-logistic_regression.fit-result.json").read_text())
                             ["failure"]["message"], "mock fit failed")
            self.assertFalse((run / "run_manifest.json").exists())

    def test_completed_mock_run_has_three_hashed_models_and_manifest_written_last(self):
        with tempfile.TemporaryDirectory() as folder:
            drive = Path(folder)
            assignment = drive / "assignment.json"
            assignment.write_text("{}")
            args = SimpleNamespace(assignment=assignment, run_id="complete", execute_stage_a=True)
            meta = {"frozen_assignment_file_sha256": runner.hash_file(assignment),
                    "split": {"training_sha256": "fixture"}}
            plan = SimpleNamespace(assignment_sha256="fixture")
            run = drive / "runs/complete"
            pipes = {name: Mock(classes_=[0, 1]) for name in stage_a.MODEL_ORDER}
            for pipe in pipes.values():
                pipe.named_steps = {"clf": Mock(get_params=Mock(return_value={}))}
                pipe.predict_proba.return_value = np.array([[0.1, 0.9], [0.9, 0.1]])
            partition = stage_a.Partition("features", [1, 0])
            with (patch.object(runner, "DRIVE", drive),
                  patch.object(runner, "DRIVE_MOUNT", Mock(is_mount=Mock(return_value=True))),
                  patch.object(stage_a, "partitions", return_value=(partition, partition)),
                  patch.object(stage_a, "model_pipelines", return_value=pipes),
                  patch.object(runner, "write_json", wraps=runner.write_json) as record,
                  patch.object(runner, "open_official_holdout") as holdout,
                  patch("joblib.dump", side_effect=lambda artifact, out: out.write(b"mock model"))):
                runner.execute_stage_a(args, None, plan, meta)
            holdout.assert_not_called()
            self.assertEqual(record.call_args_list[-1].args[0].name, "run_manifest.json")
            manifest = json.loads((run / "run_manifest.json").read_text())
            self.assertEqual(manifest["actual_fit_count"], 3)
            self.assertEqual(manifest["official_test"], "not opened")
            self.assertEqual(set(manifest["models"]), set(stage_a.MODEL_ORDER))
            for name, report in manifest["models"].items():
                pipes[name].fit.assert_called_once()
                artifact = report["artifact"]
                self.assertEqual(artifact["sha256"], runner.hash_file(run / artifact["filename"]))
