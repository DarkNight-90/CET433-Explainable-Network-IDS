import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from grouped_splits import SplitPlan
import m03_runner as runner
import stage_a
import stage_b


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.plan = SplitPlan(np.array([f"group-{i}" for i in range(12)]), "00V11223344V")
        labels = pd.Series([0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1])
        features = pd.DataFrame({"dur": range(12)})
        fit, val = self.plan.fit, self.plan.validation
        self.fit = stage_a.Partition(features.iloc[fit], labels.iloc[fit])
        self.validation = stage_a.Partition(features.iloc[val], labels.iloc[val])
        self.folds = stage_b.frozen_folds(self.plan, self.fit.labels)

    def test_candidates_approved_distinct_and_ordered(self):
        config = stage_b.configuration()
        self.assertEqual(tuple(config["candidates"]), stage_b.MODELS)
        self.assertEqual(list(map(len, config["candidates"].values())), [5, 5])
        self.assertEqual(config, stage_b.configuration())
        for name, approved in (
            ("random_forest", ({100, 200}, {None, 15, 25}, {1, 2, 5})),
            ("lightgbm", ({200, 400}, {0.05, 0.1}, {31, 63}, {20, 50})),
        ):
            choices = [tuple(candidate.values()) for candidate in config["candidates"][name]]
            self.assertEqual(len(set(choices)), 5)
            self.assertTrue(all(values in allowed for choice in choices
                                for values, allowed in zip(choice, approved, strict=True)))

    def test_pipeline_templates_are_distinct_and_unfitted(self):
        lightgbm = Mock()
        with patch.dict(sys.modules, {"lightgbm": SimpleNamespace(LGBMClassifier=lightgbm)}):
            templates = stage_b.pipelines(["dur", "proto"], 42)
        self.assertEqual(tuple(templates), stage_b.MODELS)
        rf = templates["random_forest"]
        lgbm = templates["lightgbm"]
        self.assertEqual(tuple(rf.named_steps), ("pre", "clf"))
        self.assertIsNot(rf.named_steps.pre, lgbm.named_steps.pre)
        candidate = stage_b.candidate_pipeline(rf, {"n_estimators": 100})
        self.assertIsNot(rf.named_steps.pre, candidate.named_steps.pre)
        self.assertFalse(hasattr(candidate.named_steps.pre, "transformers_"))
        self.assertEqual(rf.named_steps.clf.n_jobs, 2)
        lightgbm.assert_called_once_with(verbose=-1, random_state=42, class_weight="balanced", n_jobs=2)

    def test_frozen_fold_order_and_group_crossings(self):
        held = [self.fit.features.index[list(b)].tolist() for _, b in self.folds]
        self.assertEqual(held, [[0, 1], [3, 4], [5, 6], [7, 8], [9, 10]])
        self.plan.groups[3] = self.plan.groups[0]
        with self.assertRaisesRegex(ValueError, "crosses a CV fold"):
            stage_b.frozen_folds(self.plan, self.fit.labels)
        self.plan.groups[2] = self.plan.groups[0]
        with self.assertRaisesRegex(ValueError, "fit/validation"):
            stage_b.frozen_folds(self.plan, self.fit.labels)

    def test_50_cv_fits_two_refits_and_validation_never_used_by_cv(self):
        templates = dict.fromkeys(stage_b.MODELS, object())
        made = []

        def clone(*_):
            pipe = Mock(classes_=[0, 1], named_steps={"clf": Mock(get_params=Mock(return_value={}))})
            pipe.predict_proba.return_value = np.array([[0.9, 0.1], [0.1, 0.9]])
            made.append(pipe)
            return pipe

        scorer = Mock(return_value=0.9)
        record, persist = Mock(), Mock()
        journal = stage_b.FitJournal(record)
        with (patch.object(stage_b, "candidate_pipeline", side_effect=clone),
              patch("sklearn.metrics.get_scorer", return_value=scorer) as scorer_factory):
            reports, winner = stage_b.evaluate_models(templates, self.fit, self.validation,
                                                        self.folds, journal, persist)
        scorer_factory.assert_called_once_with("f1")
        self.assertEqual(journal.counts, {"cv": 50, "refit": 2})
        for name in stage_b.MODELS:
            self.assertEqual(journal.per_model[name], {"cv": 25, "refit": 1})
            self.assertEqual(reports[name]["cv_selection"]["best_candidate"], 1)
        self.assertEqual(scorer.call_count, 50)
        self.assertEqual(winner["model"], "random_forest")
        self.assertEqual(persist.call_count, 2)
        for call_index, (pipe, score_call) in enumerate(zip(made[:50], scorer.call_args_list, strict=True)):
            pipe.fit.assert_called_once()
            pipe.predict_proba.assert_not_called()
            fit_rows, labels = pipe.fit.call_args.args
            scored_pipe, held_rows, held_labels = score_call.args
            train, held = self.folds[call_index % 5]
            self.assertIs(scored_pipe, pipe)
            self.assertEqual(list(fit_rows.index), list(self.fit.features.index[train]))
            self.assertEqual(list(held_rows.index), list(self.fit.features.index[held]))
            self.assertEqual(list(labels.index), list(fit_rows.index))
            self.assertEqual(list(held_labels.index), list(held_rows.index))
        for pipe in made[-2:]:
            pipe.fit.assert_called_once_with(self.fit.features, self.fit.labels)
            pipe.predict_proba.assert_called_once_with(self.validation.features)
        extra = Mock()
        for phase in ("cv", "refit"):
            with self.assertRaisesRegex(RuntimeError, "budget"):
                journal.fit(extra, self.fit.features, self.fit.labels, phase, {"model": "random_forest"})
        extra.fit.assert_not_called()

    def test_unexpected_model_and_automatic_refit_rejected_before_fit(self):
        journal = stage_b.FitJournal(Mock())
        with self.assertRaisesRegex(ValueError, "model set"):
            stage_b.search({"knn": object()}, self.fit, self.folds, journal, refit=False)
        with self.assertRaisesRegex(ValueError, "refit=False"):
            stage_b.search(dict.fromkeys(stage_b.MODELS), self.fit, self.folds, journal, refit=True)
        self.assertEqual(journal.counts, {"cv": 0, "refit": 0})


class RunnerTests(unittest.TestCase):
    def test_cli_explicit_exclusive_and_default_no_fit(self):
        flags = ["--snapshot", runner.STAGE_B_SNAPSHOT, "--snapshot-sha256", "a" * 64,
                 "--source-revision", "a" * 40, "--lock", "l", "--assignment", "a"]
        with (patch.object(runner, "preflight", return_value=(None, None, {}, {})),
              patch.object(runner, "execute_stage_a") as a,
              patch.object(runner, "execute_stage_b") as b,
              patch.object(runner, "execute") as legacy, patch("builtins.print")):
            runner.main(flags)
            b.assert_not_called()
            runner.main([*flags, "--execute-stage-b", "--run-id", "fixture"])
            b.assert_called_once()
            for rejected in (["--execute-stage-b"], ["--run-id", "fixture"],
                             ["--execute-stage-b", "--execute-stage-a", "--run-id", "fixture"],
                             ["--execute-stage-b", "--execute-m03", "--run-id", "fixture"],
                             ["--execute-m03", "--run-id", "fixture"]):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    runner.main([*flags, *rejected])
            a.assert_not_called()
            legacy.assert_not_called()

    def test_stage_b_preflight_uses_training_only_and_constructs_no_models(self):
        args = SimpleNamespace(snapshot=runner.STAGE_B_SNAPSHOT, snapshot_sha256="s", lock="l",
                               assignment="a", source_revision="r", execute_m03=False,
                               execute_stage_a=False, execute_stage_b=False)
        smoke = SimpleNamespace(stdout='{"matplotlib_backend":"Agg"}')
        raw = pd.DataFrame({"label": [0, 1] * 6})
        plan = SplitPlan(np.arange(12), "0011223344VV")
        with (patch.object(runner, "verify_environment"),
              patch.object(runner, "verify_controlled_paths"),
              patch.dict(os.environ, dict.fromkeys(stage_b.THREAD_VARIABLES, "2")),
              patch.object(runner, "verify_snapshot", return_value="s"),
              patch.object(runner, "verify_lock", return_value="l"),
              patch.object(runner.subprocess, "run", return_value=smoke) as subprocess,
              patch.object(runner, "hash_file", return_value="a"),
              patch.object(runner, "load_verified_split", return_value=(raw, plan, {})) as split,
              patch.multiple(stage_a, TOTAL_ROWS=12, FIT_ROWS=10, VALIDATION_ROWS=2),
              patch.object(runner, "reviewed_cells") as cells,
              patch.object(runner, "open_official_holdout") as holdout,
              patch.object(stage_b, "pipelines") as pipelines):
            _, _, meta, compiled = runner.preflight(args)
        self.assertEqual(split.call_args.args[1].name, "UNSW_NB15_training-set.csv")
        self.assertIn("--without-shap", subprocess.call_args.args[0])
        self.assertEqual(compiled, {})
        self.assertEqual(meta["mode"], "preflight")
        self.assertFalse(meta["execution_enabled"])
        self.assertEqual(meta["dataset_access"]["official_test"], "not opened")
        self.assertEqual(len(meta["stage_b_plan"]["verified_fold_sizes"]), 5)
        cells.assert_not_called()
        holdout.assert_not_called()
        pipelines.assert_not_called()

    def test_thread_cap_rejects_unbounded_native_pool_environment(self):
        with patch.dict(os.environ, dict.fromkeys(stage_b.THREAD_VARIABLES, "2"), clear=True):
            stage_b.verify_thread_environment()
            with patch.dict(os.environ, {"OPENBLAS_NUM_THREADS": "3"}):
                with self.assertRaisesRegex(ValueError, "OPENBLAS_NUM_THREADS"):
                    stage_b.verify_thread_environment()

    def test_manifest_written_last_only_for_complete_mock_run(self):
        with tempfile.TemporaryDirectory() as folder:
            drive = Path(folder)
            assignment = drive / "assignment.json"
            assignment.write_text("{}")
            args = SimpleNamespace(assignment=assignment, run_id="complete", execute_stage_b=True)
            metadata = {"mode": "stage_b", "split": {"training_sha256": "fixture"},
                        "frozen_assignment_file_sha256": runner.hash_file(assignment)}
            plan = SimpleNamespace(assignment_sha256="fixture")
            partition = stage_a.Partition("features", "labels")

            def evaluate(templates, fit, validation, folds, journal, persist):
                journal.record("cv_results.json", {})
                reports = {}
                for name in stage_b.MODELS:
                    journal.per_model[name].update(cv=25, refit=1)
                    reports[name] = {"artifact": persist(name, None, 0.5)}
                journal.counts.update(cv=50, refit=2)
                return reports, {"model": "random_forest"}

            with (patch.object(runner, "DRIVE", drive),
                  patch.object(runner, "verify_controlled_paths"),
                  patch.object(stage_b, "verify_thread_environment"),
                  patch.object(stage_a, "partitions", return_value=(partition, partition)),
                  patch.object(stage_b, "frozen_folds"),
                  patch.object(stage_b, "pipelines", return_value={}),
                  patch.object(stage_b, "evaluate_models", side_effect=evaluate),
                  patch.object(runner, "write_json", wraps=runner.write_json) as record,
                  patch("joblib.dump", side_effect=lambda artifact, out: out.write(b"mock model"))):
                runner.execute_stage_b(args, None, plan, metadata)
            self.assertEqual(record.call_args_list[-1].args[0].name, "run_manifest.json")
            run = drive / "runs/complete"
            manifest = json.loads((run / "run_manifest.json").read_text())
            self.assertEqual(manifest["actual_fit_count"], 52)
            self.assertEqual(manifest["official_test"], "not opened")
            for report in manifest["models"].values():
                artifact = report["artifact"]
                self.assertEqual(artifact["sha256"], runner.hash_file(run / artifact["filename"]))

    def test_failed_and_interrupted_fits_are_journaled_without_overwrite(self):
        for failure, status in ((RuntimeError("mock failure"), "failed"),
                                (KeyboardInterrupt(), "interrupted")):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as folder:
                drive = Path(folder)
                assignment = drive / "assignment.json"
                assignment.write_text("{}")
                args = SimpleNamespace(assignment=assignment, run_id="test", execute_stage_b=True)
                metadata = {"mode": "stage_b", "split": {"training_sha256": "fixture"},
                            "frozen_assignment_file_sha256": runner.hash_file(assignment)}
                plan = SimpleNamespace(assignment_sha256="fixture")
                partition = stage_a.Partition("features", "labels")
                pipe = Mock()
                pipe.fit.side_effect = failure

                def fail(templates, fit, validation, folds, journal, persist):
                    journal.fit(pipe, fit.features, fit.labels, "cv",
                                {"model": "random_forest", "candidate": 1, "fold": 0})

                with (patch.object(runner, "DRIVE", drive),
                      patch.object(runner, "verify_controlled_paths"),
                      patch.object(stage_b, "verify_thread_environment"),
                      patch.object(stage_a, "partitions", return_value=(partition, partition)),
                      patch.object(stage_b, "frozen_folds"),
                      patch.object(stage_b, "pipelines", return_value={}),
                      patch.object(stage_b, "evaluate_models", side_effect=fail),
                      patch.object(runner, "open_official_holdout") as holdout,
                      patch("joblib.load") as load):
                    with self.assertRaises(type(failure)):
                        runner.execute_stage_b(args, None, plan, metadata)
                    with self.assertRaises(FileExistsError):
                        runner.execute_stage_b(args, None, plan, metadata)
                load.assert_not_called()
                holdout.assert_not_called()
                pipe.fit.assert_called_once()
                run = drive / "runs/test"
                failure_record = json.loads((run / "run_failure.json").read_text())
                self.assertEqual(failure_record["status"], status)
                self.assertEqual(failure_record["fit_counts"], {"cv": 1, "refit": 0})
                result = json.loads((run / "cv-01.fit-result.json").read_text())
                self.assertEqual(result["status"], status)
                self.assertFalse((run / "run_manifest.json").exists())
