import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

import m03_validation_audit as audit


class ThresholdTests(unittest.TestCase):
    def test_strict_gate_and_infeasible_cost_benchmark(self):
        labels = np.array([False] * 100 + [True] * 20)
        scores = np.array([0.8] + [0.1] * 99 + [0.8] * 19 + [0.1])
        saved = audit.operating_point(labels, scores, 0.8)
        result = audit.boundary_audit(labels, scores, saved)
        self.assertEqual(saved["cost_minutes"], 125)
        self.assertFalse(result["jointly_feasible"])
        self.assertEqual(result["maximum_recall_under_fpr_cap"], 0)
        self.assertEqual(result["minimum_fpr_at_recall_floor"], 0.01)
        self.assertTrue(result["saved_threshold_matches_cost_and_tie_break"])
        self.assertNotIn("threshold", result)

    def test_full_boundary_sweep_agrees_with_brute_force(self):
        rng = np.random.default_rng(7)
        y = np.array([False] * 105 + [True] * 25)
        for scores in [rng.choice(np.linspace(0, 1, 21), len(y)),
                       np.r_[np.zeros(104), 0.4, np.ones(24), 0.0]]:
            points = [audit.operating_point(y, scores, x) for x in set(np.r_[scores, 0, 1])]
            feasible = [p for p in points if p["recall"] >= 0.95 and p["fpr"] < 0.01]
            ranked = sorted(feasible or points, key=lambda p: (p["cost_minutes"], p["fpr"], -p["threshold"]))
            result = audit.boundary_audit(y, scores, ranked[0])
            self.assertEqual(result["jointly_feasible"], bool(feasible))
            self.assertTrue(result["saved_threshold_matches_cost_and_tie_break"])
            self.assertEqual(result["maximum_recall_under_fpr_cap"],
                             max((p["recall"] for p in points if p["fpr"] < 0.01), default=None))
            if len(ranked) > 1:
                self.assertFalse(audit.boundary_audit(y, scores, ranked[1])[
                    "saved_threshold_matches_cost_and_tie_break"])

    def test_lower_fpr_then_higher_threshold_break_equal_cost(self):
        # Capturing one more attack costs 24 additional false alarms (120 minutes).
        y = np.array([True] + [False] * 24)
        scores = np.full(25, 0.5)
        low_fpr = audit.operating_point(y, scores, 1.0)
        self.assertEqual(low_fpr["cost_minutes"], audit.operating_point(y, scores, 0.5)["cost_minutes"])
        self.assertTrue(audit.boundary_audit(y, scores, low_fpr)["saved_threshold_matches_cost_and_tie_break"])
        endpoint = audit.operating_point(y, np.r_[1.0, np.zeros(24)], 1.0)
        self.assertTrue(audit.boundary_audit(y, np.r_[1.0, np.zeros(24)], endpoint)[
            "saved_threshold_matches_cost_and_tie_break"])


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        work = tempfile.TemporaryDirectory()
        self.addCleanup(work.cleanup)
        self.run = Path(work.name)
        self.manifest = {
            "run_id": audit.RUN_ID, "stage": "A", "status": "complete", "mode": "stage_a",
            "planned_fit_count": 3, "actual_fit_count": 3, "random_seed": 42,
            "official_test": "not opened", "snapshot_sha256": audit.SOURCE_SHA,
            "source_base_revision": audit.BASE_REVISION, "dependency_lock_sha256": audit.LOCK_SHA,
            "interpreter": str(audit.PYTHON), "sys_prefix": str(audit.PYTHON.parent.parent),
            "sys_base_prefix": "/usr", "python": "3.13.18", "matplotlib_backend": "Agg",
            "execution_enabled": True, "source_identity": "fixture source",
            "dataset_access": {"training": "verified", "official_test": "not opened"},
            "validation_policy": audit.POLICY,
        }
        self.write("run_started.json", {**self.manifest, "status": "started"})
        self.write("configuration.json", {**self.manifest, "status": "verified_no_fit_preflight"})
        self.manifest.update(models={}, model_parameters=dict.fromkeys(audit.MODELS, {}))
        for i, name in enumerate(audit.MODELS, 1):
            prefix = f"{i:02d}-{name}"
            binary = name.encode()
            (self.run / f"{name}.joblib").write_bytes(binary)
            fit = dict(status="complete", failure=None, seconds=1.0, warnings=[])
            report = dict(fit=fit, artifact={"filename": f"{name}.joblib", "sha256": audit.digest(binary)},
                          validation={"at_0_5": {"f1": 0.7}, "threshold_selection": {"threshold": 0.5}})
            self.manifest["models"][name] = report
            self.write(f"{prefix}.fit-started.json", {"model": name, "attempt": i})
            self.write(f"{prefix}.fit-result.json", fit)
            self.write(f"{prefix}.validation.json", report)
        self.manifest["selection"] = {
            "model": "random_forest", "rule": "tree validation F1 at 0.5; Random Forest first on ties",
            "threshold": 0.5,
        }
        self.write("run_manifest.json", self.manifest)
        self.anchor = audit.file_digest(self.run / "run_manifest.json")

    def write(self, name, data):
        (self.run / name).write_text(json.dumps(data))

    def test_valid_manifest_and_journals_verify_without_deserializing(self):
        with patch("joblib.load") as load:
            self.assertEqual(audit.verify_run(self.run, self.anchor), self.manifest)
            load.assert_not_called()

    def test_tamper_checks_fail_before_load(self):
        with patch("joblib.load") as load:
            with self.assertRaisesRegex(ValueError, "trust-anchor"):
                audit.verify_run(self.run, "a" * 64)
            (self.run / "lightgbm.joblib").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Model artifact mismatch"):
                audit.verify_run(self.run, self.anchor)
            with self.assertRaisesRegex(ValueError, "before deserialization"):
                audit.load_verified_model(self.run, "lightgbm", self.manifest["models"]["lightgbm"]["artifact"])
            load.assert_not_called()

    def test_extra_fit_and_changed_report_fail(self):
        extra = self.run / "04-extra.fit-started.json"
        extra.touch()
        with self.assertRaisesRegex(ValueError, "journal"):
            audit.verify_run(self.run, self.anchor)
        extra.unlink()
        self.write("01-logistic_regression.validation.json", {})
        with self.assertRaisesRegex(ValueError, "Validation report mismatch"):
            audit.verify_run(self.run, self.anchor)


class InferenceTests(unittest.TestCase):
    def test_mock_orchestration_writes_only_new_diagnostics(self):
        with tempfile.TemporaryDirectory() as folder:
            drive = Path(folder)
            run = drive / "runs" / audit.RUN_ID
            run.mkdir(parents=True)
            marker = run / "run_manifest.json"
            marker.write_text("fixture")
            manifest = {
                "actual_fit_count": 3, "official_test": "not opened", "source_identity": "fixture",
                "frozen_assignment_file_sha256": "assignment", "frozen_split_manifest_sha256": "split",
                "dataset_manifest_sha256": "dataset", "models": {
                    name: {"artifact": {"sha256": name}} for name in audit.MODELS
                },
            }
            split = {"training_sha256": "training", "assignment_sha256": "codes",
                     "support": {"validation": {"rows": 35068}}}
            expected = {"record_mismatches": [], "threshold_feasibility": {"jointly_feasible": False}}
            with (patch.object(audit, "DRIVE", drive), patch.object(audit, "verify_environment"),
                  patch.object(audit, "verify_run", return_value=manifest),
                  patch.object(audit, "stage_source"),
                  patch.object(audit, "reconstruct", return_value=("features", "val", "groups", split, "id")),
                  patch.object(audit, "load_verified_model") as load,
                  patch.object(audit, "audit_model", return_value=expected),
                  patch("builtins.print")):
                self.assertEqual(audit.main(["--audit-id", "new", "--trusted-run-manifest-sha256", "a" * 64]), 0)
            self.assertEqual([call.args[1] for call in load.call_args_list], list(audit.MODELS))
            self.assertEqual({path.name for path in run.iterdir()}, {"run_manifest.json"})
            self.assertEqual(marker.read_text(), "fixture")
            output = audit.read_json(drive / "audits/new/inference_audit.json")
            self.assertFalse(output["any_existing_model_jointly_feasible"])
            self.assertEqual(output["validation"]["ordered_position_group_label_sha256"], "id")

    def test_mock_model_only_scores_validation_and_preserves_saved_threshold(self):
        n, p = 11200, 23868
        labels = np.r_[np.zeros(n, dtype=int), np.ones(p, dtype=int)]
        scores = np.r_[np.zeros(n), np.ones(p)]
        features = pd.DataFrame({"feature": np.arange(n + p)})
        raw = pd.DataFrame({"label": labels})
        groups = np.arange(n + p)
        frozen_threshold = 1.0
        report = {"fit": {"status": "complete"}, "artifact": {"sha256": "fixture"},
                  "validation": {"support": {"rows": n+p, "normal": n, "attack": p},
                                 "attack_prevalence": p / (n+p), "roc_auc": 1.0,
                                 "at_0_5": audit.operating_point(labels == 1, scores, 0.5),
                                 "threshold_selection": {
                                     **audit.operating_point(labels == 1, scores, frozen_threshold),
                                     "status": "FEASIBLE", "purpose": "feasible_operating_point",
                                 }}}
        classifier = type("LogisticRegression", (), {"get_params": lambda self: {}})()
        pipe = Mock(feature_names_in_=["feature"], classes_=[0, 1],
                    named_steps={"pre": Mock(), "clf": classifier})
        pipe.predict_proba.return_value = np.column_stack((1-scores, scores))
        packages = dict.fromkeys(("numpy", "pandas", "scikit-learn", "lightgbm", "joblib", "shap"), "fixture")
        manifest = {"python": "fixture", "package_versions": packages,
                    "model_parameters": {"logistic_regression": {}}}
        artifact = dict(model_name="logistic_regression", feature_cols=["feature"], pipeline=pipe,
                        threshold=frozen_threshold, metadata={"artifact_version": 1,
                        "python": "fixture", "packages": packages})
        with patch.object(audit, "error_analysis"), patch.object(audit, "preprocessing_analysis"):
            result = audit.audit_model(artifact, "logistic_regression", features, raw, groups, report, manifest)
        pipe.predict_proba.assert_called_once_with(features)
        for name in ("fit", "fit_transform", "partial_fit"):
            getattr(pipe, name).assert_not_called()
        self.assertEqual(result["record_mismatches"], [])
        self.assertEqual(result["at_saved_threshold"]["threshold"], frozen_threshold)

    def test_aggregate_errors_and_transformed_collisions(self):
        val = pd.DataFrame({"label": [0, 1, 0, 1], "attack_cat": ["Normal", "Worms", "Normal", "DoS"],
                            "proto": ["tcp", "tcp", "other", "udp"],
                            "service": ["http", "http", "-", "http"], "state": ["FIN"] * 4})
        groups = np.array(["a", "a", "b", "c"])
        report = audit.error_analysis(val, groups, np.array([True, True, False, False]))
        self.assertEqual(report["mixed_binary_label_groups"]["false_positives"], 1)
        self.assertEqual(report["error_concentration"]["FP"]["rows_in_repeated_groups"], 1)
        self.assertEqual(sum(row["false_negatives"] for row in report["false_negatives_by_family"]), 1)
        encoder = SimpleNamespace(categories_=[["tcp", "udp"], ["http", "-"], ["FIN"]],
                                  infrequent_categories_=[["udp"], None, None],
                                  handle_unknown="infrequent_if_exist")
        pre = Mock(named_transformers_={"cat": encoder})
        pre.transform.return_value = np.array([[0., 1.], [-0., 1.], [1., 1.]])
        preprocessing = audit.preprocessing_analysis(pre, val, groups)
        self.assertEqual(preprocessing["encoder_coverage"]["proto"]["unknown_rows"], 1)
        self.assertEqual(preprocessing["transformed_exact_collisions"]["raw_groups_in_collisions"], 2)
        pre.fit.assert_not_called()

    def test_existing_output_is_refused_before_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            drive = Path(folder)
            (drive / "runs" / audit.RUN_ID).mkdir(parents=True)
            (drive / "audits/used").mkdir(parents=True)
            with (patch.object(audit, "DRIVE", drive), patch.object(audit, "verify_environment"),
                  patch.object(audit, "load_verified_model") as load):
                with self.assertRaisesRegex(ValueError, "already exists"):
                    audit.main(["--audit-id", "used", "--trusted-run-manifest-sha256", "a" * 64])
            load.assert_not_called()
