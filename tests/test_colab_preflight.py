import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from colab_preflight import DRIVE_PROJECT, dataset_directory, runtime_inventory, verify_partition
from colab_snapshot import FILES, build_snapshot


ROOT = Path(__file__).resolve().parents[1]


class ConfigurationTests(unittest.TestCase):
    def test_local_cloud_and_override_paths(self):
        self.assertEqual(dataset_directory(ROOT), ROOT / "data/official")
        self.assertEqual(dataset_directory(ROOT, cloud=True), DRIVE_PROJECT / "data/official")
        self.assertEqual(dataset_directory(ROOT, cloud=True, override="/tmp/reviewed"), Path("/tmp/reviewed"))
        with self.assertRaisesRegex(ValueError, "absolute"):
            dataset_directory(ROOT, override="relative/path")

    def test_inventory_handles_an_absent_package(self):
        with tempfile.TemporaryDirectory() as directory:
            pins = Path(directory) / "requirements.txt"
            pins.write_text("# comment\n\ncet433-test-package-not-installed==1.0\n")
            inventory = runtime_inventory(pins)
            self.assertTrue(inventory["python"])
            self.assertEqual(inventory["packages"]["cet433-test-package-not-installed"],
                             {"pinned": "1.0", "installed": None, "matches": False})

    def test_source_snapshot_is_repeatable_and_allowlisted(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = [Path(directory) / name for name in ("a.zip", "b.zip")]
            self.assertEqual(build_snapshot(ROOT, a), build_snapshot(ROOT, b))
            with ZipFile(a) as archive:
                self.assertEqual(archive.namelist(), list(FILES))
                for name in FILES:
                    self.assertEqual(archive.read(name), (ROOT / name).read_bytes())
            with self.assertRaises(FileExistsError):
                build_snapshot(ROOT, a)


class PartitionTests(unittest.TestCase):
    def setUp(self):
        self.columns = ["id", "attack_cat", "label"]
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "fixture.csv"

    def write_fixture(self, data):
        payload = data.encode("utf-8-sig")
        self.path.write_bytes(payload)
        return {"size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
                "validation": {"rows": 2, "columns": 3, "binary_labels": {"0": 1, "1": 1},
                               "attack_categories": {"Normal": 1, "Generic": 1}}}

    def test_bom_schema_and_targets(self):
        metadata = self.write_fixture("id,attack_cat,label\r\n1,Normal,0\r\n2,Generic,1\r\n")
        self.assertEqual(verify_partition(self.path, self.columns, metadata)["rows"], 2)
        metadata["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            verify_partition(self.path, self.columns, metadata)

    def test_rejects_wrong_header_target_and_dimensions(self):
        for data, reason in (
            ("id,label,attack_cat\n1,0,Normal\n2,1,Generic\n", "schema"),
            ("id,attack_cat,label\n1,Normal,1\n2,Generic,1\n", "targets"),
            ("id,attack_cat,label\n1,Normal,0\n2,Unknown,1\n", "targets"),
            ("id,attack_cat,label\n1,Normal,0\n", "dimensions"),
            ("id,attack_cat,label\n1,Normal,0,extra\n", "field count"),
        ):
            with self.subTest(reason=reason):
                metadata = self.write_fixture(data)
                with self.assertRaisesRegex(ValueError, reason):
                    verify_partition(self.path, self.columns, metadata)

    def test_default_notebook_guard_stops_before_model_imports(self):
        path = ROOT / "CET433_Project2_IDS_M01R.ipynb"
        cells = json.loads(path.read_text())["cells"]
        guard = next("".join(cell["source"]) for cell in cells
                     if any("ALLOW_MODEL_EXECUTION =" in line for line in cell["source"]))
        with self.assertRaisesRegex(RuntimeError, "stops at preflight"):
            exec(compile(guard, str(path), "exec"), {})


if __name__ == "__main__":
    unittest.main()
