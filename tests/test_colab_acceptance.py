import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from colab_acceptance import load_assignment, verify_model_guard, verify_snapshot
from colab_snapshot import FILES, MAX_SNAPSHOT_BYTES, build_snapshot


ROOT = Path(__file__).resolve().parents[1]


class SnapshotAcceptanceTests(unittest.TestCase):
    def test_bootstrap_and_builder_agree_on_members_and_limit(self):
        cells = json.loads((ROOT / FILES[0]).read_text())["cells"]
        code = next("".join(cell["source"]) for cell in cells
                    if cell["source"] and cell["source"][0].startswith("# M03.4 source bootstrap"))
        assignments = {
            node.targets[0].id: node.value
            for node in ast.walk(ast.parse(code))
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
        }
        self.assertEqual(ast.literal_eval(assignments["expected_files"]), set(FILES))
        left = assignments["SNAPSHOT_MAX_BYTES"]
        self.assertIsInstance(left.op, ast.Mult)
        self.assertEqual(left.left.value * left.right.value, MAX_SNAPSHOT_BYTES)

    def test_snapshot_rejects_wrong_digest_and_changed_runtime_source(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            snapshot = folder / "source.zip"
            digest = build_snapshot(ROOT, snapshot)
            self.assertEqual(verify_snapshot(ROOT, snapshot, digest), digest)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                verify_snapshot(ROOT, snapshot, "0" * 64)
            with ZipFile(snapshot) as archive:
                for name in FILES:
                    path = folder / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(archive.read(name))
            (folder / "grouped_splits.py").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Runtime source differs"):
                verify_snapshot(folder, snapshot, digest)

    def test_model_guard_is_disabled(self):
        verify_model_guard(ROOT / FILES[0], model_execution=False)
        with self.assertRaisesRegex(ValueError, "disabled"):
            verify_model_guard(ROOT / FILES[0], model_execution=True)


class FrozenAssignmentTests(unittest.TestCase):
    def test_verifies_bytes_codes_and_metadata_separately(self):
        codes = "V01234"
        record = {"policy": "example", "assignment_sha256": hashlib.sha256(codes.encode()).hexdigest()}
        payload = json.dumps({**record, "assignments": codes}).encode()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "assignment.json"
            path.write_bytes(payload)
            manifest = {**record, "assignment_file_sha256": hashlib.sha256(payload).hexdigest()}
            self.assertEqual(load_assignment(path, manifest, 6), codes)
            with self.assertRaisesRegex(ValueError, "one valid assignment"):
                load_assignment(path, manifest, 7)
            with self.assertRaisesRegex(ValueError, "metadata"):
                load_assignment(path, {**manifest, "policy": "different"}, 6)
            path.write_bytes(payload + b" ")
            with self.assertRaisesRegex(ValueError, "file SHA-256"):
                load_assignment(path, manifest, 6)


if __name__ == "__main__":
    unittest.main()
