import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from grouped_splits import build_split_plan, feature_groups, split_record, write_assignment


class FingerprintTests(unittest.TestCase):
    def test_order_and_normalizations(self):
        x = pd.DataFrame({"dur": [0.0, -0.0, None, np.nan, 1],
                          "proto": ["tcp", "tcp", None, pd.NA, "Tcp"]})
        columns = ["dur", "proto"]
        a, b, c, d, e = feature_groups(x, columns)
        self.assertEqual(a, b)
        self.assertEqual(c, d)
        self.assertEqual(len({a, c, e}), 3)
        self.assertNotEqual(e, feature_groups(x.assign(proto="tcp"), columns)[-1])
        with self.assertRaisesRegex(ValueError, "ordered"):
            feature_groups(x[columns[::-1]], columns)
        with self.assertRaisesRegex(ValueError, "ordered"):
            feature_groups(x.assign(label=0), [*columns, "label"])

    def test_distinct_tuples_cannot_silently_share_a_digest(self):
        x = pd.DataFrame({"dur": [1, 2]})
        with patch("grouped_splits.hashlib.sha256") as digest:
            digest.return_value.hexdigest.return_value = "duplicate digest"
            with self.assertRaisesRegex(ValueError, "collision"):
                feature_groups(x, ["dur"])

    def test_unsafe_numeric_values_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "safe numeric"):
            feature_groups(pd.DataFrame({"dur": [2**53 + 1]}), ["dur"])


class GroupSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        categories = ["Normal", "Generic", "Worms"]
        data = [(n, c, int(c != "Normal")) for n in range(60) for c in categories]
        cls.rows = pd.DataFrame(data, columns=["dur", "attack_cat", "label"])
        cls.features = cls.rows[["dur"]]
        cls.plan = cls.build()

    @classmethod
    def build(cls):
        return build_split_plan(cls.features, ["dur"], cls.rows.attack_cat, cls.rows.label)

    def test_all_boundaries_have_disjoint_groups_and_class_support(self):
        plan = self.plan
        boundaries = [(plan.fit, plan.validation)]
        boundaries += [(plan.fit[a], plan.fit[b]) for a, b in plan.cv().split()]
        self.assertEqual(len(boundaries), 6)
        for left, right in boundaries:
            self.assertTrue(set(plan.groups[left]).isdisjoint(plan.groups[right]))
            for side in (left, right):
                self.assertEqual(set(self.rows.iloc[side].label), {0, 1})
                self.assertEqual(set(self.rows.iloc[side].attack_cat), {"Normal", "Generic", "Worms"})
        self.assertEqual(len(plan.fit) + len(plan.validation), len(self.rows))

    def test_conflicting_targets_share_an_assignment_without_rewriting(self):
        for i in range(0, len(self.rows), 3):
            self.assertEqual(len(set(self.plan.assignments[i:i+3])), 1)
        self.assertEqual(self.rows.label.value_counts().to_dict(), {1: 120, 0: 60})

    def test_persisted_identifiers_are_stable_and_no_overwrite(self):
        other = self.build()
        self.assertEqual(other.assignments, self.plan.assignments)
        self.assertEqual(other.assignment_sha256, self.plan.assignment_sha256)
        record = split_record(self.plan, self.rows, "1" * 64)
        with tempfile.TemporaryDirectory() as work:
            path = Path(work) / "assignments.json"
            write_assignment(path, self.plan, record)
            saved = json.loads(path.read_text())
            self.assertEqual(saved["training_sha256"], "1" * 64)
            self.assertEqual(saved["assignment_sha256"], hashlib.sha256(saved["assignments"].encode()).hexdigest())
            self.assertEqual(saved["assignments"], other.assignments)
            with self.assertRaises(FileExistsError):
                write_assignment(path, self.plan, record)

    def test_insufficient_class_support_is_rejected(self):
        rows = self.rows[self.rows.dur.lt(3)]
        with self.assertRaises(ValueError):
            build_split_plan(rows[["dur"]], ["dur"], rows.attack_cat, rows.label)


if __name__ == "__main__":
    unittest.main()
