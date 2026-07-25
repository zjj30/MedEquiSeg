#!/usr/bin/env python3
"""Regression tests for the corrected ClinicDB and BUS-BRA splits."""

from __future__ import annotations

import sys
import unittest
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "smoke_tests"))

from build_group_disjoint_public5_manifests import (  # noqa: E402
    MANIFEST_DIR,
    recover_busbra_label,
    recover_busbra_patient,
    recover_cvc_sequence,
    rows_fingerprint,
)
from protocol_v3.core import read_manifest, validate_manifest_rows  # noqa: E402


def load(name: str) -> tuple[list[str], list[dict[str, str]]]:
    return read_manifest(MANIFEST_DIR / name)


def groups_by_split(rows: list[dict[str, str]]) -> dict[str, set[str]]:
    return {
        split: {row["group_id"] for row in rows if row["split"] == split}
        for split in ("train", "val", "test")
    }


def test_rows_from_development_groups(
    rows: list[dict[str, str]],
    recover,
) -> tuple[int, int]:
    development = {
        recover(row)
        for row in rows
        if row["split"] in {"train", "val"}
    }
    test = [row for row in rows if row["split"] == "test"]
    return sum(recover(row) in development for row in test), len(test)


class GroupDisjointManifestTests(unittest.TestCase):
    def assert_disjoint(self, rows: list[dict[str, str]]) -> None:
        groups = groups_by_split(rows)
        self.assertFalse(groups["train"] & groups["val"])
        self.assertFalse(groups["train"] & groups["test"])
        self.assertFalse(groups["val"] & groups["test"])

    def test_clinicdb_grouped_manifest(self) -> None:
        source_fields, source = load("medclipseg_clinicdb_full.csv")
        fields, grouped = load("medclipseg_clinicdb_grouped.csv")
        self.assertEqual(validate_manifest_rows(fields, grouped, check_files=False)["split_counts"],
                         {"test": 61, "train": 490, "val": 61})
        self.assertEqual(rows_fingerprint(source), rows_fingerprint(grouped))
        self.assertEqual({row["group_type"] for row in grouped}, {"video_sequence"})
        self.assertEqual(len({row["group_id"] for row in grouped}), 29)
        self.assert_disjoint(grouped)
        for row in grouped:
            self.assertEqual(
                row["group_id"],
                f"clinicdb_sequence_{recover_cvc_sequence(row['case_id']):02d}",
            )
        self.assertTrue(set(source_fields).issubset(fields))

    def test_busbra_grouped_manifest(self) -> None:
        source_fields, source = load("medclipseg_busbra_full.csv")
        fields, grouped = load("medclipseg_busbra_grouped.csv")
        self.assertEqual(validate_manifest_rows(fields, grouped, check_files=False)["split_counts"],
                         {"test": 282, "train": 1311, "val": 282})
        self.assertEqual(rows_fingerprint(source), rows_fingerprint(grouped))
        self.assertEqual({row["group_type"] for row in grouped}, {"patient"})
        self.assertEqual(len({row["group_id"] for row in grouped}), 1064)
        self.assert_disjoint(grouped)

        patient_labels: dict[str, set[str]] = defaultdict(set)
        patient_images = Counter()
        for row in grouped:
            prefix, _ = recover_busbra_patient(row["case_id"])
            expected = f"busbra_patient_{prefix}"
            self.assertEqual(row["patient_id"], expected)
            self.assertEqual(row["group_id"], expected)
            patient_labels[expected].add(recover_busbra_label(row["text"]))
            patient_images[recover_busbra_label(row["text"])] += 1
        self.assertTrue(all(len(labels) == 1 for labels in patient_labels.values()))
        self.assertEqual(Counter(next(iter(v)) for v in patient_labels.values()),
                         Counter({"benign": 722, "malignant": 342}))
        self.assertEqual(patient_images, Counter({"benign": 1268, "malignant": 607}))
        self.assertTrue(set(source_fields).issubset(fields))

    def test_legacy_group_leakage_is_reproduced(self) -> None:
        _, clinic = load("medclipseg_clinicdb_full.csv")
        _, busbra = load("medclipseg_busbra_full.csv")
        self.assertEqual(
            test_rows_from_development_groups(
                clinic,
                lambda row: recover_cvc_sequence(row["case_id"]),
            ),
            (61, 61),
        )
        self.assertEqual(
            test_rows_from_development_groups(
                busbra,
                lambda row: recover_busbra_patient(row["case_id"])[0],
            ),
            (203, 282),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
