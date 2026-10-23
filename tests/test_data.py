"""Small artificial fixtures exercise ingestion, not empirical NASA findings."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from aevora.config import load_config
from aevora.data.loader import load_trajectory, load_rul_targets, load_split
from aevora.data.validation import COLUMNS, DataValidationError, quality_report


def row(unit=1, cycle=1):
    return [unit, cycle, 0, 0, 100, *range(1, 22)]


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, rows, name="train.txt"):
        path = self.root / name
        path.write_text("\n".join("  \t".join(map(str, r)) + "  " for r in rows) + "\n")
        return path

    def test_valid_training_and_test_schema(self):
        for name in ["train.txt", "test.txt"]:
            frame = load_trajectory(self.write([row(), row(cycle=2), row(unit=2)], name))
            self.assertEqual(frame.shape, (3, 26))
            self.assertEqual(list(frame.columns), COLUMNS)
            self.assertEqual(frame.unit_id.dtype, np.dtype("int64"))

    def test_missing_and_unreadable(self):
        with self.assertRaisesRegex(OSError, "Cannot read dataset"):
            load_trajectory(self.root / "missing")
        with patch.object(Path, "open", side_effect=PermissionError("denied")):
            with self.assertRaisesRegex(OSError, "denied"):
                load_trajectory(self.root / "protected")

    def test_column_counts_empty_and_blank_lines(self):
        for rows in [[row()[:-1]], [row() + [0]], [row(), []]]:
            with self.subTest(rows=rows), self.assertRaisesRegex(DataValidationError, "columns"):
                load_trajectory(self.write(rows))
        p = self.root / "empty"
        p.write_text("")
        with self.assertRaisesRegex(DataValidationError, "empty"):
            load_trajectory(p)

    def test_bad_numeric_and_nonfinite(self):
        for value in ["bad", "NaN", "inf", "-inf"]:
            values = row()
            values[5] = value
            with self.subTest(value=value), self.assertRaises(DataValidationError):
                load_trajectory(self.write([values]))

    def test_invalid_ids_and_cycles(self):
        for index in [0, 1]:
            for value in [0, -1, 1.5, float("nan"), float("inf"), 2**54]:
                values = row()
                values[index] = value
                with self.subTest(index=index, value=value), self.assertRaises(DataValidationError):
                    load_trajectory(self.write([values]))

    def test_duplicates_ordering_and_gaps(self):
        for cycles in [[1, 1], [2, 1], [1, 3], [2, 3]]:
            with self.subTest(cycles=cycles), self.assertRaises(DataValidationError):
                load_trajectory(self.write([row(cycle=c) for c in cycles]))

    def test_report_valid_and_nonmutating(self):
        frame = pd.DataFrame([row(), row(cycle=2), row(unit=2)], columns=COLUMNS)
        before = frame.copy(deep=True)
        report = quality_report(frame)
        self.assertTrue(report["valid"])
        self.assertEqual(report["unique_engines"], 2)
        self.assertEqual(report["cycles_per_engine"], {"1": 2, "2": 1})
        self.assertEqual(report["constant_sensors"], COLUMNS[5:])
        pd.testing.assert_frame_equal(frame, before)

    def test_invalid_report(self):
        frame = pd.DataFrame([row(), row()], columns=COLUMNS, dtype=float)
        frame.loc[0, "sensor_1"] = np.nan
        frame.loc[1, "sensor_2"] = np.inf
        report = quality_report(frame)
        self.assertFalse(report["valid"])
        self.assertEqual(report["missing_values"], 1)
        self.assertEqual(report["nonfinite_values"], 2)
        self.assertEqual(report["duplicate_engine_cycle_records"], 1)
        self.assertFalse(quality_report(frame.drop(columns="cycle"))["valid"])
        self.assertFalse(quality_report(pd.DataFrame(columns=COLUMNS))["valid"])

    def test_near_constant_threshold(self):
        frame = pd.DataFrame([row(cycle=i) for i in range(1, 101)], columns=COLUMNS)
        frame.loc[99, "sensor_1"] = 2
        self.assertIn("sensor_1", quality_report(frame)["near_constant_sensors"])
        with self.assertRaises(ValueError):
            quality_report(frame, 1.1)

    def test_target_engine_order_not_appearance_order(self):
        test = load_trajectory(self.write([row(unit=2), row(unit=1)]))
        targets = load_rul_targets(self.write([[10], [0]], "targets.txt"), test)
        self.assertEqual(targets.to_dict(), {1: 10, 2: 0})

    def test_invalid_targets(self):
        test = load_trajectory(self.write([row()]))
        for targets in [[[-1]], [[1.5]], [["bad"]], [["nan"]], [["inf"]], [[1, 2]], [[1], [2]]]:
            with self.subTest(targets=targets), self.assertRaises(DataValidationError):
                load_rul_targets(self.write(targets, "targets.txt"), test)

    def test_reject_ambiguous_target_alignment(self):
        test = load_trajectory(self.write([row(unit=2)]))
        with self.assertRaisesRegex(DataValidationError, "1..N"):
            load_rul_targets(self.write([[5]], "targets.txt"), test)

    def test_split_config_and_no_implicit_target_access(self):
        config = Path(__file__).resolve().parents[1] / "configs/default.toml"
        settings = load_config(config)
        with patch("aevora.data.loader.load_trajectory") as loader:
            load_split(settings)
            loader.assert_called_once_with(settings.train_path)
            load_split(settings, "test")
            self.assertEqual(loader.call_args.args[0], settings.test_path)
        with self.assertRaises(ValueError):
            load_split(settings, "targets")


if __name__ == "__main__":
    unittest.main()
