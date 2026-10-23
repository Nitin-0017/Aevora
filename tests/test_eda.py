"""EDA integration tests use explicitly artificial temporary data only."""

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from aevora.config import load_config
from aevora.eda import run_eda


class EDATests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        default = load_config(Path(__file__).resolve().parents[1] / "configs/default.toml")
        self.settings = replace(default, raw_data_dir=self.root / "raw", artifact_dir=self.root / "out")

    def test_missing_data_creates_no_reports(self):
        with self.assertRaises(OSError):
            run_eda(self.settings)
        self.assertFalse(self.settings.artifact_dir.exists())

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "Plotting dependency not installed")
    def test_outputs_and_no_target_access(self):
        self.settings.raw_data_dir.mkdir()
        rows = [[unit, cycle, 0, 0, 100, *[i + cycle * .01 for i in range(21)]]
                for unit in [1, 2] for cycle in range(1, 5)]
        self.settings.train_path.write_text("\n".join(" ".join(map(str, r)) for r in rows))
        # Neither test trajectories nor official target files exist.
        with patch("aevora.data.loader.load_rul_targets", side_effect=AssertionError("targets accessed")):
            output = run_eda(self.settings)
        report = json.loads((output / "quality_report.json").read_text())
        self.assertEqual(report["rows"], 8)
        self.assertEqual(report["unique_engines"], 2)
        self.assertEqual(len(list(output.glob("*.png"))), 4)
        self.assertEqual(len(list(output.glob("*.csv"))), 3)
        self.assertTrue((output / "findings.md").exists())
        for png in output.glob("*.png"):
            self.assertTrue(png.read_bytes().startswith(b"\x89PNG"))
