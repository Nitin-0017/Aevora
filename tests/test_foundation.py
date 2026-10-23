"""Foundation tests: synthetic configuration only, no simulated dataset records."""

from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest

from aevora import __version__
from aevora.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/default.toml"


class FoundationTests(unittest.TestCase):
    def test_version(self):
        self.assertEqual(__version__, "0.1.0")

    def test_defaults_and_official_file_names(self):
        settings = load_config(CONFIG)
        self.assertEqual(settings.project_root, ROOT)
        self.assertEqual(settings.random_seed, 42)
        self.assertEqual(settings.sequence_window, 30)
        self.assertEqual(settings.train_path.name, "train_FD001.txt")
        self.assertEqual(settings.test_path.name, "test_FD001.txt")
        self.assertEqual(settings.test_targets_path.name, "RUL_FD001.txt")
        self.assertEqual(settings.models["lstm"]["device"], "cpu")

    def test_invalid_settings(self):
        cases = [
            ('dataset = "FD001"', 'dataset = "FD002"'),
            ('sequence_window = 30', 'sequence_window = 0'),
            ('random_seed = 42', 'random_seed = true'),
            ('validation_fraction = 0.2', 'validation_fraction = 1.0'),
            ('validation_fraction = 0.2', 'validation_fraction = nan'),
            ('alpha = 1.0', 'alpha = -1.0'),
            ('max_iter = 200', 'max_iter = 1.5'),
            ('max_leaf_nodes = 15', 'max_leaf_nodes = 1'),
            ('device = "cpu"', 'device = "cuda"'),
            ('learning_rate = 0.001', 'learning_rate = 0'),
            ('hidden_size = 32', 'hidden_size = 0'),
            ('sequence_window = 30', 'window_typo = 30'),
        ]
        for old, new in cases:
            with self.subTest(setting=new), tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "invalid.toml"
                path.write_text(CONFIG.read_text().replace(old, new))
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_loading_has_no_filesystem_side_effects(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "settings.toml"
            path.write_text(CONFIG.read_text().replace('project_root = ".."', 'project_root = "project"'))
            settings = load_config(path)
            self.assertEqual(settings.project_root, (Path(temp) / "project").resolve())
            self.assertFalse(settings.project_root.exists())
            self.assertEqual(list(Path(temp).iterdir()), [path])

    def test_cli_from_another_working_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
            result = subprocess.run(
                [sys.executable, "-m", "aevora", "--config", str(CONFIG)],
                cwd=temp, env=env, capture_output=True, text=True, check=True,
            )
            self.assertIn("Configuration valid", result.stdout)
            self.assertIn(str(ROOT / "data/raw/CMAPSSData/train_FD001.txt"), result.stdout)

    def test_missing_config_reports_error(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileNotFoundError):
                load_config(Path(temp) / "absent.toml")


if __name__ == "__main__":
    unittest.main()
