import unittest
from pathlib import Path
import tempfile
import pandas as pd
from aevora.data.labels import training_rul
from aevora.data.validation import COLUMNS
from aevora.config import load_config


def histories(lengths=(3, 2)):
    return pd.DataFrame([[u, c, 0, 0, 100, *[c + s * u for s in range(21)]]
                         for u, n in enumerate(lengths, 1) for c in range(1, n + 1)], columns=COLUMNS)


class LabelTests(unittest.TestCase):
    def test_lifetimes_alignment_and_caps(self):
        frame = histories().iloc[[3, 0, 4, 1, 2]].copy()
        frame.index = [9, 9, 3, 5, 8]
        before = frame.copy()
        labels = training_rul(frame, run_to_failure=True, cap=1)
        self.assertEqual(labels.rul_uncapped.tolist(), [1, 2, 0, 1, 0])
        self.assertEqual(labels.rul.tolist(), [1, 1, 0, 1, 0])
        pd.testing.assert_frame_equal(labels[["unit_id", "cycle"]], frame[["unit_id", "cycle"]])
        pd.testing.assert_frame_equal(frame, before)
        uncapped = training_rul(frame, run_to_failure=True)
        self.assertEqual(uncapped.rul.tolist(), uncapped.rul_uncapped.tolist())

    def test_invalid_inputs_and_attestation(self):
        for cap in [0, -1, 2.5, True]:
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                training_rul(histories(), run_to_failure=True, cap=cap)
        with self.assertRaises(ValueError):
            training_rul(histories(), run_to_failure=False)
        bad = histories()
        bad.loc[0, "cycle"] = 0
        with self.assertRaises(ValueError):
            training_rul(bad, run_to_failure=True)

    def test_optional_cap_configuration(self):
        text = (Path(__file__).resolve().parents[1] / "configs/default.toml").read_text()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.toml"
            path.write_text(text)
            self.assertIsNone(load_config(path).rul_cap)
            path.write_text(text.replace("# rul_cap = 125", "rul_cap = 125"))
            self.assertEqual(load_config(path).rul_cap, 125)
            path.write_text(text.replace("# rul_cap = 125", "rul_cap = 0"))
            with self.assertRaises(ValueError):
                load_config(path)
