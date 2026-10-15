"""Source and rendering checks for plot_recorded_loss.py (stdlib unittest)."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import plot_recorded_loss as plot


class RecordedLossTests(unittest.TestCase):
    def test_receipt_retains_raw_windows_and_exact_token_clocks(self):
        receipt = json.loads(Path(__file__).with_name("e97_recorded_loss.json").read_text())
        self.assertEqual(receipt["tokens_per_workstation_step"], 65_536)
        self.assertEqual([len(w["step_loss"]) for w in receipt["windows"]], [869, 870, 838])
        self.assertEqual([w["sha256"] for w in receipt["windows"]], [
            "ccdb06b44c63324ae7f68cc03c934584e01d00cf7687ef67cae17323723241c8",
            "c9ad3b589bf98ef9f61e2871c8d92f27889c5e690e49fb246eac671e2797985b",
            "a749feabfcee9665af8281bda365324c62771f596c6f18f8ce971014ce03d521",
        ])
        for window in receipt["windows"]:
            points = window["step_loss"]
            self.assertTrue(all(b[0] - a[0] == 25 for a, b in zip(points, points[1:])))
            self.assertTrue(all(1.95 < loss < 3.05 for _, loss in points))
        self.assertEqual(len(receipt["workstation_checkpoints"]), 28)
        self.assertEqual(receipt["frontier_anchors"], [
            {"accepted_tokens": tokens, "loss": loss} for tokens, loss in plot.FRONTIER_ANCHORS])
        self.assertEqual(receipt["corpus_boundary_accepted_tokens"], 150_793_748_480)
        self.assertEqual(receipt["gdn2_reference"], {
            "step": 2_323_616, "accepted_tokens": 152_280_498_176, "loss": 2.4034})

    def test_window_rejects_checksum_drift_and_recorded_gaps(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "tail.txt"
            source.write_text("step 25 | loss 2.5 | lr 1e-3\nstep 75 | loss 2.4 | lr 1e-3\n")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                plot.read_window(source, "0" * 64)
            with self.assertRaisesRegex(ValueError, "contiguous per-25-step"):
                plot.read_window(source, digest)

    def test_checkpoint_parser_never_opens_payloads(self):
        receipt = json.loads(Path(__file__).with_name("e97_recorded_loss.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for anchor in receipt["workstation_checkpoints"]:
                original = Path(anchor["path"])
                (root / original.parent.name).mkdir(exist_ok=True)
                # Even directories named .pt suffice: only names are consulted.
                (root / original.parent.name / original.name).mkdir()
            with patch.object(Path, "open", side_effect=AssertionError("Payload access")):
                anchors = plot.read_checkpoints(root)
            self.assertEqual([a["loss"] for a in anchors],
                             [a["loss"] for a in receipt["workstation_checkpoints"]])

    def test_only_raw_window_series_are_connected(self):
        receipt = json.loads(Path(__file__).with_name("e97_recorded_loss.json").read_text())
        from matplotlib.axes import Axes
        calls = []
        original_plot = Axes.plot

        def record(ax, x, y, *args, **kwargs):
            calls.append((list(x), list(y)))
            return original_plot(ax, x, y, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory, patch.object(Axes, "plot", record):
            plot.plot_record(copy.deepcopy(receipt), Path(directory) / "figure.png")
        expected = []
        for window in receipt["windows"]:
            series = ([s * 65_536 / 1e9 for s, _ in window["step_loss"]],
                      [loss for _, loss in window["step_loss"]])
            expected.extend([series, series])  # detail and full-record panels
        self.assertEqual(calls, expected)


if __name__ == "__main__":
    unittest.main()
