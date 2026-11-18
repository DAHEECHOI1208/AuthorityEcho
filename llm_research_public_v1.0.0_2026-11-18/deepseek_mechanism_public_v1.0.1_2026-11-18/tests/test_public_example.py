"""Benign public example uses existing input and metadata contracts."""
import json
from pathlib import Path
import tempfile
import unittest

from mechanism.inputs import load_cases, prepare, rows_for_case
from mechanism.io import ROOT, load_run


class PublicExampleTest(unittest.TestCase):
    def test_six_conditions_and_explicit_benign_metadata(self):
        case = json.loads((ROOT / "input.json").read_text())
        corpus = json.loads((ROOT / "examples/benign_corpus.json").read_text())
        self.assertEqual(corpus["cases"][0]["input"], case)
        rows = load_cases(ROOT / "examples/benign_corpus.json", corpus=True)
        self.assertEqual(len(rows), 6)
        self.assertEqual({r["variant"] for r in rows}, {"benign"})
        by_c = {r["condition_id"]: r for r in rows}
        self.assertEqual(by_c["C0"]["persona"], "")
        self.assertEqual(by_c["C0"]["document"], "")
        self.assertEqual(by_c["C2"]["document"], by_c["C4"]["document"])
        self.assertEqual(by_c["C2"]["persona"], "")
        self.assertEqual(by_c["C3-E"]["persona"], by_c["C4"]["persona"])
        self.assertIn("Organization B", by_c["C3-E"]["document"])
        self.assertIn("Organization A", by_c["C3-P"]["document"])
        self.assertIn("library-inventory", by_c["C3-P"]["document"])
        self.assertIn("adult book-club", by_c["C4"]["document"])

    def test_benign_baseline_prepare_needs_no_private_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            prepare(ROOT / "examples/benign_corpus.json", run, corpus=True, study="baseline")
            manifest, rows = load_run(run)
            self.assertEqual(manifest["study"], "baseline")
            self.assertEqual(len(rows), 6)
            self.assertEqual({r["variant"] for r in rows}, {"benign"})
            self.assertFalse((run / "centers.npz").exists())


if __name__ == "__main__":
    unittest.main()
