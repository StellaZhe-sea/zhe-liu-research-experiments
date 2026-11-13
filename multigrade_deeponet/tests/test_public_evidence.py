"""Dependency-free consistency check for the published pendulum comparison."""
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class PublishedPendulumEvidence(unittest.TestCase):
    def test_matched_budget_and_recorded_outcome(self):
        protocol = json.loads((ROOT / "evidence/pendulum_protocol.json").read_text(encoding="utf-8"))
        summary = json.loads((ROOT / "evidence/pendulum_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(protocol["total_updates"], sum(protocol["grade_steps"]))
        self.assertEqual(protocol["parameters"]["traditional"], protocol["parameters"]["multigrade"])
        self.assertEqual(summary["traditional"]["completed_updates"], summary["multigrade"]["completed_updates"])
        self.assertEqual(summary["traditional"]["parameters"], summary["multigrade"]["parameters"])
        self.assertLess(summary["multigrade"]["selected_metrics"]["test_mse"],
                        summary["traditional"]["selected_metrics"]["test_mse"])


if __name__ == "__main__":
    unittest.main()
