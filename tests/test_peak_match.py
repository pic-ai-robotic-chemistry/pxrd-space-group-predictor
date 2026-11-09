import importlib.util
import unittest
from unittest.mock import patch

import numpy as np


@unittest.skipUnless(importlib.util.find_spec("pymatgen"), "requires structural dependencies")
class PeakMatchTests(unittest.TestCase):
    def setUp(self):
        from src.runtime import pxrd_peak_match
        self.pm = pxrd_peak_match

    def test_tolerance_uses_pair_mean_angle(self):
        self.assertEqual(self.pm.tolerance_for_pair(5.0, 5.9), 1.00)
        self.assertAlmostEqual(self.pm.tolerance_for_pair(39.5, 40.5), 1.21)
        source = (np.array([40.0]), np.array([1.0]))
        inside = (np.array([41.2]), np.array([0.05]))
        outside = (np.array([41.25]), np.array([0.05]))
        self.assertTrue(self.pm.strong_guard(source, inside)["guard"])
        self.assertFalse(self.pm.strong_guard(source, outside)["guard"])

    def test_guard_is_bidirectional_with_floors(self):
        exp = (np.array([10.0, 20.0]), np.array([1.0, 0.69]))
        calc = (np.array([10.0]), np.array([1.0]))
        self.assertTrue(self.pm.strong_guard(exp, calc)["guard"])
        # A calculated strong peak absent from experiment fails the reverse direction.
        calc = (np.array([10.0, 30.0]), np.array([1.0, 0.7]))
        self.assertFalse(self.pm.strong_guard(exp, calc)["guard"])
        # A counterpart below 0.05 does not cover a source strong peak.
        calc = (np.array([10.0, 20.0]), np.array([1.0, 0.049]))
        self.assertFalse(self.pm.strong_guard((np.array([10.0, 20.0]), np.array([1.0, 0.7])), calc)["guard"])

    def test_score_uses_reported_weights(self):
        exp = (np.array([10.0, 20.0, 30.0]), np.array([1.0, 0.5, 0.2]))
        calc = (np.array([10.5, 20.0]), np.array([0.8, 0.5]))
        score = self.pm.score_components(exp, calc)
        expected_m = 3 * 8 ** (-20 / 30) * 0.2 ** 2 / (1 + 0.25 + 0.04 + 0.64 + 0.25)
        self.assertAlmostEqual(score["M"], expected_m)
        self.assertAlmostEqual(score["D"], 0.5 / (1.0 + 0.007 * 0.25) / 1.5)
        self.assertAlmostEqual(score["H"], 0.2 / 1.5)
        self.assertAlmostEqual(score["B"], np.exp(-(score["M"] + 0.3 * score["D"] + 0.3 * score["H"])))
        empty = self.pm.score_components(exp, (np.empty(0), np.empty(0)))
        self.assertEqual((empty["D"], empty["H"]), (1.0, 1.0))

    def _pool(self, restarts):
        def fake(task):
            cid, _, restart, *_ = task
            objective, b, guard = restarts[cid][restart]
            return {"candidate_id": cid, "restart": restart, "refine_objective": objective, "B": b,
                    "guard_guard": guard, "match": guard and b >= 0.80}
        candidates = [(cid, f"{cid}.cif", float(i)) for i, cid in enumerate(restarts)]
        with patch.object(self.pm, "score_restart", side_effect=fake):
            return self.pm.evaluate_pool({}, 2.0, 70.0, candidates)

    def test_restart_with_best_objective_is_kept_not_best_b(self):
        rows, match = self._pool({"a": [(0.5, 0.95, True), (0.9, 0.70, True), (0.1, 0.99, True)]})
        self.assertEqual((rows[0]["restart"], rows[0]["B"]), (1, 0.70))
        self.assertEqual((rows[0]["restart_B_min"], rows[0]["restart_B_max"]), (0.70, 0.99))
        self.assertIsNone(match)

    def test_only_top_b_candidate_decides(self):
        guard_failed = {"top": [(1, 0.95, False)] * 3, "second": [(1, 0.90, True)] * 3}
        rows, match = self._pool(guard_failed)
        self.assertEqual(rows[0]["candidate_id"], "top")
        self.assertIsNone(match)
        rows, match = self._pool({"top": [(1, 0.80, True)] * 3, "second": [(1, 0.79, True)] * 3})
        self.assertEqual(match["candidate_id"], "top")

    def test_seed_ignores_opt_suffix_and_directory(self):
        from pathlib import Path
        self.assertEqual(self.pm.refinement_seed(Path("/a/x_opt.cif"), 1), self.pm.refinement_seed(Path("x.cif"), 1))
        self.assertNotEqual(self.pm.refinement_seed(Path("x.cif"), 0), self.pm.refinement_seed(Path("x.cif"), 1))


if __name__ == "__main__":
    unittest.main()
