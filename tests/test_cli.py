"""CLI orchestration contracts, including failure and resume boundaries."""
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from src.cli import parser
from src.runtime.workflow import Run, check_checkpoint, choose_groups, experimental_inputs, execute_stages, invoke, pool_groups


class CliTests(unittest.TestCase):
    def test_run_requires_explicit_budget_and_gpu(self):
        base = ["run", "--pxrd", "observed.xy", "--smiles", "CCO", "--wavelength", "1.54184",
                "--checkpoint", "model.pt", "--checkpoint-sha256", "0" * 64,
                "--chemberta", "chemberta", "--bomlip-root", "engine", "--mace-model", "mace.model", "--output", "run"]
        with self.assertRaises(SystemExit):
            parser().parse_args(base + ["--gpu", "0"])
        with self.assertRaises(SystemExit):
            parser().parse_args(base + ["--count", "1"])
        self.assertEqual(parser().parse_args(base + ["--gpu", "0", "--count", "1"]).count, 1)

    def test_resume_rejects_changed_identity_and_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            def work(p):
                p.mkdir(); (p / "result").write_text("complete")
            run = Run(root, {"input": "a"})
            out = run.stage("one", work)
            run.lock.close()
            resumed = Run(root, {"input": "a"}, True)
            self.assertEqual(resumed.stage("one", lambda _: self.fail("should reuse")), out)
            (out / "result").write_text("changed")
            with self.assertRaises(ValueError): resumed.stage("one", work)
            resumed.lock.close()
            with self.assertRaises(ValueError): Run(root, {"input": "b"}, True)
            with self.assertRaises(ValueError): Run(root, {"input": "a"})

    def test_retry_preserves_partial_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            run = Run(root, {})
            def fail(p):
                p.mkdir(); (p / "partial").write_text("evidence"); raise RuntimeError("interrupted")
            with self.assertRaises(RuntimeError): run.stage("one", fail)
            self.assertEqual(json.loads((root / "run.json").read_text())["status"], "failed")
            def success(p):
                p.mkdir(); (p / "result").write_text("complete")
            second = run.stage("one", success)
            self.assertEqual(second.name, "one-002")
            self.assertTrue((root / "one-001/partial").exists())
            run.lock.close()

    def test_concurrent_run_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            run = Run(root, {})
            with self.assertRaises(ValueError): Run(root, {}, True)
            run.lock.close()

    def test_others_is_not_automatically_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "predictions.csv"
            p.write_text('sample_id,predicted_space_groups\none,14;Others;2\n')
            with self.assertRaisesRegex(ValueError, "Others"): choose_groups(p)
            self.assertEqual(choose_groups(p, [14, 2]), [14, 2])
            with self.assertRaises(ValueError): choose_groups(p, [14, 14])

    def test_failure_between_stages_is_recorded_and_unlocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            with self.assertRaisesRegex(ValueError, "shortlist"):
                with Run(root, {}) as run:
                    raise ValueError("invalid shortlist")
            self.assertEqual(json.loads((root / "run.json").read_text())["status"], "failed")
            with Run(root, {}, True):
                pass

    def test_engine_launch_does_not_prepend_predictor_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            with Run(Path(tmp) / "run", {}) as run:
                captured = []
                with patch("src.runtime.workflow.run_logged", side_effect=lambda command, **kwargs: captured.append((command, kwargs))):
                    invoke(run, "tools.generate_crystal_candidates", ["--help"], run.root / "generate", python=sys.executable)
                command, options = captured[0]
                result = subprocess.run(command, cwd=run.root, env=options["env"], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--space-groups", result.stdout)
                self.assertIn("sys.path.append", command[2])

    def test_stale_installed_adapters_do_not_shadow_running_code(self):
        # An older wheel left in the interpreter (or on PYTHONPATH) once ran the
        # previous scoring tool from a checkout run.
        with tempfile.TemporaryDirectory() as tmp:
            stale = Path(tmp) / "stale"
            for package in ("src", "tools"):
                (stale / package).mkdir(parents=True)
                (stale / package / "__init__.py").write_text("")
            (stale / "tools" / "score_pxrd_candidates.py").write_text("print('STALE')")
            with Run(Path(tmp) / "run", {}) as run:
                captured = []
                with patch("src.runtime.workflow.run_logged", side_effect=lambda command, **kwargs: captured.append((command, kwargs))):
                    invoke(run, "tools.score_pxrd_candidates", ["--help"], run.root / "score")
                command, options = captured[0]
                env = dict(options["env"], PYTHONPATH=str(stale))
                result = subprocess.run(command, cwd=run.root, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--experimental", result.stdout)
        self.assertNotIn("STALE", result.stdout)

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux process-lifecycle regression")
    def test_timeout_stops_nested_engine_process(self):
        from src.runtime.downstream_execution import run_logged
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            engine = "import os,time; from pathlib import Path; Path('engine.pid').write_text(str(os.getpid())); time.sleep(60)"
            inner = ("import os,sys; from pathlib import Path; "
                     "from src.runtime.downstream_execution import run_logged; "
                     f"run_logged([sys.executable,'-c',{engine!r}],cwd=Path.cwd(),env=os.environ.copy(),log=Path('inner.log'),timeout=60)")
            env = os.environ.copy()
            env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
            with self.assertRaises(subprocess.TimeoutExpired):
                run_logged([sys.executable, "-c", inner], cwd=root, env=env, log=root / "outer.log", timeout=2)
            pid = int((root / "engine.pid").read_text())
            try:
                self.assertFalse(Path(f"/proc/{pid}").exists(), "timed-out workflow left its engine alive")
            finally:
                if Path(f"/proc/{pid}").exists():
                    os.kill(pid, 15)

    def test_peak_match_input_is_min_subtracted_on_measured_grid(self):
        from src.data.pxrd import preprocess_experimental
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "observed.xy"
            theta = np.arange(10.005, 50.001, .01)
            intensity = 7 + 50 * np.exp(-((theta - 20) / .1) ** 2)
            intensity[1:3] = -3  # brackets the 10.02 grid point; clipped to zero before the minimum
            np.savetxt(p, np.column_stack([theta, intensity]))
            prepared, measured = experimental_inputs(p)
            np.testing.assert_array_equal(prepared, preprocess_experimental(theta, intensity))
            # 0.02 degree grid restricted to the measured range, not the input spacing.
            self.assertAlmostEqual(measured[0, 0], 10.02)
            self.assertAlmostEqual(measured[-1, 0], 49.98)  # input ends at 49.995
            np.testing.assert_allclose(np.diff(measured[:, 0]), 0.02, atol=1e-9)
            expected = np.interp(measured[:, 0], theta, np.clip(intensity, 0, None))
            np.testing.assert_allclose(measured[:, 1], (expected - expected.min()) / (expected.max() - expected.min()) * 100)
            self.assertAlmostEqual(measured[:, 1].max(), 100)
            self.assertGreater(measured[:, 1].min(), -1e-12)

    def test_zero_signal_and_wrong_model_hash_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.xy"
            np.savetxt(p, [[10, 0], [20, 0]])
            with self.assertRaises(ValueError): experimental_inputs(p)
            with self.assertRaises(ValueError): check_checkpoint(p, "0" * 64)

    def test_rank_order_must_be_unique_and_present(self):
        rows = [{"space_group": "14"}, {"space_group": "2"}]
        self.assertEqual(pool_groups(rows, [2, 14]), [2, 14])
        with self.assertRaisesRegex(ValueError, "no candidates"): pool_groups(rows, [2, 19])
        with self.assertRaisesRegex(ValueError, "unique"): pool_groups(rows, [2, 2])

    def test_search_stops_at_first_matching_space_group(self):
        from argparse import Namespace
        from src.runtime.crystal_candidates import candidate_row, write_candidates
        signals = {14: "CONTINUE_NEXT_SPACE_GROUP", 2: "STOP_SUCCESS", 19: "STOP_SUCCESS"}
        called = []
        def fake(run, module, arguments, directory, **_):
            called.append((module, directory.name))
            directory.mkdir()
            if module == "tools.process_crystal_candidates":
                (directory / "representatives.csv").write_text("placeholder")
            else:
                group = int(directory.name[2:5])
                (directory / "decision.json").write_text(json.dumps(
                    {"signal": signals[group], "top_candidate": {"B": 0.5}}))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for group in (14, 2, 19):
                cif = root / f"c{group}.cif"
                cif.write_text("data_x")
                rows.append(candidate_row("s", group, cif, energy=0.0, status="completed"))
            write_candidates(root / "relaxed.csv", rows)
            args = Namespace(command="rank", manifest=root / "relaxed.csv", space_groups=[14, 2, 19],
                             workers=1, timeout=10)
            with Run(root / "run", {}) as run, patch("src.runtime.workflow.invoke", side_effect=fake):
                execute_stages(args, run, np.zeros(3400, dtype=np.float32), np.ones((5, 2)), None, None)
            state = json.loads((root / "run" / "run.json").read_text())
        self.assertEqual([e["space_group"] for e in state["searched"]], [14, 2])
        self.assertEqual(state["result"], "STOP_SUCCESS")
        self.assertFalse(any(name.startswith("sg019") for _, name in called))

if __name__ == "__main__":
    unittest.main()
