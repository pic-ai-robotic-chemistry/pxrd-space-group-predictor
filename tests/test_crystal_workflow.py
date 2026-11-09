"""Identity guards and scientific downstream behavior on synthetic structures."""
import importlib.util
from pathlib import Path
import tempfile
import json
import os
import subprocess
import sys
from unittest.mock import patch
import unittest
import numpy as np
from src.runtime.crystal_candidates import candidate_row, read_candidates, write_candidates
from src.runtime.crystal_processing import low_energy_pool


class ManifestTests(unittest.TestCase):
    def test_hash_units_and_duplicate_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cif = root / "example.cif"
            cif.write_text("synthetic")
            row = candidate_row("test", 1, cif, energy=-10, status="completed")
            path = root / "manifest.csv"
            write_candidates(path, [row])
            self.assertEqual(read_candidates(path)[0]["energy_unit"], "kJ/mol")
            duplicate = root / "duplicate.csv"
            write_candidates(duplicate, [row, row])
            with self.assertRaisesRegex(ValueError, "unique"):
                read_candidates(duplicate)
            bad_unit = root / "unit.csv"
            write_candidates(bad_unit, [dict(row, energy_unit="eV")])
            with self.assertRaisesRegex(ValueError, "kJ/mol"):
                read_candidates(bad_unit)
            cif.write_text("changed")
            with self.assertRaisesRegex(ValueError, "hash"):
                read_candidates(path)


class ExecutionTests(unittest.TestCase):
    def test_normalization_rejects_wrong_molecule_size_and_stale_manifest(self):
        from tools.relax_crystal_candidates import molecular_atom_count
        from src.runtime.crystal_candidates import sha256
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, metadata = root / "candidates.csv", root / "generation.json"
            manifest.write_text("synthetic manifest")
            metadata.write_text(json.dumps(dict(atoms_per_molecule=9, sample_id="ethanol", manifest_sha256=sha256(manifest))))
            self.assertEqual(molecular_atom_count(manifest, metadata, "ethanol"), 9)
            with self.assertRaisesRegex(ValueError, "differs"):
                molecular_atom_count(manifest, metadata, "ethanol", 3)
            manifest.write_text("changed")
            with self.assertRaisesRegex(ValueError, "does not match"):
                molecular_atom_count(manifest, metadata, "ethanol")

    def test_external_failure_and_timeout_do_not_report_success(self):
        from src.runtime.downstream_execution import run_logged
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "exited 7"):
                run_logged([sys.executable, "-c", "raise SystemExit(7)"], cwd=root, env=os.environ.copy(), log=root / "fail.log", timeout=5)
            with self.assertRaises(subprocess.TimeoutExpired):
                run_logged([sys.executable, "-c", "import time; time.sleep(30)"], cwd=root, env=os.environ.copy(), log=root / "timeout.log", timeout=.1)

    def test_packing_requires_separate_ccdc_api(self):
        from tools.validate_crystal_packing import _load_ccdc
        with patch.dict(sys.modules, {"ccdc": None, "ccdc.crystal": None, "ccdc.io": None}):
            with self.assertRaisesRegex(RuntimeError, "separate CSD license"):
                _load_ccdc()

    def test_packing_timeout_is_unresolved_and_fails_command(self):
        import csv
        import types
        from tools.validate_crystal_packing import main
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cif = root / "structure.cif"
            cif.write_text("synthetic")
            manifest = root / "manifest.csv"
            write_candidates(manifest, [candidate_row("test", 1, cif, energy=-1, status="completed")])
            fake = types.ModuleType("ccdc.crystal")
            fake.PackingSimilarity = lambda: object()
            fake_io = types.ModuleType("ccdc.io")
            fake_io.CrystalReader = object
            argv = ["packing", "--manifest", str(manifest), "--reference", str(cif), "--output-dir", str(root / "output")]
            with patch.dict(sys.modules, {"ccdc": types.ModuleType("ccdc"), "ccdc.crystal": fake, "ccdc.io": fake_io}), patch.object(sys, "argv", argv), patch("tools.validate_crystal_packing.subprocess.run", side_effect=subprocess.TimeoutExpired("packing", 30)):
                with self.assertRaises(SystemExit) as error:
                    main()
            self.assertEqual(error.exception.code, 1)
            with (root / "output/packing.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["status"], "timeout")
            self.assertEqual(rows[0]["hit"], "")

    def test_snapshot_rejects_dirty_or_unpinned_upstream(self):
        from src.runtime.downstream_execution import snapshot, BOMLIP_COMMIT
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("subprocess.check_output", side_effect=["wrong\n", ""]):
                with self.assertRaisesRegex(ValueError, "clean checkout"):
                    snapshot(root, root / "snapshot")
            with patch("subprocess.check_output", side_effect=[BOMLIP_COMMIT, " M source"]):
                with self.assertRaisesRegex(ValueError, "clean checkout"):
                    snapshot(root, root / "snapshot")
            self.assertFalse((root / "snapshot").exists())

    def test_model_patch_is_confined_and_checks_call_layout(self):
        from src.runtime.downstream_execution import pin_mace_model, MACE_SHA256
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            engine = root / "mace-bench/src/batchopt/relaxengine.py"
            engine.parent.mkdir(parents=True)
            engine.write_text('import os\n' + 'mace_off(model="small")\n' * 3)
            with patch("src.runtime.downstream_execution.sha256", return_value=MACE_SHA256):
                result = pin_mace_model(root, root / "model")
                self.assertEqual(result["replacement_count"], 3)
                self.assertEqual(engine.read_text().count('model=os.environ["PXRD_MACE_MODEL"]'), 3)
                with self.assertRaisesRegex(ValueError, "layout"):
                    pin_mace_model(root, root / "model")

    def test_model_mismatch_and_unexpected_upstream_are_rejected(self):
        from src.runtime.downstream_execution import pin_mace_model
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "wrong.model"
            model.write_text("not a model")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                pin_mace_model(root, model)


@unittest.skipUnless(importlib.util.find_spec("pymatgen"), "requires structural dependencies")
class StructureTests(unittest.TestCase):
    def test_pool_window_dedup_and_density_prefilter(self):
        from pymatgen.core import Structure, Lattice
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for name, edge, energy in [("high", 5.6, -1.0), ("low", 5.6, -2.0), ("dense", 5.0, -1.5),
                                       ("edge", 5.6, 1.0), ("outside", 5.6, 1.01)]:
                cif = root / f"{name}.cif"
                Structure(Lattice.cubic(edge), ["Na", "Cl"], [[0, 0, 0], [.5, .5, .5]]).to(filename=str(cif))
                rows.append(candidate_row("synthetic", 1, cif, energy=energy, status="completed"))
            reps, mapping = low_energy_pool(rows)
            # 3.01 kJ/mol above the minimum is outside the window; 3.00 is inside.
            self.assertEqual([m["candidate_id"] for m in mapping], ["sg1:low", "sg1:dense", "sg1:high", "sg1:edge"])
            # StructureMatcher scales volumes, so only the density prefilter keeps "dense" separate.
            self.assertEqual([r["candidate_id"] for r in reps], ["sg1:low", "sg1:dense"])
            self.assertEqual(reps[0]["relative_energy"], 0.0)
            self.assertEqual({m["representative_candidate_id"] for m in mapping[2:]}, {"sg1:low"})
            with self.assertRaisesRegex(ValueError, "one space group"):
                low_energy_pool(rows + [dict(rows[0], space_group="2", candidate_id="x")])


@unittest.skipUnless(importlib.util.find_spec("rdkit"), "requires RDKit")
class ConformerTests(unittest.TestCase):
    def test_explicit_hydrogens_and_selected_conformer(self):
        from tools.generate_crystal_candidates import prepare_conformer
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "molecule.xyz"
            result = prepare_conformer("CCO", output)
            self.assertEqual(result["atoms_per_molecule"], 9)
            self.assertEqual(int(output.read_text().splitlines()[0]), 9)
            self.assertEqual(result["selected_conformer_id"], result["ranked_uff_status_energy_id"][0][2])
            with self.assertRaisesRegex(ValueError, "single-component"):
                prepare_conformer("CC.O", Path(tmp) / "bad.xyz")


if __name__ == "__main__":
    unittest.main()
