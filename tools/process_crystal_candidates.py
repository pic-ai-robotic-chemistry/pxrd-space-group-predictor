#!/usr/bin/env python3
"""Keep one space-group pool's low-energy window and deduplicate it."""
import argparse
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.runtime.crystal_candidates import read_candidates, write_candidates, sha256
from src.runtime.crystal_processing import DENSITY_TOLERANCE_G_CM3, low_energy_pool
from src.runtime.pxrd_peak_match import ENERGY_WINDOW_KJ_MOL


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True, help="Relaxed candidates of one space group")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    rows = read_candidates(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    reps, mapping = low_energy_pool(rows)
    write_candidates(args.output_dir / "representatives.csv", reps)
    with (args.output_dir / "deduplication.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(mapping[0]))
        writer.writeheader()
        writer.writerows(mapping)
    (args.output_dir / "provenance.json").write_text(json.dumps({
        "input_manifest_sha256": sha256(args.manifest), "input_count": len(rows),
        "space_group": int(rows[0]["space_group"]),
        "energy_window_kj_mol": ENERGY_WINDOW_KJ_MOL, "window_count": len(mapping),
        "density_tolerance_g_cm3": DENSITY_TOLERANCE_G_CM3,
        "representative_count": len(reps), "energy_unit": "kJ/mol per molecule",
        "step_limit_count": sum(r["status"] == "step_limit" for r in reps),
        "protocol": "docs/downstream-workflow.md"}, indent=2) + "\n")
    print(f"{len(mapping)} of {len(rows)} candidates within {ENERGY_WINDOW_KJ_MOL} kJ/mol; "
          f"{len(reps)} representatives after deduplication.")


if __name__ == "__main__":
    main()
