#!/usr/bin/env python3
"""Score one deduplicated space-group pool against experimental PXRD and decide whether to stop."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.runtime.crystal_candidates import read_candidates
from src.runtime import pxrd_peak_match as pm


def run(args: argparse.Namespace) -> dict:
    reps = read_candidates(args.manifest)
    if len({r["space_group"] for r in reps}) != 1:
        raise ValueError("score one space-group pool at a time")
    # The pool minimum is always a representative, so this equals the pool-relative energy.
    minimum = min(float(r["energy_per_molecule"]) for r in reps)
    candidates = [(r["candidate_id"], r["cif_path"], float(r["energy_per_molecule"]) - minimum) for r in reps]
    exp_levels, xlo, xhi = pm.experimental_peaks(pm.load_experimental_pxrd(args.experimental))
    rows, match = pm.evaluate_pool(exp_levels, xlo, xhi, candidates, workers=args.workers)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "scores.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["rank", *rows[0]])
        writer.writeheader()
        writer.writerows({"rank": rank, **row} for rank, row in enumerate(rows, 1))
    top = rows[0]
    decision = {
        "space_group": int(reps[0]["space_group"]),
        "signal": "STOP_SUCCESS" if match else "CONTINUE_NEXT_SPACE_GROUP",
        "reason": "TOP_B_PASSED" if match else
                  "TOP_B_FAILED_HARD_GUARD" if not top["guard_guard"] else "TOP_B_BELOW_THRESHOLD",
        "top_candidate": {key: top[key] for key in ("candidate_id", "relative_energy_kj_mol", "B", "M", "D", "H",
                                                    "guard_guard", "restart", "zero_shift")},
        "candidate_count": len(rows),
        "settings": {"energy_window_kj_mol": pm.ENERGY_WINDOW_KJ_MOL, "threshold": pm.SCORE_THRESHOLD,
                     "weights_MDH": [pm.WEIGHT_M, pm.WEIGHT_D, pm.WEIGHT_H],
                     "source_floor": pm.SOURCE_FLOOR, "counterpart_floor": pm.COUNTERPART_FLOOR,
                     "restarts": pm.RESTARTS, "maxiter": pm.MAXITER, "popsize": pm.POPSIZE},
        "refined_cif": None,
    }
    if match:
        from pymatgen.io.cif import CifWriter
        source = Path(next(path for cid, path, _ in candidates if cid == match["candidate_id"]))
        cell = tuple(match[key] for key in ("a", "b", "c", "alpha", "beta", "gamma"))
        CifWriter(pm.structure_with_lattice(pm.load_structure(source), cell)).write_file(
            args.output_dir / "selected_refined.cif")
        decision["refined_cif"] = "selected_refined.cif"
    (args.output_dir / "decision.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print(f"SG{decision['space_group']}: {decision['signal']} (top B={top['B']:.4f})")
    return decision


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experimental", type=Path, required=True,
                        help="Two columns: degrees 2theta, linear intensity (measured, not background-corrected)")
    parser.add_argument("--manifest", type=Path, required=True, help="representatives.csv of one space-group pool")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1, help="Processes; results do not depend on this")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
