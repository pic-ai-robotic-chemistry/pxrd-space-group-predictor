"""Low-energy window and StructureMatcher deduplication for one space-group pool."""
from __future__ import annotations

from src.runtime.pxrd_peak_match import ENERGY_WINDOW_KJ_MOL

DENSITY_TOLERANCE_G_CM3 = 0.2


def low_energy_pool(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Keep candidates within the energy window of this pool, then deduplicate them.

    Rows are compared in increasing energy (ties by ID); a candidate joins the
    first lower-energy representative within the density tolerance that
    StructureMatcher accepts. Returns representatives with ``relative_energy``
    and the candidate-to-representative mapping.
    """
    from pymatgen.analysis.structure_matcher import ElementComparator, StructureMatcher
    from pymatgen.io.cif import CifParser
    if len({int(r["space_group"]) for r in rows}) != 1:
        raise ValueError("a pool must contain exactly one space group")
    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5,
        primitive_cell=False, scale=True, attempt_supercell=False,
        comparator=ElementComparator(), ignored_species=["H"])
    minimum = min(float(r["energy_per_molecule"]) for r in rows)
    window = sorted((dict(r, relative_energy=float(r["energy_per_molecule"]) - minimum) for r in rows
                     if float(r["energy_per_molecule"]) - minimum <= ENERGY_WINDOW_KJ_MOL + 1e-12),
                    key=lambda r: (r["relative_energy"], r["candidate_id"]))
    retained, mapping = [], []
    for row in window:
        structure = CifParser(row["cif_path"], frac_tolerance=0).parse_structures(primitive=False)[0]
        representative = next((rep for rep, parsed in retained
                               if abs(parsed.density - structure.density) <= DENSITY_TOLERANCE_G_CM3
                               and matcher.fit(parsed, structure)), None)
        if representative is None:
            representative = row
            retained.append((row, structure))
        mapping.append({"candidate_id": row["candidate_id"],
                        "representative_candidate_id": representative["candidate_id"]})
    return [row for row, _ in retained], mapping
