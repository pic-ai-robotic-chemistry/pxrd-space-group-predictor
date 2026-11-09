# Downstream crystal workflow

These adapters connect the existing prediction and PXRD peak-matching tools to
BOMLIP-CSP, MACE relaxation, within-space-group deduplication and optional licensed packing
validation. They preserve the reported numerical settings. They do not supply
trained weights, licensed structures or the historical calculation environment.
Protocol agreement does not imply bitwise reproduction of historical results.

## Environment and inputs

For the primary workflow, follow [the tested Conda installation](../environments/README.md)
and [the CLI guide](cli.md). Keep prediction and engine environments separate;
pass the engine interpreter through `--engine-python`.

The per-stage Python commands below are advanced source-checkout interfaces;
`pxrd-sg run` and `pxrd-sg rank` call them once per space group in search order
and stop at the first match. Run generation and relaxation with the engine
Python, and processing/matching with the prediction Python. Consult each
command's `--help` for parameters.

Obtain a **clean** clone of
https://github.com/pic-ai-robotic-chemistry/BOMLIP-CSP at commit
`2caf530c8aa2ca14230264bfe69bfbcd6952b41e`. Both generation and relaxation make
tracked-file snapshots in new output directories. They reject a dirty or different
checkout. Original checkouts remain unchanged. MACE's model path is changed only
inside the snapshot; three hardcoded model calls are replaced with an environment
lookup, with before/after hashes recorded. Upstream MIT licensing remains upstream.

Supply a trusted local MACE-OFF23-small model with SHA256
`165cce4cfec5a34b9c64d4ebf95de15d71106bb584b7291c8470f0749977c46f`.
The adapter checks this hash before loading. Download and verify the model using
[the model asset guide](model-weights.md#mace-for-full-runs). There is no automatic
model download during execution.
Generation and MACE runs require explicit compute authorization from the operator;
a prediction request alone must not launch them. Choose available hardware first.

Use single-component molecular SMILES and explicit concrete space-group numbers.
Confirm the shortlist with the user before a costly search. `Others` is an
aggregate prediction class and cannot be passed as a space group. The retrospective
paper cases were selected with known shortlist coverage; new predictions do not
guarantee that coverage.

## Generate candidates

```bash
python tools/generate_crystal_candidates.py --smiles CCO --sample-id ethanol --space-groups 1 --bomlip-root /path/to/BOMLIP-CSP --output-dir runs/task/sg001-generated --count 5000 --workers 8
```

ETKDGv3 requests 20 explicit-hydrogen conformers, with seed 240430 and RMS pruning
0.1 Å. UFF optimizes them all. Converged conformers come first, followed by lower
UFF energy and lower conformer identifier. The first conformer is passed to
BOMLIP-CSP without further conformer generation, with Z′ = 1. Generation starts
at seed 1 in each group. Each group requests 5,000 successful structures; a smaller
`--count` is an execution smoke test, not the paper candidate budget. The timeout
bounds each group's retries. Failed or partial runs are not complete manifests.

`generation.json` records atom count, commands, settings and final manifest hash.
`candidates.csv` contains exactly one sample, unique string IDs across groups,
requested space groups, CIF paths and SHA256 hashes, blank energies and generated
status. Space-group fields describe the generation pool, not independently
redetermined final symmetry. Keep output directories intact when reusing manifests.

## Two-stage MACE relaxation

```bash
python tools/relax_crystal_candidates.py --manifest runs/task/sg001-generated/candidates.csv --bomlip-root /path/to/BOMLIP-CSP --model /path/to/MACE-OFF23_small.model --output-dir runs/task/sg001-relaxed --gpu 0 --workers 8
```

The atom count per molecule comes from the hash-linked `generation.json` beside
the manifest. `--generation-metadata` selects a relocated file. An optional
`--atoms-per-molecule` is an assertion and must agree; a divisor of the cell's
atom count alone is not sufficient to define molecular energy normalization.
The manifest sample, metadata hash link and cell divisibility are checked.

Stage 1 uses UnitCellFilter/BFGSFusedLS at 0.0006 eV Å⁻³; stage 2 uses
UnitCellFilter/BFGS at zero pressure. Each stage uses fmax 0.01 eV Å⁻¹ and at
most 3,000 steps. Batch size is zero, with two CPU threads per worker and
cuEquivariance enabled. Defaults use eight workers on one explicitly selected GPU;
change workers to fit hardware, recording the change. Upstream optimizer equations
and stopping rules remain unchanged.

`relaxed.csv` preserves candidate identities and records **kJ/mol per molecule**,
not eV per cell. Upstream JSON energies already have this conversion. A stage at
or near the step limit produces `step_limit`; other outputs are `completed`.
These flags do not independently certify force convergence. Inspect stage logs
and JSON outputs before scientific interpretation. Preserve flagged candidates
as in the reported screening protocol; report their count instead of silently
claiming all structures converged. A missing/invalid output fails the command.

## Low-energy window and deduplication

```bash
python tools/process_crystal_candidates.py --manifest runs/task/sg001-relaxed/relaxed.csv --output-dir runs/task/sg001-processed
```

The manifest must contain exactly one space group. Energies are taken relative
to that group's minimum; candidates with ΔE ≤ 3.0 kJ/mol are kept and the rest
are not processed further. The window is sorted by energy then string ID and
deduplicated with StructureMatcher (ltol 0.2, stol 0.3, angle tolerance 5°,
ElementComparator, ignored H, primitive_cell=False, scale=True,
attempt_supercell=False). A structure is compared only with retained
representatives whose density differs by at most 0.2 g/cm³ and joins the first
that matches, so the lowest-energy member represents each cluster. There is no
transitive closure or cross-group merging. `representatives.csv` (with
`relative_energy`), `deduplication.csv` and `provenance.json` record the result;
no PXRD arrays are exported.

## PXRD peak matching and stopping

```bash
python tools/score_pxrd_candidates.py --experimental observed.xy --manifest runs/task/sg001-processed/representatives.csv --output-dir runs/task/sg001-matched --workers 8
```

Supply the raw measured two-column pattern, without the classification
background filter. It is clipped at zero, interpolated onto the 0.02° 2θ grid
within its measured range, minimum-subtracted and scaled to a maximum of 100. Candidate sticks are simulated for Cu Kα
(1.54184 Å), broadened by pseudo-Voigt (FWHM 0.15°, η 0.5) and processed with
the experimental pattern by Gaussian smoothing (σ 0.075°), SNIP background
removal (1.2°) on a log(1+√I) scale, a second smoothing (σ 0.03375°) and max-1
normalisation. Major, medium and fine peak sets are extracted. Each candidate
cell is refined by differential evolution (log lengths ±0.10, angles ±4°, zero
shift ±0.40°; three restarts, maxiter 75, popsize 10, seeds from the CIF stem
without `_opt` and the restart index), keeping the restart with the largest
objective. A bidirectional guard requires every major peak ≥0.70 in either
pattern to have a counterpart ≥0.05 within 1.00 + 0.007·max(mean 2θ − 10, 0)
degrees. B = exp(−(1.00M + 0.30D + 0.30H)). A candidate matches iff the guard
passes and B ≥ 0.80.

Only the top-B candidate decides the group: `decision.json` records
`STOP_SUCCESS` if it matches, otherwise the search continues with the next group;
lower-ranked candidates are never used as a fallback. `scores.csv` lists all
representatives; `selected_refined.cif` (refined cell) is written on success.
Results do not depend on `--workers`. Reference packing labels never enter
matching. The engine is ported from Zikai Xie's `pxrd_csp_stop` 1.0.0.

## Optional packing validation

```bash
python tools/validate_crystal_packing.py --manifest runs/task/sg001-processed/representatives.csv --reference unoptimized_experimental.cif --output-dir runs/task/packing --timeout 30
```

Run this only if you have a separate CSD license and a compatible CCDC Python API environment. The repository does not install, download or redistribute the CCDC API or CSD records. Users without this license must skip this optional post-matching step. Only
standard-library dependencies are needed by this adapter; the reference and
candidate paths must be readable by that environment. PackingSimilarity requires
all 15 molecules in the shell, distance tolerance 0.2, angle tolerance 20°,
no molecular differences, artificial inversion enabled, and ignored hydrogen
positions/counts and bond counts. Each candidate runs in a bounded subprocess.
Errors and timeouts have null hit values and produce a nonzero command exit;
they are unresolved, not negative matches. The result is post-hoc validation and
cannot change the stopping decision. A peak match supports, but does not
establish, structural identity. DFT remains a separate workflow.

## Verification boundary

Tests cover manifests, hash/unit checks, molecular normalization, synthetic
structures, subprocess failure/timeout and the classifier and peak-matching workflows.
Execution smoke tests do not establish predictive accuracy or reproduce the
paper's full CSP results. The [trained classifier](model-weights.md) is a separate
public Release download. Licensed datasets remain excluded. Obtain ChemBERTa
and MACE through their upstream sources; do not substitute random weights.

The historical execution run took one real ethanol candidate through generation
and two-stage MACE relaxation with the earlier, since-removed profile ranking.
Peak matching was verified separately on recorded CSP pools. Original-helper comparisons
covered three synthetic structures. A licensed Windows CSD API self-match returned
15 matched molecules and zero RMSD, but its Python process stalled during exit.
Clean termination of that platform's complete packing CLI remains unverified.
The timeout path is regression-tested and records an unresolved result.
