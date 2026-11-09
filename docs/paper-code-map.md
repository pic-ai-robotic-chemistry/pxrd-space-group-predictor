# Correspondence to the current manuscript and Supporting Information

This guide maps the reported methods to the public implementation. It covers
numerical settings and data boundaries; installation and execution instructions
are in [the CLI guide](cli.md).

| Paper component | Public implementation | Verification or boundary |
| --- | --- | --- |
| SI-3 experimental classifier preprocessing | `src/data/pxrd.py`, `tools/prepare_pxrd.py` | Interpolation, zero fill, fifth-percentile/601-point baseline, nearest padding, linear max100. |
| SI-3 CSD training augmentation | `augment_csd_profile`, `ArrayDataset` | Independent probability 0.5 for noise and background; no experimental or validation augmentation. |
| SI-3 molecular representation | `tools/embed_smiles.py` | Pinned ChemBERTa revision, 192-token attention-mask mean, float32. Canonical SMILES preparation is external. |
| SI-4 model and graph alignment | `src/models`, `src/training/losses.py` | SERes/simple topology unchanged; inference does not require graphs. |
| SI-5 training and controls | `tools/train.py --stage` | Stage-specific defaults, explicit checkpoint initialization, optimizer reset, no test loader. |
| SI-6 ordinary/balanced recall and macro-F1 | `src/evaluation/metrics.py`, `tools/evaluate.py` | Stable descending logits; summaries plus per-class, confusion and rank-coverage JSON. |
| SI-6 five-seed paired criterion | `paired_seed_summary` | Same five seed IDs, unrounded recall fractions, sample standard deviation; mean gain >=0.02 and >=4 positive pairs. |
| SI-8 PXRD peak matching | `src/runtime/pxrd_peak_match.py`: `load_experimental_pxrd`, `corrected_pattern`, `peak_levels`, `refine_cell`, `strong_guard`, `score_components` | Measured range on the 0.02° grid, minimum-subtracted, max 100; SNIP on log(1+√I); three-restart cell refinement; bidirectional strong-peak guard; B = exp(−(1.00M + 0.30D + 0.30H)). |
| SI-8 per-group stopping rule | `evaluate_pool`, `tools/score_pxrd_candidates.py`, `execute_stages` in `src/runtime/workflow.py` | Top-B candidate only; match iff guard passes and B ≥ 0.80; stop at first matching group in search order, else `NO_MATCH_IN_SEARCH_LIST`. |
| SI-2 fixed split and molecular exclusion | External data preparation | No random split substitutes for the fixed 512/266/1655 paper protocol. |
| SI-7 generation and MACE relaxation | Downstream CLI adapters | `tools/generate_crystal_candidates.py` and `tools/relax_crystal_candidates.py` call the pinned external engine; models are not bundled. |
| SI-8 low-energy window, deduplication and retrospective packing validation | `low_energy_pool` in `src/runtime/crystal_processing.py`, `tools/process_crystal_candidates.py`, `validate_crystal_packing.py` | ΔE ≤ 3.0 kJ/mol per group; density-gated StructureMatcher. A peak match is not a packing match; matching does not inspect reference labels. |
| SI-9 VASP comparison | External DFT calculation | The three-structure result is evidence, not a classifier/peak-matching implementation. |

## Architecture and class mapping

The two 128-dimensional prediction features concatenate to 256 dimensions.
The formal SERes fusion layer maps this **256-dimensional input to 128 outputs**;
the projections then have 64 dimensions. The historical `fusion_hidden_dim=256`
configuration is used by a different optional CNN branch. It is not an extra
256-wide hidden layer in the formal SERes branch. Changing that layer would
change the verified network and checkpoint shapes. The generic API preserves
the original 4,833,230-parameter topology.

The paper class order is SG1, SG2, SG4, SG7, SG14, SG18, SG19, SG29, SG33,
SG61, Others. The configuration stores Others as `null`, whereas the internal
historical label was -1. Dataset labels remain zero-based indices (Others=10).
Others aggregates several groups; it is not a usable space-group number for CSP.

Graphs are supplied as 92-dimensional atom features with up to 12 periodic
neighbors within 8 Å and 41 Gaussian distance features (0–8 Å, 0.2 Å spacing).
The NPZ interface requires already constructed features and padded graphs.
It does not load CIFs, generate those features or recover licensed source rows.

## Training stages

| Setting | `--stage csd-pretrain` | `--stage experimental` (default) |
| --- | --- | --- |
| Maximum epochs | 150 | 120 |
| Train/evaluation batch size | 128 / 256 | 32 / 128 |
| Encoder/head learning rate | 1e-4 / 1e-4 | 3e-6 / 1e-4 |
| Weight decay | 1e-4 | 1e-4 |
| Scheduler monitor | ordinary validation TOP-3 | ordinary validation TOP-3 |
| Checkpoint selection | ordinary validation TOP-1 | ordinary validation TOP-3, then lower loss |
| PXRD augmentation | enabled on training data only | disabled |

Use `--config configs/model.example.json` to fix the paper dimensions and all
eleven classes. Pretraining uses seed 240427. Experimental comparisons use
240427, 240428, 240429, 240430 and 240431. Each adapted run loads the same
pretraining checkpoint using `--init-checkpoint`; the optimizer and scheduler
start afresh. The Experimental-only control omits this argument. The SMILES-only
control uses `--prediction-modality smiles` in **both stages** and a matching
model configuration, retains graph supervision and does not instantiate PXRD.
The no-graph control uses `--no-graph-supervision` to zero graph CE and alignment.
The optional PXRD-only API is a generic facility, not a reported experiment.

AdamW head parameters include `multi_fuse`, `multi_head`, `multi_proj`,
`graph_head` and `graph_proj`. The scheduler uses factor 0.3, patience 8 and
minimum LR 5e-7. With patience 8, the learning rate drops after nine
consecutive validation evaluations without improvement; early stopping uses patience 25. These are exposed CLI values,
not an assertion that new runs reproduce the historical checkpoints.

CSD perturbations use Gaussian sigma 0.002 times the current maximum and
constant/exponential background amplitudes bounded by 0.005/0.010 times that
maximum. The exponential term is exp(-3 position). Each operation independently
occurs with probability 0.5 and restores the original maximum and nonzero-span
padding. Optional integer `row_index` entries preserve original row identities
for deterministic augmentation; otherwise local dataset indices are used.

Balanced recall and macro-F1 average classes with positive test support, as in
the original implementation. All eleven paper test classes have positive
support. Custom partial-class tests report `active_classes` and null recalls
for absent classes, rather than implying an eleven-class evaluation.
`classification_details.json` includes ordinary rank coverage for K=1..10.
The paired-seed helper expects already matched evaluation scopes; it cannot
establish molecule-disjoint membership from metric values.

## Candidate scoring and external stages

The matcher reads the measured two-column pattern (`load_experimental_pxrd`),
clips it at zero, interpolates it onto the 0.02° grid within the measured range,
subtracts its minimum and scales it to 100. It does not use the classifier's
percentile background subtraction. Candidate sticks use Cu Kα
1.54184 Å with pseudo-Voigt FWHM 0.15° and η 0.5. `corrected_pattern` applies
Gaussian smoothing (σ 0.075°), SNIP (1.2°) on log(1+√I), a second smoothing
(σ 0.03375°) and max-1 normalisation; `peak_levels` extracts major, medium and
fine peaks. `refine_cell` varies log lengths ±0.10, angles ±4° and zero shift
±0.40° by differential evolution (three restarts, maxiter 75, popsize 10; seed
from the CIF stem without `_opt` and the restart index), keeping the largest
objective. `strong_guard` requires major peaks ≥0.70 in each pattern to have a
counterpart ≥0.05 within 1.00 + 0.007·max(mean 2θ − 10, 0) degrees.
`score_components` gives M, D and H for B. Results do not depend on worker count.
The engine is ported from Zikai Xie's `pxrd_csp_stop` 1.0.0.

Each space group is processed separately (`low_energy_pool`): energies relative
to that group's minimum, ΔE ≤ 3.0 kJ/mol, then StructureMatcher deduplication
in increasing energy against representatives within 0.2 g/cm³ density.
`evaluate_pool` ranks representatives by B; only the top candidate decides, with
no fallback. Downstream adapters call pinned BOMLIP/MACE and licensed
PackingSimilarity. VASP remains external. `docs/downstream-workflow.md` gives
launch instructions; `configs/paper_protocol.json` records numerical settings.
Independent packing matches require all 15 molecules in the packing shell. A
peak match supports, but does not establish, structural identity; packing
validation is optional and post hoc.

## Input validation

Classification preprocessing rejects profiles with no positive signal after
background correction. Array loading rejects rows with no positive signal in
float32 and reports their zero-based row indices. CSD augmentation rejects
profiles with no positive signal before drawing noise or background values.
These checks reject unusable inputs without changing valid-profile processing.

## Verification scope

Regression tests exercise numerical protocols, input contracts, checkpoint
round trips, one-epoch synthetic training, fine-tuning and graph-free prediction.
Peak matching was verified separately on recorded CSP pools.
These tests do not retrain the paper model, validate its scientific accuracy,
recompute the CSP/DFT results or redistribute private data/checkpoints.
