# Training and direct adapters

For prediction and candidate workflows, start with [the CLI guide](cli.md).
These interfaces accept prepared arrays and are for advanced use. Commands below
run from a source checkout root.

## Environment

Use the tested profiles in [the environment guide](../environments/README.md).
The sections below document lower-level adapters and training, not the primary CLI workflow.

## Array contract

Training and evaluation use an NPZ bundle. Required inference arrays are:

```text
pxrd               float32 [N, L]
smiles_embedding   float32 [N, D]
```

Training additionally requires:

```text
labels              int64   [N]
atom_fea            float32 [N, A, F]
nbr_fea             float32 [N, A, K, E]
nbr_fea_idx         int64   [N, A, K]
atom_mask           bool    [N, A]
```

Evaluation requires `labels`; graph arrays are optional because reported prediction logits do not use the graph pathway. Optional `sample_id` values are emitted in prediction files.

All feature arrays must contain finite real values representable as float32.
Labels are zero-based class indices, not space-group numbers, and must be smaller
than the model's `num_classes`. Every split must use the same index mapping.
Graph neighbor indices must be integers in `[0, A)`. Masks must be boolean with
at least one real atom per sample; neighbors of real atoms must refer to real
atoms. Padded atom rows may use index zero. NaN, infinity and malformed graph
inputs are rejected rather than repaired.

The classifier consumes already prepared profiles. The formal grid is
`2theta[i] = 2.0 + 0.02 * i` degrees for `i = 0, ..., 3399`, ending at 69.98
degrees. Use `tools/prepare_pxrd.py` for experimental classification inputs: linear
interpolation with zero fill outside measured support, a centered 601-point
fifth-percentile baseline with nearest-edge padding, clipping at zero and
maximum scaling to 100. Outputs are linear intensities, not log1p caches.
Training and prediction consume these prepared arrays without preprocessing
them a second time. CIF simulation is available in the downstream workflow; licensed database extraction remains external. Neither is performed by this command. Use the same preparation and embedding protocol for training and
inference. Matching the array length alone does not establish compatibility
with a checkpoint. Custom grids are supported for models trained on those grids.

## Class definitions and fine-tuning

Set `num_classes` in the model JSON to preserve the full label space even when
a training subset omits classes. Without an explicit count, a class mapping or
an initial checkpoint, the count is inferred as `max(train_labels) + 1`.
Validation labels outside that range are rejected before training starts.

An optional `class_space_groups` list in the model JSON maps indices to unique
International Tables space-group numbers (1 through 230), with a single
`null` entry for an aggregate Others class. Others is displayed as `Others`,
never as a concrete CSP space group. For example,
`"num_classes": 3, "class_space_groups": [1, 2, 19]` maps class index 2 to
space group 19. This is a small format example. `configs/model.example.json` contains the
paper mapping: `[1, 2, 4, 7, 14, 18, 19, 29, 33, 61, null]`.
The list can also define the class count when `num_classes` is omitted.
The mapping is saved in the checkpoint. Evaluation and inference add mapped
space-group columns when it is present; otherwise outputs remain class indices.

With `--init-checkpoint`, training inherits the initial model configuration,
including the full class count and mapping. An optional `--config` must remain
compatible with that configuration. Thus a subset missing the highest-indexed
class can be fine-tuned without shrinking the classifier head.

## Commands

```bash
python tools/embed_smiles.py --input smiles.csv --output molecular_embeddings.npz
python tools/train.py --train data/train.npz --val data/val.npz --output-dir runs/example
python tools/evaluate.py --checkpoint runs/example/best.pt --data data/test.npz --output-dir evaluation
python tools/infer.py --checkpoint runs/example/best.pt --data data/inference.npz --output predictions.csv
python tools/process_crystal_candidates.py --manifest relaxed_sg014.csv --output-dir processed
python tools/score_pxrd_candidates.py --experimental observed.xy --manifest processed/representatives.csv --output-dir matched --workers 4
```

`embed_smiles.py` defaults to `DeepChem/ChemBERTa-100M-MLM` revision
`f5c45f44d3061f0346888f5c09db17ec1146d29d`, attention-mask mean pooling and a
maximum token length of 192. Supply canonical isomeric SMILES
prepared with the same canonicalization protocol (RDKit 2025.09.1 in the paper);
this tool does not canonicalize text or reconstruct the licensed molecule splits. Install the optional embedding dependencies with
`python -m pip install -e '.[embedding]'`.

A synthetic, non-scientific smoke workflow is provided in `examples/minimal_workflow.py`. Its values only verify software contracts.

`process_crystal_candidates.py` takes the relaxed candidates of one space group,
keeps those within 3.0 kJ/mol of that group's minimum and deduplicates only that
window, in increasing energy, with pymatgen `StructureMatcher` (ltol 0.2, stol
0.3, angle_tol 5°, `ElementComparator`, hydrogen ignored, `primitive_cell=False`,
`scale=True`, `attempt_supercell=False`). A candidate is compared only with
representatives whose density differs by at most 0.2 g/cm³; the lowest-energy
member of each cluster is retained. It writes `representatives.csv`,
`deduplication.csv` and `provenance.json`.

`score_pxrd_candidates.py` scores one pool against a measured two-column pattern.
The pattern is clipped at zero, interpolated onto the 0.02° grid within its measured
range, minimum-subtracted and scaled to a maximum of 100. Candidate sticks are simulated with pymatgen for Cu Kα
(1.54184 Å) and broadened by pseudo-Voigt (FWHM 0.15°, η 0.5). Both patterns are
Gaussian-smoothed (σ 0.075°), background-corrected by SNIP (1.2°) on a
log(1+√I) scale, smoothed again (σ 0.03375°) and max-normalised; major, medium
and fine peak sets are then extracted. The candidate cell is refined by
differential evolution (log lengths ±0.10, angles ±4°, zero shift ±0.40°; three
restarts, maxiter 75, popsize 10, seeded from the CIF stem without `_opt` and the
restart index); the restart with the largest objective is kept. A bidirectional
strong-peak guard requires every major peak ≥0.70 in either pattern to have a
counterpart ≥0.05 within r = 1.00 + 0.007·max(mean 2θ − 10, 0) degrees. The score
is B = exp(−(1.00M + 0.30D + 0.30H)); a candidate matches iff the guard passes and
B ≥ 0.80. Only the top-B candidate decides the pool, with no fallback. Outputs are
`scores.csv`, `decision.json` and, on success, `selected_refined.cif`. Results do
not depend on `--workers`. The engine is ported from Chengxi Zhao's `pxrd_csp_stop`
1.0.0. These transformations are separate from classifier preprocessing.

Checkpoint loading uses PyTorch's restricted `weights_only=True` loader.
Existing staging schema-1 checkpoints without a class mapping remain readable;
arbitrary pickled Python objects are not supported. Training rejects non-finite
losses and gradients; evaluation and inference reject non-finite logits before
writing prediction outputs.

No packing labels or experimental structure are used for matching.

See [paper-to-code correspondence](paper-code-map.md) for training stages,
metrics, numerical definitions, downstream adapters and the external DFT boundary.

Run the synthetic CPU regression suite with:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_*.py'
```

If the test environment reports an MKL OpenMP symbol error, prefix the command with
`MKL_THREADING_LAYER=GNU`; this is an environment workaround, not a model setting.

To assemble prepared profiles and cached molecular embeddings, use:

```bash
python tools/build_inference_bundle.py --manifest profiles.csv --embeddings embeddings.npz --config configs/model.example.json --output runs/task/inference.npz
```

The CSV requires `sample_id,pxrd_path`; relative paths resolve against the CSV.
The NPZ requires string `sample_id` and numeric `smiles_embedding` arrays.
Rows are joined by ID, and the ID sets must match exactly. Duplicate IDs,
invalid arrays and existing output paths are rejected.

For a trusted original SE-Res/simple-graph embedding checkpoint, use
`tools/import_legacy_checkpoint.py --help`. Supply a verified SHA256, the verified
model configuration and an explicit provenance label. The importer requires the
3400-point grid and creates a separate schema-1 copy by renaming graph keys;
it does not change learned tensors. Check original-versus-converted predictions
before scientific use. The designated public checkpoint is described in the
[model guide](model-weights.md). Licensed inputs and other unpublished
research assets remain excluded.

