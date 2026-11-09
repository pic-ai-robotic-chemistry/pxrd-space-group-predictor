# Installed CLI

The installed `pxrd-sg` command connects the existing scientific adapters. It does
not change their scientific settings. Python 3.10 on Linux x86-64 is the validated target.
macOS and native Windows installations remain unverified. Full MACE runs require a compatible
NVIDIA GPU environment. Windows users should use Linux/WSL2 for this orchestration.

## Installation

For a new Linux x86-64 installation, first follow the tested Conda profiles and
resolved version locks in [`environments/README.md`](../environments/README.md).
That guide installs the CLI and, for full runs, a separate GPU engine. The YAML
files create Python environments; the guide supplies the remaining installation
commands. Use this route for the validated dependency versions.

Alternatively, install from this checkout into an existing compatible dedicated
environment:

```bash
python -m pip install '.[workflow]'
pxrd-sg --help
pxrd-sg doctor
```

This dependency-range installation is a convenience, not a frozen historical
environment. See `environments/README.md` for the tested version profiles. The
wheel includes all command adapters, so commands work outside the checkout.
The installed skill and protocol files are under
`<environment>/share/pxrd-space-group-predictor/`. Locate the active environment
with `python -c 'import sys; print(sys.prefix)'` in Conda or venv.

## Required model assets

Prediction needs a compatible schema-1 classification checkpoint and its trusted
SHA256, plus a local snapshot of `DeepChem/ChemBERTa-100M-MLM` revision
`f5c45f44d3061f0346888f5c09db17ec1146d29d` including tokenizer files. Obtain these
from the [public classifier release](model-weights.md) and the upstream ChemBERTa
repository, respectively. This CLI does not download models or contact Hugging Face
during a run. It records local snapshot hashes and canonicalizes input SMILES with
the installed RDKit version. The version is recorded and may differ from the
historical study; do not claim historical bitwise reproduction from execution alone.

Full runs additionally need the clean pinned BOMLIP checkout and MACE model
described in `downstream-workflow.md`. CSD is not required for candidate ranking.
Optional CCDC packing validation remains a separate licensed command.

## Predict one sample

`observed.xy` must contain two whitespace-separated columns: degrees 2theta and
linear intensity. The workflow currently requires the paper's wavelength; it does
not silently convert other radiation sources. Use the actual experimental data.

```bash
pxrd-sg predict --pxrd observed.xy --smiles 'CCO' --sample-id sample-1 \
  --wavelength 1.54184 --checkpoint model.pt --checkpoint-sha256 "$MODEL_SHA256" \
  --chemberta models/chemberta --output runs/prediction
```

`run.json` points to the predictions CSV and records inputs, model hashes, class
mapping, installed package versions, commands and output checksums. The raw
pattern used for peak matching and the background-corrected classification input
are kept separate.

## Full candidate workflow

```bash
pxrd-sg run --pxrd observed.xy --smiles 'CCO' --sample-id sample-1 \
  --wavelength 1.54184 --checkpoint model.pt --checkpoint-sha256 "$MODEL_SHA256" \
  --chemberta models/chemberta --bomlip-root engines/BOMLIP-CSP \
  --mace-model models/MACE-OFF23_small.model --engine-python /opt/mace/bin/python \
  --count 1 --workers 1 --gpu 0 --output runs/example
```

The explicit `run`, `--count` and `--gpu` options authorize the requested search.
The default search list is the top three predicted labels, searched in predicted
order. If it includes Others, the command stops after prediction; it does not
replace Others with an arbitrary space group. Review predictions and run with
explicit `--space-groups` (the search order) in a new output directory if needed.
`--space-groups 1` is useful for the synthetic ethanol execution example; it is an
override, not a claim about experimental symmetry. The paper budget is 5,000
candidates per group, potentially requiring substantial GPU time. A count of one
is only a software execution check.

Each group runs four stages: `sgNNN-generation`, `sgNNN-relaxation`,
`sgNNN-processing` and `sgNNN-matching`. The search stops at the first group whose
decision is `STOP_SUCCESS`; later groups are not generated.

The engine Python may be in a separate environment. The prediction environment
must contain the `workflow` dependencies; the engine environment needs the pinned
BOMLIP/MACE stack. `doctor --full` accepts the same asset and engine options and
checks their presence, imports, model hash, upstream revision and selected GPU.
It does not prove the external optimizer will converge.

Processing keeps representatives within 3.0 kJ/mol of the group minimum
(`representatives.csv`, `deduplication.csv`, `provenance.json`). Matching writes
`scores.csv` (all representatives ranked by B), `decision.json` and, on success,
`selected_refined.cif` with the refined cell. Only the top-B candidate decides;
there is no fallback to lower-ranked candidates. `run.json` records `searched`
(per-group signal and top B) and `result`: `STOP_SUCCESS` or
`NO_MATCH_IN_SEARCH_LIST`. Energies are kJ/mol per molecule. Generated space
groups are pool labels, not independently redetermined final symmetries. A peak
match supports, but does not establish, structural identity.

## Match existing candidates

```bash
pxrd-sg rank --pxrd observed.xy --wavelength 1.54184 \
  --manifest relaxed.csv --space-groups 14 2 --workers 4 --output runs/ranked
```

The manifest uses the existing candidate schema: sample ID, concrete space group,
unique candidate ID, CIF path/hash, per-molecule energy, unit and relaxation status.
It must describe one molecule and comparable completed two-stage outputs.
`--space-groups` is required: unique groups, each present in the manifest, in
search order. Each group runs `sgNNN-pool`, `sgNNN-processing` and
`sgNNN-matching` with the same stopping rule as `run`. Results do not depend on
`--workers`. No classifier or ChemBERTa checkpoint is required.

## Synthetic execution example

The bundled generator writes a small, unrestricted XY profile:

```bash
PXRD_ENV=$(python -c 'import sys; print(sys.prefix)')
python "$PXRD_ENV/share/pxrd-space-group-predictor/examples/make_synthetic_data.py" \
  --raw-only --output-dir example-input
```

Use `example-input/synthetic.xy` with SMILES `CCO` and the full command above,
adding `--space-groups 1 --count 1`. Supply the real classifier, ChemBERTa and
MACE assets. The peaks are artificial and do not represent measured ethanol PXRD.
This checks software execution, not predictive accuracy or structure recovery.

## Resume and failures

Repeat the identical command with `--resume`. Completed stages are reused only
when input/model hashes, options, implementation, environment and output hashes
match. Failed attempts remain in their directories; a retry uses a new numbered
directory. Resume is at stage boundaries, not within an interrupted optimization.
Do not move the run directory during execution. Changed inputs require a new run.
Concurrent use of one output directory is rejected. No existing outputs are erased.

Missing dependencies, hash mismatch, invalid input, unavailable GPU, unknown
labels and subprocess failures return nonzero exit codes. Inspect `run.json` and
the per-stage log. The wrapper never substitutes random weights or treats failed
packing validation as a negative structural match.
