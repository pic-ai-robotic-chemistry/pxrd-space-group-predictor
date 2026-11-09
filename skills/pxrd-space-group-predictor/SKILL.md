---
name: pxrd-space-group-predictor
description: Use the pxrd-sg CLI to predict molecular-crystal space groups from PXRD and SMILES, run a sequential BOMLIP/MACE space-group search stopped by PXRD peak matching, or peak-match existing crystal candidates.
---

# PXRD-guided space-group screening

## Locate the guides and check the environment

Resolve this skill's real path first. In a source checkout, the repository root
is two directories above the skill directory. In a wheel installation, the guide
root is `<environment>/share/pxrd-space-group-predictor/`. Resolve all guide paths
below from that root; a wheel does not create a source checkout.

Read `docs/cli.md` for command examples and `pxrd-sg --help` for available options.
For a new Linux x86-64 installation, follow the complete Conda installation steps
in `environments/README.md`, including its resolved version locks. The YAML files
alone only create Python and pip environments. Keep prediction and GPU engine
environments separate; select the engine with `--engine-python`. Other platforms
require separate verification. Reuse an existing dedicated environment after
checking it with `pxrd-sg doctor`.

## Establish inputs and choose the command

Obtain the raw two-column PXRD file (degrees 2theta and linear intensity), actual
wavelength, single-component SMILES and stable sample ID. The CLI currently
requires the paper's wavelength, 1.54184 Å. Keep raw PXRD as input: classification
preparation and peak matching use different preprocessing, handled by the CLI.

- **Predict space groups:** use `pxrd-sg predict`. Follow `docs/model-weights.md`
  to obtain the public classifier, trusted SHA256 and specified local ChemBERTa
  snapshot. Proceed when assets and `doctor` checks pass. Report missing assets
  rather than substituting a different or randomly initialized model.
- **Search space groups for a matching structure:** use `pxrd-sg run`. First read
  `docs/downstream-workflow.md` for the pinned BOMLIP source and MACE assets.
  Confirm the authorized candidate count and GPU, check available disk/hardware,
  and run `pxrd-sg doctor --full` with the engine and asset options. Start the
  requested computation only after checks pass. A prediction-only request does
  not authorize GPU search. Groups are searched in predicted (or `--space-groups`)
  order and the run stops at the first group whose top candidate matches. Use
  concrete search groups; if Others stops the workflow, obtain an explicit group
  selection before starting a new run.
- **Match existing candidates:** use `pxrd-sg rank` with raw PXRD, the
  relaxed-candidate manifest specified in `docs/cli.md` and a required
  `--space-groups` search order. This route does not require the classifier or
  ChemBERTa.

Use the CLI examples in `docs/cli.md` with explicit paths and a new output
directory. For installation validation, use its synthetic example and an
explicit one-candidate budget. This checks execution, not scientific accuracy.

## Check and deliver results

Check the exit status, `run.json` (`searched` and `result`), stage logs, output
counts and finite prediction, energy or B values. Preserve candidate IDs and report optimization step-limit
flags. Treat failed stages as unresolved. To resume, repeat the identical command
with `--resume`; the CLI checks provenance and output hashes before reusing a
completed stage. Preserve existing results and numbered attempt directories.

Deliver the executed command, code revision, input/model provenance, completed
stages, result (`STOP_SUCCESS` or `NO_MATCH_IN_SEARCH_LIST`), result paths and
unresolved issues. TOP-3 is a shortlist; a peak match supports PXRD agreement.
Neither establishes crystal-packing identity or reproduces the paper's
aggregate results by itself.

CSD is not required for the public CLI workflow. Optional reference-based packing
validation requires a separate CSD license and compatible CCDC API; consult
`docs/downstream-workflow.md` only when requested and licensed. Keep reference
labels out of candidate matching. DFT is a separate workflow. Respect asset
redistribution terms and share only authorized data.
