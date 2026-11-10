# Public classifier weights

The [weights-v1 Release](https://github.com/pic-ai-robotic-chemistry/pxrd-space-group-predictor/releases/tag/weights-v1)
provides the paper's full_pretrained classifier for seed 240427, retained at epoch
53. It uses CSD pretraining followed by experimental adaptation. This is one
trained model, not an ensemble or the complete set of five training runs.

## Download and verify

Download the checkpoint into a new model directory:

```bash
mkdir -p models
curl --fail --location \
  https://github.com/pic-ai-robotic-chemistry/pxrd-space-group-predictor/releases/download/weights-v1/full_pretrained_seed240427.pt \
  --output models/full_pretrained_seed240427.pt
export MODEL_SHA256=95e74dd41574a61839f0f1fc10f51075ba5f00de89b70b9f63bc214292369439
python - <<'PY'
import hashlib
import os
from pathlib import Path
path = Path("models/full_pretrained_seed240427.pt")
if hashlib.sha256(path.read_bytes()).hexdigest() != os.environ["MODEL_SHA256"]:
    raise SystemExit("Checkpoint SHA256 mismatch")
print("Checkpoint SHA256 verified")
PY
```

The file is 19,426,259 bytes. The release also includes `model_config.json`,
`provenance.json`, the release-time model card as `MODEL_CARD.md`, and `SHA256SUMS` for all four files.
The configuration and class mapping are embedded in the checkpoint.

Use the verified file with the installed CLI:

```bash
pxrd-sg predict --pxrd observed.xy --smiles 'CCO' --sample-id sample-1 \
  --wavelength 1.54184 --checkpoint models/full_pretrained_seed240427.pt \
  --checkpoint-sha256 "$MODEL_SHA256" --chemberta models/chemberta \
  --output runs/prediction
```

Replace `observed.xy` and `CCO` with the sample's experimental profile and SMILES.
See [the CLI guide](cli.md) for installation, input requirements and full candidate
runs. The CLI checks the hash again and loads the checkpoint with
`weights_only=True`. It never downloads model files during a run.

## Model and input protocol

The model predicts eleven labels from PXRD and a molecular embedding. The class
order is `[1, 2, 4, 7, 14, 18, 19, 29, 33, 61, null]`, where `null` means Others.
Others is an aggregate label and cannot be used as a concrete CSP search group.
The graph branch provided supervision during training; prediction needs no
crystal graph or CSD license.

The classifier expects 3,400 linear-intensity points from 2.00 to 69.98 degrees
2theta, at intervals of 0.02 degrees. The CLI interpolates experimental inputs,
subtracts a local-percentile baseline, clips negative values and scales the
maximum to 100. The filter uses percentile 5, a 601-point window and nearest-edge
padding. Already prepared arrays must not be processed a second time.

Supply a local snapshot of `DeepChem/ChemBERTa-100M-MLM`, including tokenizer files,
at revision `f5c45f44d3061f0346888f5c09db17ec1146d29d`. The embedding uses
attention-mask mean pooling, a maximum length of 192 tokens and 768 float32
features. The CLI records the RDKit version used to canonicalize SMILES.
ChemBERTa is an upstream dependency and is not included in this release.

## Provenance and scope

The released file is byte-identical to the compatible checkpoint used in the
recorded CLI validation. Its conversion changed graph parameter names only,
without changing learned tensors. The original research checkpoint has SHA256
`5248cd3e37b30c918fe3f5e0499aa5cd118ef67aba0aedfb449c131cc82d1b6d`.

The embedded provenance retains the historical label `internal_restricted` from
conversion. The author subsequently authorized this designated file for public
release. The label describes its earlier status; the bytes remain unchanged.
The checkpoint contains model configuration, tensors, epoch and provenance. It
contains no optimizer state, training rows, molecular identifiers, PXRD arrays
or crystal coordinates. This release does not redistribute CSD/PDF records.

The model was trained for molecular-crystal space-group screening within the
paper's label mapping and input protocol. One checkpoint does not reproduce
five-seed averages. A ranked shortlist does not establish structural identity or
guarantee that the experimental space group is included. The synthetic execution
example checks software operation, not scientific accuracy.

Full candidate runs also require the separately configured BOMLIP-CSP/MACE stack.
MACE weights are obtained upstream. CCDC packing validation is optional and
requires the user's own CSD license. See [the downstream guide](downstream-workflow.md).

The author has authorized public distribution of this file. The repository's
source code is MIT-licensed; formal license terms for this weight file are still
pending, and the MIT License does not extend to it.

## Download ChemBERTa

In the prediction environment, use the included `huggingface_hub` package to
fetch the fixed [upstream revision](https://huggingface.co/DeepChem/ChemBERTa-100M-MLM/tree/f5c45f44d3061f0346888f5c09db17ec1146d29d):

```bash
python - <<'PYCODE'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="DeepChem/ChemBERTa-100M-MLM",
    revision="f5c45f44d3061f0346888f5c09db17ec1146d29d",
    local_dir="models/chemberta",
    allow_patterns=["*.json", "*.txt", "*.safetensors"],
)
PYCODE
pxrd-sg doctor --checkpoint models/full_pretrained_seed240427.pt \
  --checkpoint-sha256 "$MODEL_SHA256" --chemberta models/chemberta
```

Download requires network access; subsequent CLI runs use the local snapshot.
Keep its tokenizer, configuration and safetensors files together.

## MACE for full runs

MACE-OFF23-small is supplied by [ACEsuit/mace-off](https://github.com/ACEsuit/mace-off).
Its [Academic Software License](https://github.com/ACEsuit/mace-off/blob/main/LICENSE.md)
permits academic use and restricts commercial use; users must comply with those
terms. The project's software license does not replace the model's license.

```bash
mkdir -p models
curl --fail --location \
  https://raw.githubusercontent.com/ACEsuit/mace-off/main/mace_off23/MACE-OFF23_small.model \
  --output models/MACE-OFF23_small.model
python - <<'PYCODE'
import hashlib
from pathlib import Path
expected = "165cce4cfec5a34b9c64d4ebf95de15d71106bb584b7291c8470f0749977c46f"
if hashlib.sha256(Path("models/MACE-OFF23_small.model").read_bytes()).hexdigest() != expected:
    raise SystemExit("MACE SHA256 mismatch: stop; do not substitute another model")
print("MACE SHA256 verified")
PYCODE
```

The fixed hash identifies the required bytes even if the upstream branch changes.
Use a new model directory or reuse verified files; preserve existing model assets.
