# Conda environment profiles

The CLI uses one environment for prediction, candidate processing and ranking.
It can call generation and MACE relaxation through a second Python interpreter
set with `--engine-python`. Keep the environments separate: the verified engine
uses older PyTorch and NumPy versions than the prediction stack. Neither public
workflow requires CSD. Do not install the CLI into an existing research or MACE
environment.

## CPU prediction and ranking

Run the commands below from the cloned repository root. Create a Linux x86-64
Python 3.10 environment, then install the CPU PyTorch wheel
from PyTorch's CPU index and the resolved version lock:

```bash
conda env create -f environments/predict-cpu-linux.yml
conda activate pxrd-predict-cpu
python -m pip install --no-deps --index-url https://download.pytorch.org/whl/cpu 'torch==2.7.1+cpu'
python -m pip install --no-deps -r environments/predict-cpu-linux.lock.txt
python -m pip install --no-deps .
pxrd-sg doctor
python -m pip check
```

The lock was captured from a clean Python 3.10 environment. It pins runtime
packages and transitive versions, but not wheel hashes or operating-system
libraries. This profile is not the historical training environment. Keep the package versions recorded in `run.json` when comparing
results between machines. Linux x86-64 validation does not establish support for
Apple Silicon or native Windows.

The Conda CPU profile was tested on Linux x86-64 with a fresh Python 3.10
environment: dependency checks, all 58 tests and real-checkpoint prediction
passed. The installed environment occupied approximately 1.7 GB, excluding
models and the Conda package cache. Run source tests from the repository root
because some tests read files under `examples/`:

```bash
python -m unittest discover -s tests -v
```

## NVIDIA GPU MACE engine

Use a Linux x86-64 host with an NVIDIA driver, CUDA toolkit, and a compatible GPU.
Create a separate Python 3.10 environment. The lock pins the verified CUDA 12.1
PyTorch build and PyG extension wheels. Install BOMLIP and its bundled MACE source
from the exact commit used by this project:

```bash
conda env create -f environments/mace-engine-linux.yml
conda activate pxrd-mace-engine

mkdir -p engines
git clone https://github.com/pic-ai-robotic-chemistry/BOMLIP-CSP.git engines/BOMLIP-CSP
git -C engines/BOMLIP-CSP checkout --detach 2caf530c8aa2ca14230264bfe69bfbcd6952b41e

python -m pip install --no-cache-dir -r environments/mace-engine-linux.lock.txt
# Build from a temporary copy to keep the runtime checkout clean.
BOMLIP_BUILD=$(mktemp -d)
git -C engines/BOMLIP-CSP archive HEAD | tar -x -C "$BOMLIP_BUILD"
python -m pip install --no-cache-dir --no-deps "$BOMLIP_BUILD/mace-bench/3rdparty/mace"
python -m pip install --no-cache-dir --no-deps "$BOMLIP_BUILD/mace-bench"
rm -r -- "$BOMLIP_BUILD"
git -C engines/BOMLIP-CSP status --porcelain
python -m pip check
```

The MACE source and BOMLIP batch optimizer must come from that checkout; do not
substitute the separately published `mace-torch` package. Keep the checkout clean
after installation because each run verifies and snapshots its pinned files.
The `git status` command above must produce no output. Installing directly from
the runtime checkout can leave build files that cause this check to fail.
Set `CUDA_HOME` to the host CUDA toolkit if it is not `/usr/local/cuda` and make
its `bin` directory available on `PATH`. The lock includes Python CUDA runtime
libraries; it does not install the system driver or CUDA compiler. The verified
host used PyTorch 2.4.1+cu121 with CUDA toolkit 12.6 and an NVIDIA A800.

Fresh Conda environments passed an end-to-end execution check on 22 October
2026 with version 0.2 (synthetic PXRD, ethanol, space group 1, one candidate):
all stages, resume and output-hash checks passed. That version ranked candidates
by profile similarity, which 0.3 replaced with peak matching; the 0.3 matching
stages were run on recorded CSP pools instead. This checks installation
and execution, not recovery of an experimental structure or reproduction of the
paper's results. The CPU environment used about 1.7 GB and the GPU engine about
6.3 GB, excluding models, package caches and workflow outputs. Existing research
environments were not modified. This route requires no container runtime.

From the prediction environment, run a full preflight before candidate work:

```bash
pxrd-sg doctor --full --gpu 0 \
  --engine-python /path/to/conda/envs/pxrd-mace-engine/bin/python \
  --bomlip-root engines/BOMLIP-CSP \
  --mace-model models/MACE-OFF23_small.model
```

Use the generated candidates only after reviewing the predicted space groups and
choosing an explicit count and GPU. `--count 1` is an execution smoke test, not
the paper's candidate budget. See [`docs/downstream-workflow.md`](../docs/downstream-workflow.md)
for the fixed scientific settings and input/output requirements.
