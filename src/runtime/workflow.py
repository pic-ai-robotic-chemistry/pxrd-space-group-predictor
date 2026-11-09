"""Orchestrate existing scientific adapters with explicit inputs and resumable stages."""
from __future__ import annotations
import csv
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from src.runtime.crystal_candidates import sha256
from src.runtime.downstream_execution import BOMLIP_COMMIT, MACE_SHA256, run_logged


def record(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256(path)}


def tree_identity(path):
    path = Path(path).resolve()
    if not path.is_dir():
        raise ValueError(f"missing directory: {path}")
    files = {str(f.relative_to(path)): sha256(f) for f in sorted(path.rglob("*"))
             if f.is_file() and ".cache" not in f.relative_to(path).parts}
    if not files:
        raise ValueError(f"empty model directory: {path}")
    return {"path": str(path), "files": files}


def check_checkpoint(path, expected):
    if not expected or not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
        raise ValueError("supply the trusted checkpoint's 64-character SHA256")
    identity = file_identity(path)
    if identity["sha256"] != expected.lower():
        raise ValueError("checkpoint SHA256 mismatch")
    return identity


def check_upstream(root):
    root = Path(root).resolve()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    if git("rev-parse", "HEAD") != BOMLIP_COMMIT or git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError(f"BOMLIP must be a clean checkout at {BOMLIP_COMMIT}")
    return {"path": str(root), "commit": BOMLIP_COMMIT}


def probe_environment(python, full=False, gpu=0, device="cpu", embedding=True):
    # Probe actual imports, not only installed distribution metadata.
    modules = ["numpy", "scipy", "torch", "rdkit", "pymatgen.core", "ase"]
    modules += ["mace", "cuequivariance", "cuequivariance_torch"] if full else (["transformers", "tokenizers", "safetensors"] if embedding else [])
    code = """import importlib, importlib.metadata as m, json, sys, shutil
result = {'python': sys.version, 'executable': sys.executable, 'imports': {}, 'versions': {}}
for name in json.loads(sys.argv[1]):
    try:
        importlib.import_module(name)
        result['imports'][name] = 'ok'
    except Exception as exc:
        result['imports'][name] = type(exc).__name__ + ': ' + str(exc)
for name in ['numpy','scipy','torch','rdkit','pymatgen','ase','transformers','tokenizers','safetensors','mace-torch','cuequivariance','cuequivariance-torch','ninja']:
    try: result['versions'][name] = m.version(name)
    except m.PackageNotFoundError: pass
result['ok'] = all(v == 'ok' for v in result['imports'].values())
if sys.argv[2] == 'True' or sys.argv[4].startswith('cuda'):
    try:
        import torch
        index = int(sys.argv[3]) if sys.argv[2] == 'True' else int(sys.argv[4].split(':')[1]) if ':' in sys.argv[4] else 0
        available = torch.cuda.is_available() and 0 <= index < torch.cuda.device_count()
        result['cuda'] = {'available': available, 'build': torch.version.cuda, 'index': index}
        if available: torch.ones(1, device='cuda:' + str(index)).cpu()
        result['ok'] = result['ok'] and available
    except Exception as exc:
        result['cuda'] = str(exc); result['ok'] = False
if sys.argv[2] == 'True':
    result['ninja'] = shutil.which('ninja')
    result['ok'] = result['ok'] and bool(result['ninja'])
print(json.dumps(result))
"""
    env = os.environ.copy()
    env["PATH"] = str(Path(python).absolute().parent) + os.pathsep + env.get("PATH", "")
    result = subprocess.run([str(python), "-c", code, json.dumps(modules), str(full), str(gpu), device],
                            capture_output=True, text=True, timeout=120, env=env)
    if result.returncode:
        raise RuntimeError(f"environment check failed: {result.stderr[-2000:]}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def doctor(args):
    report = {"ok": True, "checks": {}}
    def check(name, fn):
        try:
            result = fn()
            report["checks"][name] = result
            if isinstance(result, dict) and result.get("ok") is False:
                report["ok"] = False
        except Exception as exc:
            report["checks"][name] = {"error": str(exc)}
            report["ok"] = False
    check("prediction_environment", lambda: probe_environment(sys.executable, device=args.device))
    if args.checkpoint:
        check("checkpoint", lambda: check_checkpoint(args.checkpoint, args.checkpoint_sha256))
    if args.chemberta:
        check("chemberta", lambda: tree_identity(args.chemberta))
    if args.full:
        check("engine_environment", lambda: probe_environment(args.engine_python, True, args.gpu))
        check("bomlip", lambda: check_upstream(args.bomlip_root))
        check("mace_model", lambda: check_checkpoint(args.mace_model, MACE_SHA256))
    report["scope"] = "Dependency and asset checks; not a completed scientific workflow or a CSD license check."
    return report


class Run:
    """Reuse completed stages only when inputs, implementation and output hashes agree."""
    def __init__(self, root, identity, resume=False):
        self.root = Path(root).resolve()
        self.path = self.root / "run.json"
        existed = self.root.exists()
        if existed:
            if not resume or not self.path.is_file():
                raise ValueError("output already exists; use a new directory or --resume for the same run")
        else:
            if resume:
                raise ValueError("cannot resume: output directory does not exist")
            self.root.mkdir(parents=True)
        import fcntl
        self.lock = (self.root / ".run.lock").open("a")
        try:
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.lock.close()
            raise ValueError("another process is using this output directory") from exc
        try:
            if existed:
                self.state = json.loads(self.path.read_text())
                if self.state["identity"] != identity:
                    raise ValueError("resume inputs, options, models, environment or implementation changed; use a new output")
            else:
                self.state = {"schema_version": 1, "identity": identity, "stages": {}, "status": "running"}
                record(self.path, self.state)
        except BaseException:
            self.lock.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        try:
            if error is not None:
                self.state.update(status="failed", error=f"{kind.__name__}: {error}")
                record(self.path, self.state)
            elif self.state["status"] == "complete":
                self.state.pop("error", None)
                record(self.path, self.state)
        finally:
            self.lock.close()

    def stage(self, name, function):
        old = self.state["stages"].get(name)
        if old and old["status"] == "complete":
            for relative, digest in old["files"].items():
                if sha256(self.root / relative) != digest:
                    raise ValueError(f"completed stage {name} was modified: {relative}")
            print(f"Reusing verified stage: {name}", flush=True)
            return self.root / old["directory"]
        attempt = (old or {}).get("attempt", 0) + 1
        directory = self.root / f"{name}-{attempt:03d}"
        entry = {"status": "running", "attempt": attempt, "directory": directory.name}
        self.state["stages"][name] = entry
        self.state["status"] = "running"
        record(self.path, self.state)
        print(f"Running stage: {name}", flush=True)
        try:
            function(directory)
            files = {str(p.relative_to(self.root)): sha256(p) for p in sorted(directory.rglob("*"))
                     if p.is_file() and "__pycache__" not in p.parts}
            if not files:
                raise ValueError(f"stage {name} produced no files")
            entry.update(status="complete", files=files)
        except BaseException as exc:
            entry.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            self.state["status"] = "failed"
            record(self.path, self.state)
            raise
        record(self.path, self.state)
        return directory


def invoke(run, module, arguments, directory, *, python=None, timeout=86400):
    # A checkout and an installed wheel both contain src/ and tools/ beside each other.
    package_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = env["TRANSFORMERS_OFFLINE"] = "1"
    # Append our adapter location after the selected interpreter's dependencies.
    # Prepending another venv's site-packages can load the wrong NumPy/PyTorch.
    bootstrap = "import runpy,sys; sys.path.append(sys.argv.pop(1)); runpy.run_module(sys.argv.pop(1), run_name='__main__')"
    command = [str(python or sys.executable), "-c", bootstrap, str(package_root), module, *map(str, arguments)]
    entry = {"command": command, "cwd": str(run.root)}
    run.state.setdefault("commands", []).append(entry)
    record(run.path, run.state)
    run_logged(command, cwd=run.root, env=env, log=run.root / (directory.name + ".log"), timeout=timeout)


def experimental_inputs(path):
    """Classification profile and the measured pattern used for peak matching."""
    import numpy as np
    from src.data.pxrd import preprocess_experimental
    from src.runtime.pxrd_peak_match import load_experimental_pxrd
    data = np.loadtxt(path, ndmin=2)
    if data.shape[1] != 2:
        raise ValueError("PXRD input must have exactly two columns")
    return preprocess_experimental(data[:, 0], data[:, 1]), load_experimental_pxrd(path)


def choose_groups(predictions, override=None):
    with Path(predictions).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise ValueError("expected one prediction row")
    values = override if override is not None else rows[0]["predicted_space_groups"].split(";")
    try:
        groups = [int(g) for g in values]
    except (TypeError, ValueError) as exc:
        raise ValueError("predicted shortlist contains Others; supply explicit --space-groups, no substitution is automatic") from exc
    if not groups or len(set(groups)) != len(groups) or any(not 1 <= g <= 230 for g in groups):
        raise ValueError("search groups must be unique integers in 1..230")
    return groups


def pool_groups(rows, order):
    present = {int(r["space_group"]) for r in rows}
    if len(set(order)) != len(order):
        raise ValueError("search groups must be unique")
    if missing := [g for g in order if g not in present]:
        raise ValueError(f"manifest has no candidates for space groups {missing}")
    return order


def execute(args):
    import numpy as np
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    if not math.isclose(args.wavelength, 1.54184, rel_tol=0, abs_tol=1e-5):
        raise ValueError("this model workflow requires Cu K-alpha wavelength 1.54184 angstrom; conversion is not automatic")
    prepared, measured = experimental_inputs(args.pxrd)
    identity = {"command": args.command, "input": file_identity(args.pxrd), "wavelength": args.wavelength,
                "timeout": args.timeout}
    package = Path(__file__).resolve().parents[2]
    identity["implementation"] = {str(p.relative_to(package)): sha256(p)
        for folder in ("src", "tools") for p in sorted((package / folder).rglob("*.py"))}
    identity["environment"] = probe_environment(sys.executable, device=getattr(args, "device", "cpu"), embedding=args.command != "rank")
    if not identity["environment"]["ok"]:
        raise RuntimeError("environment is incomplete; run pxrd-sg doctor for details")
    if args.command != "rank":
        from rdkit import Chem, rdBase
        from src.runtime.checkpoints import load_checkpoint
        import torch
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.sample_id):
            raise ValueError("sample ID must contain letters, digits, underscores or hyphens")
        molecule = Chem.MolFromSmiles(args.smiles)
        if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
            raise ValueError("a valid single-component SMILES is required")
        smiles = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        identity.update(smiles=smiles, input_smiles=args.smiles, rdkit_version=rdBase.rdkitVersion,
                        sample_id=args.sample_id, top_k=args.top_k, device=args.device,
                        checkpoint=check_checkpoint(args.checkpoint, args.checkpoint_sha256),
                        chemberta=tree_identity(args.chemberta))
        model, _ = load_checkpoint(args.checkpoint, device=torch.device("cpu"))
        config = model.config.to_dict()
        del model
        if config["pxrd_length"] != 3400 or config["smiles_embedding_dim"] != 768 or config["prediction_modality"] != "pxrd_smiles":
            raise ValueError("checkpoint is incompatible with this PXRD/ChemBERTa workflow")
        if not config.get("class_space_groups") or not 1 <= args.top_k <= config["num_classes"]:
            raise ValueError("checkpoint must supply class mapping and top-k must fit its classes")
        identity["model_config"] = config
    if args.command == "run":
        if args.count < 1 or args.workers < 1 or args.gpu < 0:
            raise ValueError("count and workers must be positive; GPU index must be nonnegative")
        identity.update(count=args.count, workers=args.workers, gpu=args.gpu, space_groups=args.space_groups,
                        bomlip=check_upstream(args.bomlip_root), mace=check_checkpoint(args.mace_model, MACE_SHA256),
                        engine=probe_environment(args.engine_python, True, args.gpu))
        if not identity["engine"]["ok"]:
            raise RuntimeError("MACE environment/GPU check failed; run doctor --full")
    if args.command == "rank":
        from src.runtime.crystal_candidates import read_candidates
        candidates = read_candidates(args.manifest)
        identity["manifest"] = file_identity(args.manifest)
        identity["cifs"] = [file_identity(r["cif_path"]) for r in candidates]
        identity["space_groups"] = pool_groups(candidates, args.space_groups)
    with Run(args.output, identity, args.resume) as run:
        return execute_stages(args, run, prepared, measured,
                              smiles if args.command != "rank" else None,
                              config if args.command != "rank" else None)


def execute_stages(args, run, prepared, measured, smiles, config):
    import numpy as np
    def prepare(directory):
        directory.mkdir()
        np.save(directory / "classification.npy", prepared, allow_pickle=False)
        np.savetxt(directory / "experimental.xy", measured)
        if args.command != "rank":
            record(directory / "model.json", config)
            with (directory / "molecules.csv").open("w", newline="") as stream:
                writer = csv.writer(stream); writer.writerow(["sample_id", "smiles"]); writer.writerow([args.sample_id, smiles])
            with (directory / "profiles.csv").open("w", newline="") as stream:
                writer = csv.writer(stream); writer.writerow(["sample_id", "pxrd_path"]); writer.writerow([args.sample_id, "classification.npy"])
    inputs = run.stage("inputs", prepare)
    def call(module, arguments, directory, python=None):
        invoke(run, module, arguments, directory, python=python, timeout=args.timeout)
    if args.command != "rank":
        def embed(directory):
            directory.mkdir()
            call("tools.embed_smiles", ["--input", inputs / "molecules.csv", "--output", directory / "embeddings.npz",
                 "--model-name", args.chemberta.resolve(), "--device", args.device], directory)
        embedded = run.stage("embedding", embed)
        def predict(directory):
            from tools.build_inference_bundle import build_bundle
            directory.mkdir()
            build_bundle(inputs / "profiles.csv", embedded / "embeddings.npz", inputs / "model.json", directory / "bundle.npz")
            call("tools.infer", ["--checkpoint", args.checkpoint.resolve(), "--data", directory / "bundle.npz",
                 "--output", directory / "predictions.csv", "--device", args.device, "--top-k", args.top_k], directory)
        predicted = run.stage("prediction", predict)
        if args.command == "predict":
            run.state.update(status="complete", predictions=str((predicted / "predictions.csv").relative_to(run.root)))
            record(run.path, run.state)
            print(f"Predictions: {predicted / 'predictions.csv'}")
            return
        groups = choose_groups(predicted / "predictions.csv", args.space_groups)
    else:
        from src.runtime.crystal_candidates import read_candidates, write_candidates
        groups = args.space_groups
        rows = read_candidates(args.manifest.resolve())
    run.state["search_groups"] = groups
    searched = []
    for group in groups:
        tag = f"sg{group:03d}"
        if args.command == "run":
            def generate(directory, group=group):
                call("tools.generate_crystal_candidates", ["--smiles", smiles, "--sample-id", args.sample_id,
                    "--space-groups", group, "--bomlip-root", args.bomlip_root.resolve(), "--output-dir", directory,
                    "--count", args.count, "--workers", args.workers, "--timeout", args.timeout], directory, args.engine_python)
            generated = run.stage(f"{tag}-generation", generate)
            def relax(directory):
                call("tools.relax_crystal_candidates", ["--manifest", generated / "candidates.csv", "--bomlip-root", args.bomlip_root.resolve(),
                    "--model", args.mace_model.resolve(), "--output-dir", directory, "--gpu", args.gpu,
                    "--workers", args.workers, "--timeout", args.timeout], directory, args.engine_python)
            manifest = run.stage(f"{tag}-relaxation", relax) / "relaxed.csv"
        else:
            def split(directory, group=group):
                directory.mkdir()
                pool = [r for r in rows if int(r["space_group"]) == group]
                write_candidates(directory / "pool.csv", pool)
            manifest = run.stage(f"{tag}-pool", split) / "pool.csv"
        def process(directory, manifest=manifest):
            call("tools.process_crystal_candidates", ["--manifest", manifest, "--output-dir", directory], directory)
        processed = run.stage(f"{tag}-processing", process)
        def score(directory, processed=processed):
            call("tools.score_pxrd_candidates", ["--experimental", inputs / "experimental.xy",
                "--manifest", processed / "representatives.csv", "--output-dir", directory,
                "--workers", args.workers], directory)
        matched = run.stage(f"{tag}-matching", score)
        decision = json.loads((matched / "decision.json").read_text())
        searched.append({"space_group": group, "signal": decision["signal"],
                         "top_B": decision["top_candidate"]["B"], "directory": matched.name})
        if decision["signal"] == "STOP_SUCCESS":
            break
    run.state.update(status="complete", searched=searched,
                     result="STOP_SUCCESS" if searched[-1]["signal"] == "STOP_SUCCESS" else "NO_MATCH_IN_SEARCH_LIST",
                     interpretation="A PXRD peak match supports, but does not establish, structural identity.")
    record(run.path, run.state)
    for entry in searched:
        print(f"SG{entry['space_group']}: {entry['signal']} (top B={entry['top_B']:.4f})")
    print(f"Result: {run.state['result']}; details in {run.path}")
