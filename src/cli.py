"""Installed command-line interface. Help does not import scientific libraries."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys


def parser():
    p = argparse.ArgumentParser(prog="pxrd-sg", description="PXRD and molecular-crystal screening")
    p.add_argument("--version", action="version", version="pxrd-sg 0.3.0.dev1")
    sub = p.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="Check imports and optional model/GPU assets")
    doctor.add_argument("--full", action="store_true")
    doctor.add_argument("--checkpoint", type=Path)
    doctor.add_argument("--checkpoint-sha256")
    doctor.add_argument("--chemberta", type=Path)
    doctor.add_argument("--bomlip-root", type=Path)
    doctor.add_argument("--mace-model", type=Path)
    doctor.add_argument("--engine-python", type=Path, default=Path(sys.executable))
    doctor.add_argument("--gpu", type=int, default=0)
    doctor.add_argument("--device", default="cpu")
    for name in ("predict", "run", "rank"):
        c = sub.add_parser(name, help={"predict":"Raw PXRD + SMILES to space-group probabilities",
            "run":"Prediction, then sequential space-group CSP with PXRD peak-match stopping (explicit budget)",
            "rank":"Raw PXRD + relaxed candidate manifest to sequential peak-match decisions"}[name])
        c.add_argument("--pxrd", type=Path, required=True, help="Two columns: degrees 2theta, linear intensity")
        c.add_argument("--wavelength", type=float, required=True, help="Measured wavelength in angstrom; paper workflow requires 1.54184")
        c.add_argument("--output", type=Path, required=True)
        c.add_argument("--resume", action="store_true", help="Reuse only hash-verified completed stages of this run")
        c.add_argument("--timeout", type=float, default=86400, help="Seconds per external stage")
        if name == "rank":
            c.add_argument("--manifest", type=Path, required=True, help="Relaxed candidate CSV in project schema, energies kJ/mol per molecule")
            c.add_argument("--space-groups", type=int, nargs="+", required=True, help="Search order; stops at the first matching group")
            c.add_argument("--workers", type=int, default=1)
        else:
            c.add_argument("--smiles", required=True)
            c.add_argument("--sample-id", default="sample")
            c.add_argument("--checkpoint", type=Path, required=True)
            c.add_argument("--checkpoint-sha256", required=True, help="Expected hash from the trusted model supplier")
            c.add_argument("--chemberta", type=Path, required=True, help="Local ChemBERTa-100M-MLM snapshot; no automatic download")
            c.add_argument("--device", default="cpu")
            c.add_argument("--top-k", type=int, default=3)
        if name == "run":
            c.add_argument("--space-groups", type=int, nargs="+", help="Override the predicted search order with explicit space groups")
            c.add_argument("--count", type=int, required=True, help="Candidates per group; 5000 is the paper budget, small values are execution checks")
            c.add_argument("--workers", type=int, default=1)
            c.add_argument("--gpu", type=int, required=True)
            c.add_argument("--bomlip-root", type=Path, required=True)
            c.add_argument("--mace-model", type=Path, required=True)
            c.add_argument("--engine-python", type=Path, default=Path(sys.executable))
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        if args.command == "doctor":
            from src.runtime.workflow import doctor
            result = doctor(args)
            print(json.dumps(result, indent=2))
            return 0 if result["ok"] else 1
        from src.runtime.workflow import execute
        execute(args)
        return 0
    except KeyboardInterrupt:
        print("Interrupted. Completed stages are retained; use the same arguments with --resume.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"pxrd-sg: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
