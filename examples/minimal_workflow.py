#!/usr/bin/env python3
"""Minimal graph-free forward smoke example."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.xraypro import ModelConfig, XRayProModel


def main() -> None:
    torch.manual_seed(7)
    model = XRayProModel(
        ModelConfig(
            pxrd_length=60,
            smiles_embedding_dim=8,
            num_classes=3,
            atom_dim=4,
            nbr_dim=5,
            hidden_dim=8,
            projection_dim=4,
            graph_layers=1,
            dropout=0.0,
        )
    ).eval()
    with torch.no_grad():
        logits = model(
            {
                "pxrd": torch.ones((2, 60)),
                "smiles_embedding": torch.ones((2, 8)),
            }
        )["multi_logits"]
    print({"logits_shape": list(logits.shape)})


if __name__ == "__main__":
    main()
