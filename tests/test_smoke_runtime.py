from __future__ import annotations

import unittest

import numpy as np
import torch

from src.models.xraypro import ModelConfig, XRayProModel
from src.training.losses import LossConfig, classification_loss


class RuntimeSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(3)
        self.model = XRayProModel(
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
        self.prediction = {
            "pxrd": torch.ones((2, 60)),
            "smiles_embedding": torch.ones((2, 8)),
        }
        self.graph = {
            "atom_fea": torch.ones((2, 3, 4)),
            "nbr_fea": torch.ones((2, 3, 2, 5)),
            "nbr_fea_idx": torch.zeros((2, 3, 2), dtype=torch.long),
            "atom_mask": torch.ones((2, 3), dtype=torch.bool),
        }

    def test_graph_free_logits_equal_training_path_logits(self) -> None:
        with torch.no_grad():
            graph_free = self.model(self.prediction)["multi_logits"]
            with_graph = self.model({**self.prediction, **self.graph})["multi_logits"]
        self.assertTrue(torch.equal(graph_free, with_graph))

    def test_single_modal_controls_require_only_the_selected_input(self) -> None:
        for modality, input_key in (("smiles", "smiles_embedding"), ("pxrd", "pxrd")):
            with self.subTest(modality=modality):
                config = ModelConfig(
                    pxrd_length=60,
                    smiles_embedding_dim=8,
                    num_classes=3,
                    atom_dim=4,
                    nbr_dim=5,
                    hidden_dim=8,
                    projection_dim=4,
                    graph_layers=1,
                    dropout=0.0,
                    prediction_modality=modality,
                )
                model = XRayProModel(config).eval()
                with torch.no_grad():
                    output = model({input_key: self.prediction[input_key]})
                self.assertEqual(tuple(output["multi_logits"].shape), (2, 3))

    def test_formal_loss_uses_graph_teacher(self) -> None:
        output = self.model({**self.prediction, **self.graph})
        losses = classification_loss(output, torch.tensor([0, 1]), LossConfig())
        self.assertTrue(torch.isfinite(losses["loss"]))



if __name__ == "__main__":
    unittest.main()
