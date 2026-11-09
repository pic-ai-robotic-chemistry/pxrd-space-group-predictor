from __future__ import annotations

import argparse
import contextlib
import csv
import datetime
import io
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from examples.make_synthetic_data import make_bundle
from src.data.arrays import load_array_bundle
from src.data.pxrd import CLASSIFICATION_GRID, preprocess_experimental, augment_csd_profile
from src.evaluation.metrics import topk_summary
from src.models.xraypro import CrystalGraphEncoder, ModelConfig, XRayProModel
from src.runtime.checkpoints import load_checkpoint, save_checkpoint
from src.runtime.io import model_config_from_bundle, validate_bundle_for_model
from src.training.losses import LossConfig, classification_loss
from tools import evaluate, infer, train


class InputContractTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="xraypro-contract-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "data.npz"
        make_bundle(self.path, size=3, seed=2, labels=True, graph=True)
        with np.load(self.path) as archive:
            self.arrays = {key: archive[key] for key in archive.files}

    def load(self, **changes):
        np.savez(self.path, **{**self.arrays, **changes})
        return load_array_bundle(
            self.path, prediction_modality="pxrd_smiles",
            require_labels=True, require_graph=True,
        )

    def test_metrics_reject_nonfinite_logits_and_invalid_labels(self) -> None:
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite"):
                topk_summary(torch.tensor([[value, 0.], [0., value]]), torch.tensor([0, 1]))
        for labels in (torch.tensor([-1, 0]), torch.tensor([0, 2]), torch.tensor([0., 1.])):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                topk_summary(torch.zeros(2, 2), labels)

    def test_valid_metrics_preserve_denominators(self) -> None:
        logits = torch.tensor([[4., 2., 1.], [4., 2., 1.], [4., 2., 1.]])
        metrics = topk_summary(logits, torch.tensor([0, 0, 1]))
        self.assertEqual(metrics["n"], 3)
        self.assertAlmostEqual(metrics["top1"], 2 / 3)
        self.assertEqual(metrics["balanced_top1"], 0.5)
        self.assertEqual(metrics["top3"], 1.)

    def test_arrays_reject_nonfinite_and_overflowing_features(self) -> None:
        for key in ("pxrd", "smiles_embedding", "atom_fea", "nbr_fea"):
            for value in (np.nan, np.inf, 1e100):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.load(**{key: np.full(self.arrays[key].shape, value)})

    def test_classifier_rejects_profiles_without_positive_signal(self) -> None:
        cases = [([80., 81.], [10., 100.]),
                 (CLASSIFICATION_GRID, np.zeros(3400)),
                 (CLASSIFICATION_GRID, np.ones(3400))]
        for theta, intensity in cases:
            with self.subTest(points=len(theta)), self.assertRaisesRegex(ValueError, "positive signal"):
                preprocess_experimental(theta, intensity)

    def test_bundle_reports_zero_signal_row(self) -> None:
        for value in (0., 1e-100):
            profiles = self.arrays["pxrd"].astype(np.float64)
            profiles[1] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, r"positive signal.*\[1\]"):
                self.load(pxrd=profiles)
        self.assertEqual(self.load().size, 3)

    def test_csd_augmentation_rejects_zero_signal_before_random_draws(self) -> None:
        for epoch in (0, 1, 2):
            with self.subTest(epoch=epoch), self.assertRaisesRegex(ValueError, "positive signal"):
                augment_csd_profile(torch.zeros(3400), seed=240427, epoch=epoch,
                                    index=0, row_index=0)

    def test_graph_contract_rejects_bad_indices_axes_and_masks(self) -> None:
        changes = [
            {"nbr_fea_idx": np.full((3, 3, 2), -1.5)},
            {"nbr_fea_idx": np.full((3, 3, 2), -1)},
            {"nbr_fea_idx": np.full((3, 3, 2), 3)},
            {"nbr_fea": np.ones((3, 2, 2, 5))},
            {"atom_mask": np.zeros((3, 3), dtype=bool)},
            {"atom_mask": np.ones((3, 3), dtype=float)},
            {"atom_mask": np.tile([True, True, False], (3, 1)),
             "nbr_fea_idx": np.full((3, 3, 2), 2)},
        ]
        for change in changes:
            with self.subTest(change=list(change)), self.assertRaises(ValueError):
                self.load(**change)

    def test_valid_padding_is_accepted(self) -> None:
        bundle = self.load(
            atom_mask=np.tile([True, True, False], (3, 1)),
            nbr_fea_idx=np.zeros((3, 3, 2), dtype=np.int64),
        )
        self.assertEqual(bundle.size, 3)

    def test_direct_graph_api_rejects_negative_indices(self) -> None:
        encoder = CrystalGraphEncoder(4, 5, 8, 1)
        with self.assertRaisesRegex(ValueError, "neighbor indices"):
            encoder(
                torch.ones(2, 3, 4), torch.ones(2, 3, 2, 5),
                torch.full((2, 3, 2), -1, dtype=torch.long),
                torch.ones(2, 3, dtype=torch.bool),
            )

    def test_explicit_classes_allow_missing_training_classes(self) -> None:
        bundle = self.load(labels=np.array([0, 1, 0]))
        path = self.root / "model.json"
        path.write_text(json.dumps({"num_classes": 3, "class_space_groups": [1, 2, 19]}))
        config = model_config_from_bundle(path, bundle, prediction_modality="pxrd_smiles")
        self.assertEqual(config.num_classes, 3)
        inherited = model_config_from_bundle(
            None, bundle, prediction_modality="pxrd_smiles", base_config=config,
        )
        self.assertEqual(inherited, config)
        with self.assertRaisesRegex(ValueError, "num_classes"):
            validate_bundle_for_model(self.load(labels=np.array([0, 1, 3])), config)

    def test_mapping_defines_class_count_and_rejects_invalid_mappings(self) -> None:
        path = self.root / "model.json"
        path.write_text(json.dumps({"class_space_groups": [1, 2, 19]}))
        config = model_config_from_bundle(path, self.load(), prediction_modality="pxrd_smiles")
        self.assertEqual(config.num_classes, 3)
        for groups in ([1, 2], [1, 1, 2], [0, 2, 19], [1, 2, 231], [1, 2, 19.5]):
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                ModelConfig(num_classes=3, class_space_groups=groups)

    def test_alignment_nan_stops_loss(self) -> None:
        output = {
            "multi_logits": torch.zeros(2, 3), "graph_logits": torch.zeros(2, 3),
            "multi_proj": torch.ones(2, 4), "graph_proj": torch.full((2, 4), float("nan")),
        }
        with self.assertRaisesRegex(ValueError, "non-finite"):
            classification_loss(output, torch.tensor([0, 1]), LossConfig())

    def test_checkpoint_safe_roundtrip_and_legacy_config(self) -> None:
        model = XRayProModel(ModelConfig(
            prediction_modality="smiles", smiles_embedding_dim=8,
            num_classes=3, class_space_groups=[1, 2, 19],
        ))
        optimizer = torch.optim.AdamW(model.parameters())
        model({"smiles_embedding": torch.randn(2, 8)})["multi_logits"].sum().backward()
        optimizer.step()
        path = self.root / "model.pt"
        save_checkpoint(path, model=model, epoch=1, metrics={"loss": 1.}, optimizer=optimizer)
        restored, payload = load_checkpoint(path, device=torch.device("cpu"))
        self.assertEqual(restored.config, model.config)
        self.assertTrue(payload["optimizer_state"]["state"])
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, restored.state_dict()[key]))
        del payload["model_config"]["class_space_groups"]
        torch.save(payload, path)
        legacy, _ = load_checkpoint(path, device=torch.device("cpu"))
        self.assertIsNone(legacy.config.class_space_groups)
        torch.save({"unexpected_object": datetime.date(2026, 9, 9)}, path)
        with self.assertRaises(pickle.UnpicklingError):
            load_checkpoint(path, device=torch.device("cpu"))

    def test_training_finetuning_and_prediction_workflow(self) -> None:
        config_path = self.root / "model.json"
        config_path.write_text(json.dumps({
            "hidden_dim": 8, "projection_dim": 4, "graph_layers": 1,
            "dropout": 0., "num_classes": 3, "class_space_groups": [1, 2, 19],
        }))
        with patch("sys.argv", [
            "train.py", "--train", str(self.path), "--val", str(self.path),
            "--output-dir", str(self.root / "run"), "--config", str(config_path),
            "--epochs", "1", "--batch-size", "3", "--device", "cpu",
        ]):
            args = train.parse_args()
        with contextlib.redirect_stdout(io.StringIO()):
            result = train.run(args)
        self.assertEqual(result["best_epoch"], 1)
        checkpoint = args.output_dir / "best.pt"
        make_bundle(self.root / "test.npz", size=3, seed=4, labels=True, graph=False)
        eval_args = argparse.Namespace(
            checkpoint=checkpoint, data=self.root / "test.npz", device="cpu",
            batch_size=3, output_dir=self.root / "evaluation",
        )
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(evaluate.run(eval_args)["n"], 3)
        with (eval_args.output_dir / "predictions.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([int(row["label_space_group"]) for row in rows], [1, 2, 19])
        make_bundle(self.root / "infer.npz", size=2, seed=5, labels=False, graph=False)
        infer_args = argparse.Namespace(
            checkpoint=checkpoint, data=self.root / "infer.npz", device="cpu",
            batch_size=2, top_k=3, output=self.root / "predictions.csv",
        )
        infer.run(infer_args)
        with infer_args.output.open() as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            indices = [int(value) for value in row["predicted_class_indices"].split(";")]
            self.assertEqual(row["predicted_space_groups"], ";".join(str([1, 2, 19][i]) for i in indices))
        np.savez(self.root / "subset.npz", **{
            key: value[:2] for key, value in self.arrays.items()
        })
        args.train = self.root / "subset.npz"
        args.config = None
        args.init_checkpoint = checkpoint
        args.output_dir = self.root / "finetune"
        with contextlib.redirect_stdout(io.StringIO()):
            result = train.run(args)
        self.assertEqual(result["model_config"]["class_space_groups"], [1, 2, 19])
        self.assertEqual(result["model_config"]["num_classes"], 3)
        np.savez(self.root / "bad_val.npz", **{
            **self.arrays, "labels": np.array([0, 1, 3]),
        })
        args.val = self.root / "bad_val.npz"
        args.output_dir = self.root / "invalid_validation"
        with self.assertRaisesRegex(ValueError, "num_classes"):
            train.run(args)
        self.assertFalse(args.output_dir.exists())
        args.val = self.path
        args.output_dir = self.root / "invalid_gradient"
        original_forward = XRayProModel.forward

        def forward_with_bad_gradient(model, batch):
            output = original_forward(model, batch)
            output["multi_logits"].register_hook(lambda gradient: torch.full_like(gradient, float("inf")))
            return output

        with patch.object(XRayProModel, "forward", forward_with_bad_gradient):
            with patch.object(torch.optim.AdamW, "step") as step:
                with self.assertRaisesRegex(RuntimeError, "non-finite"):
                    train.run(args)
                step.assert_not_called()
        self.assertFalse((args.output_dir / "best.pt").exists())
        infer_args.output = self.root / "invalid.csv"
        with patch.object(XRayProModel, "forward", return_value={"multi_logits": torch.full((2, 3), float("nan"))}):
            with self.assertRaisesRegex(ValueError, "finite"):
                infer.run(infer_args)
        self.assertFalse(infer_args.output.exists())


if __name__ == "__main__":
    unittest.main()
