"""Regression checks against the current manuscript's numerical definitions."""
import argparse
import csv
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from src.data.pxrd import CLASSIFICATION_GRID, preprocess_experimental, augment_csd_profile
from src.data.arrays import ArrayBundle, ArrayDataset
from src.models.xraypro import ModelConfig, XRayProModel
from src.runtime.io import display_space_group
from src.evaluation.metrics import classification_details, topk_summary, paired_seed_summary
from tools import train
from examples.make_synthetic_data import make_bundle
from src.runtime.checkpoints import save_checkpoint, load_checkpoint
from tools import infer


class PaperProtocolTest(unittest.TestCase):
    def test_interpolation_zero_fill_and_percentile_baseline(self):
        theta = CLASSIFICATION_GRID[100:900]
        y = np.ones(800) * 20
        y[390:400] = 60
        result = preprocess_experimental(theta[::-1], y[::-1])
        expected = np.zeros(3400, dtype=np.float32)
        # Zero-filled support affects the rolling baseline near its boundaries.
        expected[100:370] = 50
        expected[630:900] = 50
        expected[490:500] = 100
        np.testing.assert_allclose(result, expected)
        self.assertEqual(result.dtype, np.float32)
        with self.assertRaises(ValueError):
            preprocess_experimental([2, 2], [1, 2])

    def test_percentile_is_order_statistic_with_nearest_padding(self):
        y = np.linspace(1, 10, 3400)
        windows = np.lib.stride_tricks.sliding_window_view(np.pad(y, 300, mode="edge"), 601)
        expected = y - np.sort(windows, axis=1)[:, 30]
        expected *= 100 / expected.max()
        np.testing.assert_allclose(preprocess_experimental(CLASSIFICATION_GRID, y), expected, rtol=1e-6)

    def test_augmentation_is_csd_only_repeatable_and_preserves_support(self):
        values = torch.zeros(100)
        values[10:90] = torch.linspace(1, 100, 80)
        kwargs = dict(seed=240427, epoch=1, index=0, row_index=15)
        a = augment_csd_profile(values, **kwargs)
        torch.testing.assert_close(a, augment_csd_profile(values, **kwargs))
        self.assertEqual(float(a.max()), 100)
        self.assertEqual(float(a[:10].sum() + a[90:].sum()), 0)
        bundle = ArrayBundle({"pxrd": values.numpy()[None, :]}, 1, False, False)
        torch.testing.assert_close(ArrayDataset(bundle)[0]["pxrd"], values)
        variants = [augment_csd_profile(values, **{**kwargs, "epoch": e}) for e in range(10)]
        self.assertTrue(any(not torch.equal(values, v) for v in variants))

    def test_stage_defaults_and_head_parameter_group(self):
        common = ["train", "--train", "train.npz", "--val", "val.npz", "--output-dir", "run"]
        with patch("sys.argv", common + ["--stage", "csd-pretrain"]):
            args = train.parse_args()
        self.assertEqual((args.epochs, args.batch_size, args.eval_batch_size), (150, 128, 256))
        self.assertEqual((args.selection_metric, args.learning_rate, args.head_learning_rate), ("top1", 1e-4, 1e-4))
        with patch("sys.argv", common):
            args = train.parse_args()
        self.assertEqual((args.selection_metric, args.learning_rate, args.head_learning_rate), ("top3", 3e-6, 1e-4))
        model = XRayProModel(ModelConfig(prediction_modality="smiles"))
        optimizer = train.make_optimizer(model, args)
        rates = {id(parameter): group["lr"] for group in optimizer.param_groups
                 for parameter in group["params"]}
        self.assertEqual(rates[id(model.graph_proj.weight)], 1e-4)
        self.assertEqual(rates[id(model.smiles[1].weight)], 3e-6)

    def test_others_mapping_and_architecture(self):
        config = ModelConfig(class_space_groups=[1, 2, 4, 7, 14, 18, 19, 29, 33, 61, None])
        model = XRayProModel(config)
        self.assertEqual(model.multi_fuse[0].in_features, 256)
        self.assertEqual(model.multi_fuse[0].out_features, 128)
        self.assertEqual(display_space_group(config.class_space_groups[-1]), "Others")
        self.assertEqual(sum(p.numel() for p in model.parameters()), 4833230)

    def test_metrics_confusion_rank_coverage_and_absent_classes(self):
        logits = torch.tensor([[2., 1., 0.], [2., 1., 0.], [2., 1., 0.]])
        labels = torch.tensor([0, 0, 1])
        metrics = topk_summary(logits, labels)
        self.assertAlmostEqual(metrics["macro_f1"], 0.4)
        self.assertEqual(metrics["active_classes"], 2)
        details = classification_details(logits, labels)
        self.assertEqual(details["confusion"], [[2, 0, 0], [1, 0, 0], [0, 0, 0]])
        self.assertIsNone(details["per_class"][2]["top1"])
        self.assertEqual(details["rank_coverage"]["2"], 1.)

    def test_paired_seed_rule_uses_ids_and_sample_standard_deviation(self):
        base = {s: 0.25 for s in range(5)}
        treated = {s: v for s, v in enumerate([0.30, 0.30, 0.30, 0.30, 0.20])}
        result = paired_seed_summary(base, treated)
        self.assertTrue(result["criterion_passed"])
        self.assertEqual(result["positive_pairs"], 4)
        self.assertAlmostEqual(result["treatment_sample_std"], np.std(list(treated.values()), ddof=1))
        with self.assertRaises(ValueError):
            paired_seed_summary(base, {s + 1: v for s, v in treated.items()})

    def test_others_roundtrip_and_inference_preserve_logit_ranking(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = ModelConfig(prediction_modality="smiles", smiles_embedding_dim=8,
                                 num_classes=3, class_space_groups=[1, 2, None])
            model = XRayProModel(config)
            checkpoint = root / "model.pt"
            save_checkpoint(checkpoint, model=model, epoch=0, metrics={})
            restored, _ = load_checkpoint(checkpoint, device=torch.device("cpu"))
            self.assertEqual(restored.config.class_space_groups[-1], None)
            np.savez(root / "input.npz", smiles_embedding=np.ones((1, 8), dtype=np.float32))
            args = argparse.Namespace(checkpoint=checkpoint, data=root / "input.npz",
                    output=root / "output.csv", device="cpu", batch_size=1, top_k=3)
            # Softmax underflows both low probabilities to zero, but logits order 2 before 1.
            with patch.object(XRayProModel, "forward", return_value={"multi_logits": torch.tensor([[1000., -2., -1.]])}):
                infer.run(args)
            with args.output.open() as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["predicted_class_indices"], "0;2;1")
            self.assertEqual(row["predicted_space_groups"], "1;Others;2")

    def test_pretraining_selects_top1_while_adaptation_selects_top3(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data.npz"
            make_bundle(data, size=3, seed=4, labels=True, graph=True)
            metrics = [{"top1": .8, "top3": .6, "loss": .2},
                       {"top1": .7, "top3": .9, "loss": .1}]
            for stage, expected in [("csd-pretrain", 1), ("experimental", 2)]:
                argv = ["train", "--train", str(data), "--val", str(data), "--output-dir", str(root / stage),
                        "--epochs", "2", "--stage", stage, "--device", "cpu", "--batch-size", "3"]
                with patch("sys.argv", argv):
                    args = train.parse_args()
                with patch.object(train, "evaluate", side_effect=metrics), contextlib.redirect_stdout(io.StringIO()):
                    result = train.run(args)
                self.assertEqual(result["best_epoch"], expected)
                self.assertEqual(result["scheduler_metric"], "top3")


if __name__ == "__main__":
    unittest.main()
