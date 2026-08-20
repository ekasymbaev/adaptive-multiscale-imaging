from __future__ import annotations

import unittest

import numpy as np
import torch

from adaptive_multiscale.data.coarse import downsample_image, downsample_mask
from adaptive_multiscale.models import CompactUNet
from adaptive_multiscale.training.losses import WeightedBCEDiceLoss
from adaptive_multiscale.training.metrics import binary_segmentation_metrics


class CoarseModelTests(unittest.TestCase):
    def test_downsampling_shape_and_binary_mask(self) -> None:
        image = np.tile(np.arange(2048, dtype=np.float32), (1536, 1))
        image = np.clip(image / image.max() * 255, 0, 255).astype(np.uint8)
        mask = np.zeros((1536, 2048), dtype=np.uint8)
        mask[200:900, 300:1200] = 255
        coarse_image = downsample_image(image, 384, 512)
        coarse_mask = downsample_mask(mask, 384, 512)
        self.assertEqual(coarse_image.shape, (384, 512))
        self.assertEqual(coarse_image.dtype, np.uint8)
        self.assertEqual(coarse_mask.shape, (384, 512))
        self.assertEqual(set(np.unique(coarse_mask)), {0, 1})

    def test_compact_unet_preserves_spatial_shape(self) -> None:
        model = CompactUNet(encoder_channels=(16, 32, 64, 128))
        model.eval()
        with torch.inference_mode():
            output = model(torch.zeros(1, 1, 384, 512))
        self.assertEqual(tuple(output.shape), (1, 1, 384, 512))
        self.assertLess(sum(parameter.numel() for parameter in model.parameters()), 1_000_000)

    def test_loss_is_finite_and_differentiable(self) -> None:
        logits = torch.zeros(2, 1, 8, 8, requires_grad=True)
        target = torch.zeros_like(logits)
        target[:, :, 2:6, 2:6] = 1.0
        loss = WeightedBCEDiceLoss(positive_weight=3.0)(logits, target)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_metrics_use_foreground_class(self) -> None:
        target = np.array([[1, 1], [0, 0]], dtype=bool)
        prediction = np.array([[1, 0], [1, 0]], dtype=bool)
        metrics = binary_segmentation_metrics(prediction, target)
        self.assertEqual(metrics["true_positive"], 1)
        self.assertEqual(metrics["false_positive"], 1)
        self.assertEqual(metrics["false_negative"], 1)
        self.assertAlmostEqual(metrics["dice"], 0.5)
        self.assertAlmostEqual(metrics["iou"], 1 / 3)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
