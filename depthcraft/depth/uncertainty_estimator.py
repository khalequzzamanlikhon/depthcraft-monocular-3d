"""Depth uncertainty via Test-Time Augmentation (TTA).

Runs the depth engine on flipped/rotated variants of the input and reports
the pixelwise standard deviation as an uncertainty map, per Phase 1 §2.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class DepthEngineProtocol(Protocol):
    def infer(self, image: np.ndarray) -> np.ndarray: ...


def estimate_uncertainty_tta(
    engine: DepthEngineProtocol,
    image: np.ndarray,
    n_augmentations: int = 4,
    max_rotation_deg: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Test-time augmentation uncertainty estimation.

    Returns (mean_depth, uncertainty_map) both shaped like the base depth map.
    """
    import cv2

    h, w = image.shape[:2]
    predictions = [engine.infer(image)]

    # horizontal flip
    flipped = engine.infer(np.ascontiguousarray(image[:, ::-1, :]))
    predictions.append(flipped[:, ::-1])

    # small rotations
    rng = np.random.default_rng(42)
    for _ in range(max(0, n_augmentations - 2)):
        angle = rng.uniform(-max_rotation_deg, max_rotation_deg)
        center = (w / 2, h / 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_img = cv2.warpAffine(image, rot_mat, (w, h), borderMode=cv2.BORDER_REFLECT)
        pred = engine.infer(rotated_img)
        inv_rot_mat = cv2.invertAffineTransform(rot_mat)
        pred_unrotated = cv2.warpAffine(pred, inv_rot_mat, (w, h), borderMode=cv2.BORDER_REFLECT)
        predictions.append(pred_unrotated)

    stack = np.stack(predictions, axis=0)
    mean_depth = stack.mean(axis=0)
    uncertainty = stack.std(axis=0)
    return mean_depth.astype(np.float32), uncertainty.astype(np.float32)


def uncertainty_at_pixel(uncertainty_map: np.ndarray, x: int, y: int, window: int = 3) -> float:
    """Local-window average uncertainty around a clicked pixel (more stable than a single pixel)."""
    h, w = uncertainty_map.shape
    x0, x1 = max(0, x - window), min(w, x + window + 1)
    y0, y1 = max(0, y - window), min(h, y + window + 1)
    return float(np.mean(uncertainty_map[y0:y1, x0:x1]))
