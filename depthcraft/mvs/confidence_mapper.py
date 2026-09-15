"""Extract & normalize MVS confidence maps for downstream fusion."""
from __future__ import annotations

import numpy as np


def normalize_confidence(raw_confidence: np.ndarray) -> np.ndarray:
    """Normalize PatchMatch photometric-consistency scores to [0, 1]."""
    c = raw_confidence.astype(np.float32)
    c = np.clip(c, 0, None)
    max_val = np.percentile(c, 99) + 1e-8
    return np.clip(c / max_val, 0, 1)


def estimate_confidence_from_depth_consistency(
    depth_a: np.ndarray, depth_b_reprojected: np.ndarray, threshold: float = 0.05
) -> np.ndarray:
    """Cheap confidence proxy when true PatchMatch confidence isn't available:
    agreement between two independently estimated depth maps of the same view
    (e.g. left-right consistency, or two MVS sweeps). Returns [0, 1] confidence.
    """
    rel_diff = np.abs(depth_a - depth_b_reprojected) / (depth_a + 1e-6)
    confidence = np.clip(1.0 - rel_diff / threshold, 0, 1)
    return confidence.astype(np.float32)
