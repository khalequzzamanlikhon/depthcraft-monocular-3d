"""Confidence-weighted fusion of monocular depth and MVS depth (Section 5.2).

Fully real, numpy-only implementation -- no heavy dependencies.
"""
from __future__ import annotations

import numpy as np


def confidence_fusion(
    mono_depth: np.ndarray,
    mono_conf: np.ndarray,
    mvs_depth: np.ndarray,
    mvs_conf: np.ndarray,
    mvs_conf_threshold: float = 0.3,
    mvs_fail_threshold: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse monocular and MVS depth using confidence-weighted (inverse-variance-like)
    averaging, falling back to scaled monocular depth where MVS has effectively failed.

    All input arrays must be the same (H, W) shape, with confidences in [0, 1].
    Returns (fused_depth, fused_confidence).
    """
    assert mono_depth.shape == mvs_depth.shape == mono_conf.shape == mvs_conf.shape

    total_conf = mono_conf + mvs_conf
    fused = (mono_depth * mono_conf + mvs_depth * mvs_conf) / (total_conf + 1e-8)

    mvs_failed = mvs_conf < mvs_fail_threshold
    fused[mvs_failed] = mono_depth[mvs_failed]

    # where MVS is reliable, still let it dominate the weighting further
    mvs_reliable = mvs_conf > mvs_conf_threshold
    fused[mvs_reliable] = (
        mono_depth[mvs_reliable] * (1 - mvs_conf[mvs_reliable])
        + mvs_depth[mvs_reliable] * mvs_conf[mvs_reliable]
    )

    fused_conf = np.maximum(mono_conf, mvs_conf)
    return fused.astype(np.float32), fused_conf.astype(np.float32)


def inverse_variance_fusion(
    depth_a: np.ndarray, sigma_a: np.ndarray, depth_b: np.ndarray, sigma_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Classical Gaussian (inverse-variance) fusion of two depth estimates with
    known standard deviations. Returns (fused_depth, fused_sigma).
    """
    var_a = np.clip(sigma_a, 1e-6, None) ** 2
    var_b = np.clip(sigma_b, 1e-6, None) ** 2

    w_a = 1.0 / var_a
    w_b = 1.0 / var_b

    fused = (depth_a * w_a + depth_b * w_b) / (w_a + w_b)
    fused_var = 1.0 / (w_a + w_b)
    return fused.astype(np.float32), np.sqrt(fused_var).astype(np.float32)
