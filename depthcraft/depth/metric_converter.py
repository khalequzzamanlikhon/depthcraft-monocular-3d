"""Scale/shift alignment between relative monocular depth and sparse SfM depth.

This is a fully real, dependency-light implementation of Section 5.1 of the
implementation guide (RANSAC replaces naive median scaling).
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import RANSACRegressor


def align_scale_ransac(
    mono_depth_map: np.ndarray,
    sfm_points_3d: np.ndarray,
    camera_intrinsics: np.ndarray,
    camera_pose_w2c: np.ndarray,
    residual_threshold: float = 0.1,
    min_samples: float = 0.1,
) -> tuple[float, float, np.ndarray]:
    """Align relative monocular depth to metric scale using sparse SfM points.

    Args:
        mono_depth_map: (H, W) relative depth from the mono estimator.
        sfm_points_3d: (N, 3) world-frame 3D points from COLMAP.
        camera_intrinsics: (3, 3) K matrix.
        camera_pose_w2c: (4, 4) world-to-camera transform for this frame.
        residual_threshold: RANSAC inlier residual in metric units (meters).
        min_samples: fraction (or int) of samples per RANSAC iteration.

    Returns:
        scale, shift, inlier_mask (over the points that projected inside the image).

    The model fit is: depth_metric = scale * mono_relative + shift.
    """
    if sfm_points_3d.shape[0] < 4:
        raise ValueError("Need at least 4 SfM points for a robust RANSAC fit.")

    points_cam = (camera_pose_w2c[:3, :3] @ sfm_points_3d.T + camera_pose_w2c[:3, 3:4]).T
    depths_mvs = points_cam[:, 2]

    pixels_h = (camera_intrinsics @ points_cam.T).T
    with np.errstate(invalid="ignore", divide="ignore"):
        pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
    pixels_int = np.round(pixels).astype(int)

    h, w = mono_depth_map.shape
    valid = (
        (pixels_int[:, 0] >= 0)
        & (pixels_int[:, 0] < w)
        & (pixels_int[:, 1] >= 0)
        & (pixels_int[:, 1] < h)
        & (depths_mvs > 1e-4)
        & np.isfinite(depths_mvs)
    )

    pixels_int = pixels_int[valid]
    depths_mvs = depths_mvs[valid]
    mono_samples = mono_depth_map[pixels_int[:, 1], pixels_int[:, 0]]

    if len(mono_samples) < 4:
        raise ValueError("Fewer than 4 valid re-projected points; cannot fit scale.")

    x = mono_samples.reshape(-1, 1)
    y = depths_mvs

    ransac = RANSACRegressor(min_samples=min_samples, residual_threshold=residual_threshold)
    ransac.fit(x, y)

    scale = float(ransac.estimator_.coef_[0])
    shift = float(ransac.estimator_.intercept_)
    inlier_mask = ransac.inlier_mask_

    # Optional least-squares refinement on inliers only, per spec section 5.1
    if inlier_mask.sum() >= 2:
        scale, shift = np.polyfit(mono_samples[inlier_mask], depths_mvs[inlier_mask], 1)
        scale, shift = float(scale), float(shift)

    return scale, shift, inlier_mask


def apply_scale(mono_depth_map: np.ndarray, scale: float, shift: float) -> np.ndarray:
    """Convert a full relative depth map to metric depth using fitted (scale, shift)."""
    return mono_depth_map * scale + shift


def align_scale_known_reference(
    mono_depth_map: np.ndarray,
    reference_pixel_box: tuple[int, int, int, int],
    reference_real_distance_m: float,
) -> tuple[float, float]:
    """Method A fallback from Phase 0: scale via a single known reference object
    (e.g. A4 paper / credit card) when no SfM points are available yet.

    reference_pixel_box: (x1, y1, x2, y2) region covering the reference object.
    reference_real_distance_m: known real-world depth of that object's plane.
    """
    x1, y1, x2, y2 = reference_pixel_box
    region = mono_depth_map[y1:y2, x1:x2]
    mono_mean = float(np.median(region))
    if mono_mean <= 0:
        raise ValueError("Reference region has non-positive relative depth.")
    scale = reference_real_distance_m / mono_mean
    shift = 0.0
    return scale, shift
