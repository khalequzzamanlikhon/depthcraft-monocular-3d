"""Core geometry helpers: camera projection, back-projection, transforms.

These are pure-numpy so they work with zero heavy dependencies and are
unit-testable without a GPU.
"""
from __future__ import annotations

import numpy as np


def default_intrinsics(width: int, height: int, fov_deg: float = 60.0) -> np.ndarray:
    """Build a plausible pinhole intrinsics matrix when no calibration exists.

    fx = fy is derived from a horizontal FOV assumption. This is only a
    fallback -- real intrinsics should come from `measurement.calibrator`.
    """
    fx = (width / 2.0) / np.tan(np.radians(fov_deg / 2.0))
    fy = fx
    cx, cy = width / 2.0, height / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def depth_to_pointcloud(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    color: np.ndarray | None = None,
    depth_scale: float = 1.0,
    max_depth: float = 50.0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Back-project a dense metric depth map (H, W) into a camera-frame point cloud.

    Returns (points_Nx3, colors_Nx3_or_None).
    """
    h, w = depth.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.astype(np.float64) * depth_scale
    valid = (z > 0) & (z < max_depth) & np.isfinite(z)

    x = (us - cx) * z / fx
    y = (vs - cy) * z / fy

    pts = np.stack([x[valid], y[valid], z[valid]], axis=-1)
    cols = None
    if color is not None:
        cols = color.reshape(-1, color.shape[-1])[valid.reshape(-1)] / 255.0
    return pts, cols


def project_points(points_cam: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Project Nx3 camera-frame points to Nx2 pixel coordinates."""
    proj = (intrinsics @ points_cam.T).T
    return proj[:, :2] / proj[:, 2:3]


def world_to_camera(points_world: np.ndarray, pose_w2c: np.ndarray) -> np.ndarray:
    """Apply a 4x4 world-to-camera pose to Nx3 world points."""
    r, t = pose_w2c[:3, :3], pose_w2c[:3, 3]
    return (r @ points_world.T).T + t


def ray_from_pixel(px: float, py: float, intrinsics: np.ndarray) -> np.ndarray:
    """Unit ray direction (camera frame) through a pixel."""
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    d = np.array([(px - cx) / fx, (py - cy) / fy, 1.0])
    return d / np.linalg.norm(d)
