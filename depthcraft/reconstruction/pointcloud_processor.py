"""Point cloud cleaning & preparation using Open3D (Phase 4, Path A step 1-2).

Fully real -- Open3D provides all of this natively; we just sequence it
sensibly and expose clean, testable function signatures.
"""
from __future__ import annotations

import numpy as np


def numpy_to_o3d(points: np.ndarray, colors: np.ndarray | None = None):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1))
    return pcd


def o3d_to_numpy(pcd) -> tuple[np.ndarray, np.ndarray | None]:
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    return points, colors


def clean_pointcloud(
    pcd,
    voxel_size: float = 0.005,
    statistical_nb_neighbors: int = 20,
    statistical_std_ratio: float = 2.0,
    radius_nb_points: int = 16,
    radius: float | None = None,
):
    """Voxel downsample + statistical outlier removal + radius outlier removal,
    matching Phase 4 Path A step 1. Order: downsample first for speed, then filter.

    `radius=None` (default) auto-derives a sensible radius from the point
    cloud's own median nearest-neighbor spacing (2.5x) rather than a fixed
    5cm, which otherwise silently zeroes out sparse clouds -- e.g. a
    single-image MVP reconstruction, or anything less dense than a full
    multi-view capture. Pass an explicit `radius` to override.
    """
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    if len(pcd_down.points) < statistical_nb_neighbors:
        return pcd_down  # too sparse to meaningfully filter; return as-is

    pcd_stat, _ = pcd_down.remove_statistical_outlier(
        nb_neighbors=statistical_nb_neighbors, std_ratio=statistical_std_ratio
    )
    if len(pcd_stat.points) < radius_nb_points:
        return pcd_stat

    if radius is None:
        nn_dists = np.asarray(pcd_stat.compute_nearest_neighbor_distance())
        radius = float(np.median(nn_dists)) * 2.5 if len(nn_dists) else 0.05

    pcd_clean, _ = pcd_stat.remove_radius_outlier(nb_points=radius_nb_points, radius=radius)
    return pcd_clean if len(pcd_clean.points) > 0 else pcd_stat


def estimate_normals(pcd, camera_location: np.ndarray | None = None, radius: float = 0.1, max_nn: int = 30):
    """Estimate + consistently orient normals toward the camera (required for Poisson).

    Guards against degenerate/too-small point clouds (e.g. after aggressive
    outlier removal on noisy input) where normal estimation and orientation
    would otherwise raise or silently produce zero normals.
    """
    import open3d as o3d

    if len(pcd.points) < 10:
        return pcd  # not enough points for a meaningful normal estimate

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    if not pcd.has_normals() or len(pcd.normals) == 0:
        return pcd

    if camera_location is not None:
        pcd.orient_normals_towards_camera_location(camera_location)
    else:
        try:
            pcd.orient_normals_consistent_tangent_plane(k=min(30, len(pcd.points) - 1))
        except RuntimeError:
            pass  # orientation is best-effort; unoriented normals still work for many downstream steps
    return pcd


def full_cleaning_pipeline(
    points: np.ndarray,
    colors: np.ndarray | None = None,
    voxel_size: float = 0.005,
    camera_location: np.ndarray | None = None,
):
    """Convenience wrapper: numpy -> clean -> normals -> back to o3d point cloud."""
    pcd = numpy_to_o3d(points, colors)
    pcd = clean_pointcloud(pcd, voxel_size=voxel_size)
    pcd = estimate_normals(pcd, camera_location=camera_location)
    return pcd
