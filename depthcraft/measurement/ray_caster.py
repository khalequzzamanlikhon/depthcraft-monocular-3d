"""Ray casting for click-to-measure: pixel click -> 3D point in the point cloud/mesh.

Uses Open3D's raycasting scene (built on Intel Embree) for real geometric
intersection against the reconstructed mesh, with a nearest-neighbor
fallback against the raw point cloud when only points (no mesh) exist yet.
"""
from __future__ import annotations

import numpy as np

from depthcraft.utils.geometry_utils import ray_from_pixel


def pick_point_on_mesh(mesh, pixel: tuple[float, float], intrinsics: np.ndarray, pose_w2c: np.ndarray):
    """Cast a ray from a clicked pixel through the camera into the scene mesh.
    Returns the 3D world-space hit point, or None if no intersection.
    """
    import open3d as o3d

    scene = o3d.t.geometry.RaycastingScene()
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene.add_triangles(mesh_t)

    cam_to_world = np.linalg.inv(pose_w2c)
    ray_dir_cam = ray_from_pixel(pixel[0], pixel[1], intrinsics)
    ray_dir_world = cam_to_world[:3, :3] @ ray_dir_cam
    ray_origin_world = cam_to_world[:3, 3]

    rays = o3d.core.Tensor(
        [[*ray_origin_world, *ray_dir_world]], dtype=o3d.core.Dtype.Float32
    )
    result = scene.cast_rays(rays)
    t_hit = result["t_hit"].numpy()[0]

    if not np.isfinite(t_hit):
        return None
    return ray_origin_world + t_hit * ray_dir_world


def pick_point_nearest_neighbor(
    points: np.ndarray, pixel: tuple[float, float], intrinsics: np.ndarray, pose_w2c: np.ndarray,
    max_ray_distance: float = 20.0, n_samples: int = 400,
):
    """Fallback for raw point clouds (no mesh yet): sample points along the
    click ray and return the closest existing point cloud point to that ray.
    """
    from scipy.spatial import cKDTree

    cam_to_world = np.linalg.inv(pose_w2c)
    ray_dir_cam = ray_from_pixel(pixel[0], pixel[1], intrinsics)
    ray_dir_world = cam_to_world[:3, :3] @ ray_dir_cam
    ray_origin_world = cam_to_world[:3, 3]

    ts = np.linspace(0.05, max_ray_distance, n_samples)
    ray_samples = ray_origin_world[None, :] + ts[:, None] * ray_dir_world[None, :]

    tree = cKDTree(points)
    dists, idxs = tree.query(ray_samples)
    best = np.argmin(dists)
    if dists[best] > 0.05:  # 5cm tolerance -- ray didn't come near any real point
        return None
    return points[idxs[best]]
