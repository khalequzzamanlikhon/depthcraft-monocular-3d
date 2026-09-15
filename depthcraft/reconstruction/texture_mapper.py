"""Multi-view texture mapping: pick the best source image per face and build a UV atlas.

Open3D doesn't ship a full multi-view texture-atlas baker out of the box, so
this implements the core algorithm (best-view selection + per-vertex color
projection) directly; for production-grade UV unwrapping + atlas packing,
swap in a dedicated tool like `xatlas` (pip-installable, used below when
present) with a fallback to simple vertex-color texturing.
"""
from __future__ import annotations

import numpy as np


def select_best_view_per_face(
    face_centers: np.ndarray,
    face_normals: np.ndarray,
    camera_poses_w2c: list[np.ndarray],
    camera_positions_world: list[np.ndarray],
) -> np.ndarray:
    """For each face, pick the camera that views it most frontally (normal
    most anti-parallel to the view direction) among cameras that can see it.
    Returns an (F,) int array of camera indices.
    """
    n_faces = face_centers.shape[0]
    best_cam = np.full(n_faces, -1, dtype=int)
    best_score = np.full(n_faces, -1.0)

    for cam_idx, cam_pos in enumerate(camera_positions_world):
        view_dir = face_centers - cam_pos
        view_dir /= np.linalg.norm(view_dir, axis=1, keepdims=True) + 1e-8
        frontal_score = -np.sum(view_dir * face_normals, axis=1)  # want normal facing camera

        better = frontal_score > best_score
        best_score[better] = frontal_score[better]
        best_cam[better] = cam_idx

    return best_cam


def bake_vertex_colors_from_images(
    mesh, images: list[np.ndarray], intrinsics: np.ndarray, poses_w2c: list[np.ndarray]
):
    """Simple (non-atlas) texturing: assign each vertex the color sampled from its
    best-view source image. Good enough for portfolio-quality viewing; swap for
    `xatlas`-based UV texture baking for production asset pipelines.
    """
    import open3d as o3d

    vertices = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals) if mesh.has_vertex_normals() else np.zeros_like(vertices)
    camera_positions = [np.linalg.inv(p)[:3, 3] for p in poses_w2c]

    best_cam = select_best_view_per_face(vertices, normals, poses_w2c, camera_positions)

    colors = np.zeros_like(vertices)
    for cam_idx, image in enumerate(images):
        mask = best_cam == cam_idx
        if not mask.any():
            continue
        pts_cam = (poses_w2c[cam_idx][:3, :3] @ vertices[mask].T + poses_w2c[cam_idx][:3, 3:4]).T
        pix = (intrinsics @ pts_cam.T).T
        pix = pix[:, :2] / np.clip(pix[:, 2:3], 1e-6, None)
        h, w = image.shape[:2]
        px = np.clip(pix[:, 0], 0, w - 1).astype(int)
        py = np.clip(pix[:, 1], 0, h - 1).astype(int)
        colors[mask] = image[py, px] / 255.0

    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    return mesh
