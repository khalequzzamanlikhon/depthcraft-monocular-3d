"""Floor/wall plane detection via RANSAC (Phase 6 bonus)."""
from __future__ import annotations

import numpy as np


def detect_floor_plane(pcd, distance_threshold: float = 0.02, ransac_n: int = 3, num_iterations: int = 2000):
    """RANSAC plane fit; assumes the floor is the largest near-horizontal plane
    at the lowest elevation. Returns (plane_model[a,b,c,d], inlier_indices).
    """
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold, ransac_n=ransac_n, num_iterations=num_iterations
    )
    a, b, c, _d = plane_model
    normal = np.array([a, b, c])
    normal /= np.linalg.norm(normal)

    # Open3D convention: Y is often "up" after alignment, but this is scene
    # dependent -- verify horizontality via normal's dominant axis instead of
    # assuming a fixed world-up vector.
    if abs(normal[1]) < 0.85:
        import warnings

        warnings.warn(
            "Detected largest plane is not strongly horizontal -- verify the "
            "point cloud's up-axis before trusting this as the floor."
        )
    return plane_model, inliers


def detect_wall_planes(pcd, floor_normal: np.ndarray, n_walls: int = 4, distance_threshold: float = 0.02):
    """Iteratively RANSAC-fit planes orthogonal to the floor normal (walls),
    removing each plane's inliers before searching for the next.
    """
    remaining = pcd
    walls = []
    for _ in range(n_walls):
        if len(remaining.points) < 100:
            break
        plane_model, inliers = remaining.segment_plane(
            distance_threshold=distance_threshold, ransac_n=3, num_iterations=1000
        )
        normal = np.array(plane_model[:3])
        normal /= np.linalg.norm(normal)
        if abs(np.dot(normal, floor_normal)) < 0.3:  # roughly orthogonal to floor
            walls.append((plane_model, remaining.select_by_index(inliers)))
        remaining = remaining.select_by_index(inliers, invert=True)
    return walls
