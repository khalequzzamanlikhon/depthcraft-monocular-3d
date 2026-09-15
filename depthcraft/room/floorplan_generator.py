"""Top-down orthographic floor plan generation from detected floor + walls."""
from __future__ import annotations

import numpy as np


def project_to_floorplan(points: np.ndarray, floor_plane: tuple, resolution_m: float = 0.02):
    """Project a 3D point cloud onto the floor plane and rasterize a 2D occupancy
    grid -- the basis for a top-down floor plan sketch.
    """
    a, b, c, d = floor_plane
    normal = np.array([a, b, c])
    normal /= np.linalg.norm(normal)

    # project points onto plane, then express in a 2D basis on that plane
    dist = points @ normal + d
    projected = points - dist[:, None] * normal[None, :]

    # build an arbitrary orthonormal basis for the plane
    arbitrary = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, arbitrary)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)

    coords_2d = np.stack([projected @ u, projected @ v], axis=-1)

    min_xy = coords_2d.min(axis=0)
    grid_coords = ((coords_2d - min_xy) / resolution_m).astype(int)
    grid_w, grid_h = grid_coords.max(axis=0) + 1

    occupancy = np.zeros((grid_h, grid_w), dtype=np.uint8)
    occupancy[grid_coords[:, 1], grid_coords[:, 0]] = 255
    return occupancy, min_xy, resolution_m


def estimate_room_dimensions(points: np.ndarray, floor_plane: tuple) -> dict:
    """Coarse bounding-box room dimensions from the projected footprint + height range."""
    occupancy, _min_xy, res = project_to_floorplan(points, floor_plane)
    ys, xs = np.nonzero(occupancy)
    length_m = (xs.max() - xs.min()) * res if len(xs) else 0.0
    width_m = (ys.max() - ys.min()) * res if len(ys) else 0.0

    a, b, c, d = floor_plane
    normal = np.array([a, b, c]) / np.linalg.norm([a, b, c])
    heights = points @ normal + d
    height_m = float(np.percentile(heights, 99) - np.percentile(heights, 1))

    return {"length_m": float(length_m), "width_m": float(width_m), "height_m": height_m}
