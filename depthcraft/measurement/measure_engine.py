"""Point-to-point distance measurement with rigorous uncertainty propagation.

This is the "virtual tape measure" -- Section 5.3 of the spec, implemented
exactly (first-order error propagation through the Euclidean distance
function), plus the click -> ray-cast -> measure orchestration.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from depthcraft.measurement.ray_caster import pick_point_nearest_neighbor, pick_point_on_mesh


@dataclass
class MeasurementResult:
    distance_m: float
    uncertainty_1sigma_m: float
    uncertainty_95ci_m: float
    point_a: np.ndarray
    point_b: np.ndarray

    def __str__(self) -> str:
        return (
            f"Distance: {self.distance_m:.2f} m +/- {self.uncertainty_1sigma_m:.2f} m "
            f"(95% CI: {self.distance_m:.2f} +/- {self.uncertainty_95ci_m:.2f} m)"
        )


def measure_with_uncertainty(
    p1: np.ndarray, p2: np.ndarray, sigma1: np.ndarray, sigma2: np.ndarray
) -> MeasurementResult:
    """Compute 3D Euclidean distance and its propagated 1-sigma uncertainty.

    Args:
        p1, p2: (3,) 3D points (meters).
        sigma1, sigma2: (3,) per-axis standard deviations (meters) at each point,
            typically derived from the depth uncertainty map (see
            depth/uncertainty_estimator.py) projected into xyz via the pinhole model.

    Error propagation: for d = ||p1 - p2||,
        sigma_d^2 = sum_i [(p1_i - p2_i)/d]^2 * (sigma1_i^2 + sigma2_i^2)
    """
    diff = p1 - p2
    distance = float(np.linalg.norm(diff))
    if distance < 1e-9:
        return MeasurementResult(0.0, 0.0, 0.0, p1, p2)

    derivatives = diff / distance
    variance = float(np.sum(derivatives**2 * (sigma1**2 + sigma2**2)))
    uncertainty = float(np.sqrt(variance))

    return MeasurementResult(
        distance_m=distance,
        uncertainty_1sigma_m=uncertainty,
        uncertainty_95ci_m=2 * uncertainty,
        point_a=p1,
        point_b=p2,
    )


def depth_uncertainty_to_xyz_sigma(
    pixel: tuple[float, float], depth_sigma_m: float, intrinsics: np.ndarray
) -> np.ndarray:
    """Project a scalar depth-axis uncertainty into approximate per-axis (x,y,z)
    standard deviations using the pinhole model's local Jacobian. Lateral (x,y)
    uncertainty scales with depth/focal-length; z uncertainty is the depth
    uncertainty itself (dominant term for typical mono-depth noise).
    """
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    px, py = pixel
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # crude first-order lateral scaling: dx/dz = (px - cx)/fx
    sigma_x = abs((px - cx) / fx) * depth_sigma_m + 0.001
    sigma_y = abs((py - cy) / fy) * depth_sigma_m + 0.001
    sigma_z = depth_sigma_m
    return np.array([sigma_x, sigma_y, sigma_z])


def click_to_measure(
    click_a_px: tuple[float, float],
    click_b_px: tuple[float, float],
    intrinsics: np.ndarray,
    pose_w2c: np.ndarray,
    uncertainty_map: np.ndarray | None = None,
    mesh=None,
    points: np.ndarray | None = None,
) -> MeasurementResult:
    """End-to-end: two pixel clicks -> ray cast -> 3D points -> distance + uncertainty.

    Provide either `mesh` (preferred, more accurate) or a raw `points` cloud.
    """
    if mesh is not None:
        p1 = pick_point_on_mesh(mesh, click_a_px, intrinsics, pose_w2c)
        p2 = pick_point_on_mesh(mesh, click_b_px, intrinsics, pose_w2c)
    elif points is not None:
        p1 = pick_point_nearest_neighbor(points, click_a_px, intrinsics, pose_w2c)
        p2 = pick_point_nearest_neighbor(points, click_b_px, intrinsics, pose_w2c)
    else:
        raise ValueError("Must provide either `mesh` or `points`.")

    if p1 is None or p2 is None:
        raise RuntimeError("Ray cast missed geometry for one or both clicks.")

    if uncertainty_map is not None:
        from depthcraft.depth.uncertainty_estimator import uncertainty_at_pixel

        sigma_a_depth = uncertainty_at_pixel(uncertainty_map, int(click_a_px[0]), int(click_a_px[1]))
        sigma_b_depth = uncertainty_at_pixel(uncertainty_map, int(click_b_px[0]), int(click_b_px[1]))
        sigma1 = depth_uncertainty_to_xyz_sigma(click_a_px, sigma_a_depth, intrinsics)
        sigma2 = depth_uncertainty_to_xyz_sigma(click_b_px, sigma_b_depth, intrinsics)
    else:
        # conservative default: 2% of estimated depth as 1-sigma
        sigma1 = np.abs(p1) * 0.02 + 0.005
        sigma2 = np.abs(p2) * 0.02 + 0.005

    return measure_with_uncertainty(p1, p2, sigma1, sigma2)
