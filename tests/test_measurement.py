import numpy as np

from depthcraft.measurement.measure_engine import (
    depth_uncertainty_to_xyz_sigma,
    measure_with_uncertainty,
)
from depthcraft.utils.geometry_utils import default_intrinsics


def test_measure_with_uncertainty_basic_distance():
    p1 = np.array([0.0, 0.0, 1.0])
    p2 = np.array([0.0, 0.0, 4.0])
    sigma = np.array([0.01, 0.01, 0.01])

    result = measure_with_uncertainty(p1, p2, sigma, sigma)
    assert abs(result.distance_m - 3.0) < 1e-9
    assert result.uncertainty_1sigma_m > 0
    assert result.uncertainty_95ci_m == 2 * result.uncertainty_1sigma_m


def test_measure_with_uncertainty_zero_distance():
    p = np.array([1.0, 1.0, 1.0])
    result = measure_with_uncertainty(p, p, np.zeros(3), np.zeros(3))
    assert result.distance_m == 0.0
    assert result.uncertainty_1sigma_m == 0.0


def test_measure_uncertainty_scales_with_input_sigma():
    p1 = np.array([0.0, 0.0, 1.0])
    p2 = np.array([1.0, 0.0, 1.0])

    small = measure_with_uncertainty(p1, p2, np.full(3, 0.01), np.full(3, 0.01))
    large = measure_with_uncertainty(p1, p2, np.full(3, 0.1), np.full(3, 0.1))

    assert large.uncertainty_1sigma_m > small.uncertainty_1sigma_m
    assert abs(large.distance_m - small.distance_m) < 1e-9  # distance unaffected by sigma


def test_depth_uncertainty_to_xyz_sigma_center_pixel():
    intrinsics = default_intrinsics(640, 480)
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    sigma = depth_uncertainty_to_xyz_sigma((cx, cy), depth_sigma_m=0.05, intrinsics=intrinsics)
    assert sigma[2] == 0.05  # z-uncertainty equals input depth sigma at principal point
    assert sigma[0] < 0.01 and sigma[1] < 0.01  # near-zero lateral term at image center
