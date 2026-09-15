import numpy as np

from depthcraft.mvs.confidence_mapper import normalize_confidence
from depthcraft.mvs.dense_fusion import confidence_fusion, inverse_variance_fusion
from depthcraft.utils.geometry_utils import default_intrinsics, depth_to_pointcloud


def test_depth_to_pointcloud_filters_invalid():
    depth = np.array([[1.0, 0.0], [-1.0, 2.0]], dtype=np.float32)
    intrinsics = default_intrinsics(2, 2)
    points, colors = depth_to_pointcloud(depth, intrinsics)
    # only the two positive, finite, in-range depth pixels should survive
    assert points.shape[0] == 2
    assert colors is None


def test_confidence_fusion_falls_back_to_mono_when_mvs_fails():
    mono = np.full((4, 4), 2.0, dtype=np.float32)
    mvs = np.full((4, 4), 5.0, dtype=np.float32)
    mono_conf = np.full((4, 4), 0.8, dtype=np.float32)
    mvs_conf = np.zeros((4, 4), dtype=np.float32)  # MVS totally failed

    fused, _fused_conf = confidence_fusion(mono, mono_conf, mvs, mvs_conf)
    np.testing.assert_allclose(fused, mono)


def test_confidence_fusion_trusts_confident_mvs():
    mono = np.full((4, 4), 2.0, dtype=np.float32)
    mvs = np.full((4, 4), 5.0, dtype=np.float32)
    mono_conf = np.full((4, 4), 0.2, dtype=np.float32)
    mvs_conf = np.full((4, 4), 0.95, dtype=np.float32)

    fused, _ = confidence_fusion(mono, mono_conf, mvs, mvs_conf)
    assert np.all(fused > 4.0)  # should lean heavily toward MVS depth


def test_inverse_variance_fusion_weights_lower_variance_more():
    depth_a = np.array([1.0])
    depth_b = np.array([3.0])
    sigma_a = np.array([0.01])  # very confident
    sigma_b = np.array([1.0])   # very uncertain

    fused, fused_sigma = inverse_variance_fusion(depth_a, sigma_a, depth_b, sigma_b)
    assert abs(fused[0] - 1.0) < 0.1  # should land close to the confident estimate
    assert fused_sigma[0] < sigma_a[0] * 2  # combined uncertainty shouldn't blow up


def test_normalize_confidence_range():
    raw = np.array([0.0, 5.0, 10.0, 100.0])
    norm = normalize_confidence(raw)
    assert norm.min() >= 0.0
    assert norm.max() <= 1.0
