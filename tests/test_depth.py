import numpy as np

from depthcraft.depth.depth_anything_engine import DepthAnythingEngine, colorize_depth
from depthcraft.depth.metric_converter import align_scale_ransac, apply_scale
from depthcraft.depth.uncertainty_estimator import estimate_uncertainty_tta, uncertainty_at_pixel
from depthcraft.utils.geometry_utils import default_intrinsics


def _dummy_image(h=64, w=64):
    rng = np.random.default_rng(0)
    return (rng.random((h, w, 3)) * 255).astype(np.uint8)


def test_depth_engine_infers_valid_shape():
    engine = DepthAnythingEngine()
    image = _dummy_image()
    depth = engine.infer(image)
    assert depth.shape == image.shape[:2]
    assert np.all(np.isfinite(depth))
    assert depth.min() >= 0


def test_colorize_depth_shape():
    depth = np.random.rand(32, 32).astype(np.float32)
    rgb = colorize_depth(depth)
    assert rgb.shape == (32, 32, 3)
    assert rgb.dtype == np.uint8


def test_uncertainty_tta_shapes_match():
    engine = DepthAnythingEngine()
    image = _dummy_image()
    mean_depth, uncertainty = estimate_uncertainty_tta(engine, image, n_augmentations=3)
    assert mean_depth.shape == image.shape[:2]
    assert uncertainty.shape == image.shape[:2]
    assert np.all(uncertainty >= 0)


def test_uncertainty_at_pixel_local_average():
    unc = np.ones((10, 10), dtype=np.float32)
    unc[5, 5] = 5.0
    val = uncertainty_at_pixel(unc, 5, 5, window=1)
    assert val > 1.0  # local window should catch the spike


def test_align_scale_ransac_recovers_known_transform():
    h, w = 100, 100
    intrinsics = default_intrinsics(w, h)
    rng = np.random.default_rng(1)

    # Smooth (not per-pixel-iid) synthetic depth: a 1px rounding mismatch
    # between the test's sampling and the function's internal np.round()
    # should barely change the value, since real depth maps are smooth too.
    yy, xx = np.mgrid[0:h, 0:w]
    mono_depth = (0.4 + 0.3 * np.sin(xx / 15.0) + 0.3 * np.cos(yy / 20.0)).astype(np.float32)
    true_scale, true_shift = 3.0, 0.5

    n_pts = 200
    px = rng.uniform(10, w - 10, n_pts)
    py = rng.uniform(10, h - 10, n_pts)
    mono_vals = mono_depth[py.astype(int), px.astype(int)]
    metric_depth = true_scale * mono_vals + true_shift

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    x = (px - cx) * metric_depth / fx
    y = (py - cy) * metric_depth / fy
    points_cam = np.stack([x, y, metric_depth], axis=-1)

    pose_w2c = np.eye(4)
    scale, shift, inliers = align_scale_ransac(
        mono_depth, points_cam, intrinsics, pose_w2c, residual_threshold=0.05
    )

    assert abs(scale - true_scale) < 0.5
    assert inliers.sum() > n_pts * 0.5

    rescaled = apply_scale(mono_depth, scale, shift)
    assert rescaled.shape == mono_depth.shape
