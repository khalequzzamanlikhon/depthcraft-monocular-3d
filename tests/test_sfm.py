import numpy as np

from depthcraft.sfm.feature_extractor import FeatureExtractor
from depthcraft.sfm.matcher import FeatureMatcher
from depthcraft.sfm.pose_estimator import (
    pose_to_matrix,
    quaternion_to_rotation_matrix,
    relative_pose,
)


def _checkerboard_image(size=128):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    step = 16
    for i in range(0, size, step):
        for j in range(0, size, step):
            if (i // step + j // step) % 2 == 0:
                img[i:i + step, j:j + step] = 255
    return img


def _textured_image(size=128, seed=0):
    """A textured (non-repeating) image -- unlike a checkerboard, ORB
    descriptors here are distinctive enough for the ratio test to accept
    self-matches, which is what this fixture is used to verify.
    """
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (size // 8, size // 8, 3), dtype=np.uint8)
    img = np.kron(base, np.ones((8, 8, 1), dtype=np.uint8))
    return img[:size, :size]


def test_orb_feature_extraction_finds_keypoints():
    extractor = FeatureExtractor(backend="orb", n_features=200)
    image = _checkerboard_image()
    feats = extractor.extract(image)
    assert extractor.backend == "orb"
    assert feats["keypoints"].shape[0] > 0
    assert feats["descriptors"] is not None


def test_bf_matcher_matches_identical_image_features():
    extractor = FeatureExtractor(backend="orb", n_features=200)
    image = _textured_image()
    feats_a = extractor.extract(image)
    feats_b = extractor.extract(image)

    matcher = FeatureMatcher(backend="bf")
    matches = matcher.match(feats_a, feats_b)
    assert matches.shape[1] == 2
    assert matches.shape[0] > 0  # identical image should self-match well


def test_quaternion_to_rotation_matrix_identity():
    r = quaternion_to_rotation_matrix(0, 0, 0, 1)
    np.testing.assert_allclose(r, np.eye(3), atol=1e-6)


def test_pose_to_matrix_and_relative_pose():
    pose_a = pose_to_matrix(np.array([0, 0, 0]), np.array([0, 0, 0, 1]))
    pose_b = pose_to_matrix(np.array([1, 0, 0]), np.array([0, 0, 0, 1]))
    rel = relative_pose(pose_a, pose_b)
    np.testing.assert_allclose(rel[:3, 3], [1, 0, 0], atol=1e-6)
