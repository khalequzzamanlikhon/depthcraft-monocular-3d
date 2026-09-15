"""Camera pose utilities shared by the SfM and MVS stages."""
from __future__ import annotations

import numpy as np


def quaternion_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Standard quaternion -> 3x3 rotation matrix (Hamilton convention)."""
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy**2 + qz**2), 2 * (qx*qy - qz*qw), 2 * (qx*qz + qy*qw)],
        [2 * (qx*qy + qz*qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy*qz - qx*qw)],
        [2 * (qx*qz - qy*qw), 2 * (qy*qz + qx*qw), 1 - 2 * (qx**2 + qy**2)],
    ])


def pose_to_matrix(translation: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    """Build a 4x4 world-to-camera matrix from translation + quaternion (xyzw)."""
    r = quaternion_to_rotation_matrix(*quaternion)
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = translation
    return m


def relative_pose(pose_a: np.ndarray, pose_b: np.ndarray) -> np.ndarray:
    """4x4 relative transform from camera A to camera B."""
    return pose_b @ np.linalg.inv(pose_a)


def camera_frustum_lines(pose_w2c: np.ndarray, intrinsics: np.ndarray, scale: float = 0.1):
    """Return 8 line-segment endpoints (as Nx2 index pairs into 5 corner points)
    describing a camera frustum wireframe, for Open3D visualization."""
    cam_to_world = np.linalg.inv(pose_w2c)
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    corners_cam = np.array([
        [0, 0, 0],
        [-cx / fx, -cy / fy, 1],
        [cx / fx, -cy / fy, 1],
        [cx / fx, cy / fy, 1],
        [-cx / fx, cy / fy, 1],
    ]) * scale

    corners_world = (cam_to_world[:3, :3] @ corners_cam.T).T + cam_to_world[:3, 3]
    lines = np.array([[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]])
    return corners_world, lines
