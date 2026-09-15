"""Object detection + oriented 3D bounding box measurement (Phase 6 bonus)."""
from __future__ import annotations

import numpy as np


def detect_objects_2d(image: np.ndarray, weights: str = "yolov8x.pt") -> list[dict]:
    """Run YOLOv8 object detection on a source image. Returns boxes + class names."""
    from ultralytics import YOLO

    model = YOLO(weights)
    results = model.predict(image, verbose=False)[0]

    detections = []
    for box, cls_id, conf in zip(
        results.boxes.xyxy.cpu().numpy(),
        results.boxes.cls.cpu().numpy().astype(int),
        results.boxes.conf.cpu().numpy(),
    ):
        detections.append({
            "bbox_px": box.tolist(),
            "class_name": results.names[cls_id],
            "confidence": float(conf),
        })
    return detections


def compute_oriented_bbox_3d(points_in_object: np.ndarray) -> dict:
    """Given the subset of a point cloud belonging to one detected object
    (e.g. via 2D-box back-projection or 3D clustering), fit an oriented
    bounding box and report L x W x H using Open3D's OBB estimator.
    """
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_in_object)
    obb = pcd.get_oriented_bounding_box()

    extent = np.asarray(obb.extent)
    dims_sorted = np.sort(extent)[::-1]  # L >= W >= H convention
    return {
        "length_m": float(dims_sorted[0]),
        "width_m": float(dims_sorted[1]),
        "height_m": float(dims_sorted[2]),
        "center": np.asarray(obb.center).tolist(),
        "rotation": np.asarray(obb.R).tolist(),
    }


def points_in_2d_box(
    points: np.ndarray, bbox_px: list[float], intrinsics: np.ndarray, pose_w2c: np.ndarray
) -> np.ndarray:
    """Select the subset of a (world-frame) point cloud that reprojects inside a 2D detection box."""
    pts_cam = (pose_w2c[:3, :3] @ points.T + pose_w2c[:3, 3:4]).T
    in_front = pts_cam[:, 2] > 0
    pix = (intrinsics @ pts_cam.T).T
    pix = pix[:, :2] / np.clip(pix[:, 2:3], 1e-6, None)

    x1, y1, x2, y2 = bbox_px
    inside = (pix[:, 0] >= x1) & (pix[:, 0] <= x2) & (pix[:, 1] >= y1) & (pix[:, 1] <= y2)
    return points[inside & in_front]
