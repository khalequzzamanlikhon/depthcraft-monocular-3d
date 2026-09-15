"""Camera intrinsic calibration using AprilTags (Phase 0 step 3).

Uses OpenCV's built-in ArUco/AprilTag detector (cv2.aruco, available in
opencv-contrib and modern opencv-python builds) so no extra dependency
beyond what's already in pyproject.toml is strictly required. Falls back
to `pupil-apriltags` if the user has it installed and prefers it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from depthcraft.utils.logger import get_logger

log = get_logger(__name__)


class AprilTagCalibrator:
    def __init__(self, tag_family: str = "DICT_APRILTAG_36h11"):
        self.tag_family = tag_family

    def detect_tags(self, image: np.ndarray) -> list[dict]:
        import cv2

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.tag_family))
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, _ = detector.detectMarkers(gray)

        detections = []
        if ids is not None:
            for c, i in zip(corners, ids.flatten()):
                detections.append({"id": int(i), "corners": c.reshape(4, 2)})
        return detections

    def calibrate_from_images(
        self,
        image_paths: list[str | Path],
        board_tag_size_m: float = 0.05,
        board_layout: dict[int, np.ndarray] | None = None,
    ) -> dict:
        """Run OpenCV camera calibration from multiple views of an AprilTag board.

        board_layout maps tag_id -> 3D corner positions (4,3) in a shared board
        frame (meters). If not supplied, assumes a single isolated tag of size
        `board_tag_size_m` and estimates intrinsics via `calibrateCamera` with a
        planar single-tag object model (adequate for the Phase 0 MVP; for a
        full multi-tag board, generate board_layout from your printed grid).
        """
        import cv2
        from PIL import Image

        object_points_list = []
        image_points_list = []
        img_shape = None

        half = board_tag_size_m / 2
        default_obj_pts = np.array(
            [[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]],
            dtype=np.float32,
        )

        for p in image_paths:
            img = np.array(Image.open(p).convert("RGB"))
            img_shape = img.shape[:2]
            detections = self.detect_tags(img)
            if not detections:
                log.warning(f"No AprilTags detected in {p}, skipping.")
                continue

            for det in detections:
                obj_pts = (
                    board_layout[det["id"]] if board_layout and det["id"] in board_layout
                    else default_obj_pts
                )
                object_points_list.append(obj_pts.astype(np.float32))
                image_points_list.append(det["corners"].astype(np.float32))

        if len(object_points_list) < 5:
            raise ValueError(
                f"Only {len(object_points_list)} valid tag detections across "
                f"{len(image_paths)} images; need >=5-10 for a stable calibration."
            )

        assert img_shape is not None, "at least one valid detection implies img_shape was set"
        h, w = img_shape
        ret, k, dist, _rvecs, _tvecs = cv2.calibrateCamera(
            object_points_list, image_points_list, (w, h), None, None
        )

        log.info(f"Calibration RMS reprojection error: {ret:.4f} px")
        return {"intrinsics": k, "dist_coeffs": dist, "rms_error": float(ret)}
