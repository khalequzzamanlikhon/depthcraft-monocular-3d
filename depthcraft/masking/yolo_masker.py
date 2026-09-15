"""YOLOv8-seg per-frame masking fallback for dynamic objects (persons, vehicles).

Simpler and more robust to set up than SAM 2 -- runs per frame independently
(no cross-frame identity tracking), which is enough to exclude movers before
COLMAP feature extraction.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from depthcraft.utils.logger import get_logger

log = get_logger(__name__)

# COCO class ids commonly responsible for scene motion artifacts
DYNAMIC_CLASS_IDS = {0, 1, 2, 3, 5, 7}  # person, bicycle, car, motorcycle, bus, truck


class YOLOMasker:
    def __init__(self, weights: str = "yolov8x-seg.pt", dynamic_classes: set[int] | None = None):
        self.dynamic_classes = dynamic_classes or DYNAMIC_CLASS_IDS
        self._model: Any = None
        self._available = self._try_load(weights)

    def _try_load(self, weights: str) -> bool:
        try:
            from ultralytics import YOLO

            self._model = YOLO(weights)
            log.info(f"Loaded YOLO segmentation model: {weights}")
            return True
        except Exception as e:  # noqa: BLE001
            log.warning(f"Ultralytics/YOLO unavailable ({e!r}); masking will pass frames through unmodified.")
            return False

    @property
    def available(self) -> bool:
        return self._available

    def mask_frame(self, image: np.ndarray) -> np.ndarray:
        """Return a binary mask (1 = static/keep, 0 = dynamic/exclude) for one RGB frame."""
        h, w = image.shape[:2]
        if not self._available:
            return np.ones((h, w), dtype=np.uint8)

        results = self._model.predict(image, verbose=False)[0]
        mask = np.ones((h, w), dtype=np.uint8)
        if results.masks is None:
            return mask

        classes = results.boxes.cls.cpu().numpy().astype(int)
        seg_masks = results.masks.data.cpu().numpy()  # (N, h', w')
        for cls_id, seg in zip(classes, seg_masks):
            if cls_id in self.dynamic_classes:
                seg_resized = _resize_mask(seg, (h, w))
                mask[seg_resized > 0.5] = 0
        return mask


def _resize_mask(mask: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    import cv2

    return cv2.resize(mask.astype(np.float32), (out_hw[1], out_hw[0]), interpolation=cv2.INTER_LINEAR)
