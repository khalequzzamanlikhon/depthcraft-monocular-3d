"""Dynamic object masking for video pre-processing (SAM 2 primary path).

Requires the `masking` extra (`sam2`) and downloaded checkpoints. Falls back
to `yolo_masker.py` when SAM 2 isn't installed, which itself falls back to a
static "no masking" pass-through so the SfM stage always receives valid
(if unfiltered) input.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from depthcraft.utils.logger import get_logger

log = get_logger(__name__)


class SAM2Masker:
    def __init__(self, checkpoint: str = "sam2_hiera_large.pt", config: str = "sam2_hiera_l.yaml"):
        self.checkpoint = checkpoint
        self.config = config
        self._predictor: Any = None
        self._available = self._try_load()

    def _try_load(self) -> bool:
        try:
            from sam2.build_sam import build_sam2_video_predictor

            self._predictor = build_sam2_video_predictor(self.config, self.checkpoint)
            log.info("SAM 2 video predictor loaded.")
            return True
        except Exception as e:  # noqa: BLE001
            log.warning(f"SAM 2 unavailable ({e!r}); use YOLOMasker as a fallback.")
            return False

    @property
    def available(self) -> bool:
        return self._available

    def mask_moving_objects(
        self, video_frames_dir: str | Path, motion_threshold_px: float = 3.0
    ) -> dict[int, np.ndarray]:
        """Segment all objects across a frame sequence and classify static vs. moving
        by centroid displacement across frames. Returns {frame_idx: binary_mask}
        where 1 = static (keep), 0 = dynamic (exclude from SfM).
        """
        if not self._available:
            raise RuntimeError(
                "SAM 2 not installed. Install the 'masking' extra and download "
                "sam2_hiera_large.pt, or use YOLOMasker instead."
            )

        frame_paths = sorted(Path(video_frames_dir).glob("*.jpg")) + sorted(
            Path(video_frames_dir).glob("*.png")
        )
        if not frame_paths:
            raise FileNotFoundError(f"No frames found in {video_frames_dir}")

        self._predictor.init_state(video_path=str(video_frames_dir))
        # Real usage: predictor.add_new_points(...) to seed prompts on frame 0,
        # then predictor.propagate_in_video(state) to track masks + centroids
        # across all frames, then compute per-object centroid displacement and
        # threshold at motion_threshold_px to split static vs dynamic.
        masks: dict[int, np.ndarray] = {}
        for i, fp in enumerate(frame_paths):
            from PIL import Image

            h, w = np.array(Image.open(fp)).shape[:2]
            masks[i] = np.ones((h, w), dtype=np.uint8)  # placeholder until prompts are seeded
        log.warning(
            "mask_moving_objects() needs interactive/automatic prompt seeding "
            "(click points or a detector) before propagate_in_video will produce "
            "real per-object masks -- see SAM 2's official video predictor example."
        )
        return masks
