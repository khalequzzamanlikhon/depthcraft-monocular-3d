"""Descriptor matching: SuperGlue for learned features, ratio-test BFMatcher for ORB.

The ORB path is fully real and requires nothing beyond OpenCV.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from depthcraft.utils.logger import get_logger

log = get_logger(__name__)


class FeatureMatcher:
    def __init__(self, backend: str = "superglue", ratio_thresh: float = 0.75):
        self.ratio_thresh = ratio_thresh
        self._superglue: Any = None
        self.backend = self._resolve_backend(backend)

    def _resolve_backend(self, backend: str) -> str:
        if backend == "bf":
            return "bf"
        try:
            import torch  # noqa: F401
            from lightglue import SuperGlue  # community SuperGlue/LightGlue port

            self._superglue = SuperGlue({"weights": "outdoor"}).eval()
            return "superglue"
        except Exception as e:  # noqa: BLE001
            log.warning(f"SuperGlue unavailable ({e!r}); falling back to BFMatcher + ratio test.")
            return "bf"

    def match(self, feats_a: dict, feats_b: dict) -> np.ndarray:
        """Returns (M, 2) int array of matched indices [idx_in_a, idx_in_b]."""
        if self.backend == "bf":
            return self._match_bf(feats_a, feats_b)
        return self._match_superglue(feats_a, feats_b)

    def _match_bf(self, feats_a: dict, feats_b: dict) -> np.ndarray:
        import cv2

        da, db = feats_a["descriptors"], feats_b["descriptors"]
        if da is None or db is None or len(da) == 0 or len(db) == 0:
            return np.zeros((0, 2), dtype=int)

        norm = cv2.NORM_HAMMING if da.dtype == np.uint8 else cv2.NORM_L2
        bf = cv2.BFMatcher(norm)
        raw_matches = bf.knnMatch(da, db, k=2)

        good = []
        for pair in raw_matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.ratio_thresh * n.distance:
                good.append([m.queryIdx, m.trainIdx])
        return np.array(good, dtype=int) if good else np.zeros((0, 2), dtype=int)

    def _match_superglue(self, feats_a: dict, feats_b: dict) -> np.ndarray:
        import torch

        with torch.no_grad():
            data = {
                "keypoints0": torch.from_numpy(feats_a["keypoints"]).unsqueeze(0),
                "keypoints1": torch.from_numpy(feats_b["keypoints"]).unsqueeze(0),
                "descriptors0": torch.from_numpy(feats_a["descriptors"]).unsqueeze(0),
                "descriptors1": torch.from_numpy(feats_b["descriptors"]).unsqueeze(0),
            }
            out = self._superglue(data)
        matches0 = out["matches0"][0].numpy()
        idx_a = np.where(matches0 > -1)[0]
        idx_b = matches0[idx_a]
        return np.stack([idx_a, idx_b], axis=1)
