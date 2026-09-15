"""Feature extraction: ORB fast path is fully real (OpenCV, zero extra deps).

SuperPoint requires extra weights (kornia or the original SuperPoint repo);
wired up with graceful fallback to ORB, exactly like the depth engines.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from depthcraft.utils.logger import get_logger

log = get_logger(__name__)


class FeatureExtractor:
    def __init__(self, backend: str = "superpoint", n_features: int = 4000):
        self.requested_backend = backend
        self.n_features = n_features
        self._orb: Any = None
        self._superpoint: Any = None
        self.backend = self._resolve_backend(backend)

    def _resolve_backend(self, backend: str) -> str:
        if backend == "orb":
            self._orb = _make_orb(self.n_features)
            return "orb"
        try:
            import kornia.feature as KF

            self._superpoint = KF.SuperPoint(pretrained=True).eval()
            return "superpoint"
        except Exception as e:  # noqa: BLE001
            log.warning(f"SuperPoint unavailable ({e!r}); falling back to ORB.")
            self._orb = _make_orb(self.n_features)
            return "orb"

    def extract(self, image: np.ndarray) -> dict:
        """Returns {'keypoints': (N,2) float32, 'descriptors': (N,D), 'scores': (N,)}"""
        if self.backend == "orb":
            return self._extract_orb(image)
        return self._extract_superpoint(image)

    def _extract_orb(self, image: np.ndarray) -> dict:
        import cv2

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        kps, descs = self._orb.detectAndCompute(gray, None)
        if not kps:
            return {"keypoints": np.zeros((0, 2)), "descriptors": np.zeros((0, 32)), "scores": np.zeros((0,))}
        pts = np.array([kp.pt for kp in kps], dtype=np.float32)
        scores = np.array([kp.response for kp in kps], dtype=np.float32)
        return {"keypoints": pts, "descriptors": descs, "scores": scores}

    def _extract_superpoint(self, image: np.ndarray) -> dict:
        import torch

        gray = _to_gray_tensor(image)
        with torch.no_grad():
            out = self._superpoint({"image": gray})
        return {
            "keypoints": out["keypoints"][0].numpy(),
            "descriptors": out["descriptors"][0].numpy(),
            "scores": out["keypoint_scores"][0].numpy(),
        }


def _make_orb(n_features: int):
    import cv2

    # cv2 stubs omit ORB_create despite it existing at runtime.
    return cv2.ORB_create(nfeatures=n_features)  # type: ignore[attr-defined]


def _to_gray_tensor(image: np.ndarray):
    import torch

    gray = image.mean(axis=-1, keepdims=True) / 255.0
    return torch.from_numpy(gray).permute(2, 0, 1).unsqueeze(0).float()
