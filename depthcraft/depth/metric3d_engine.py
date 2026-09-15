"""Metric3D alternative backend -- better for indoor architectural scenes per spec.

Same fallback philosophy as depth_anything_engine.py: tries the real model,
degrades gracefully to a classical estimator otherwise, so callers never
have to special-case "no GPU available".
"""
from __future__ import annotations

from typing import Any

import numpy as np

from depthcraft.depth.depth_anything_engine import DepthAnythingEngine
from depthcraft.utils.logger import get_logger

log = get_logger(__name__)


class Metric3DEngine:
    def __init__(self, device: str | None = None):
        self.device = device
        self._backend = "fallback"
        self._model: Any = None
        self._device: str = device or "cpu"
        self._fallback_engine: DepthAnythingEngine | None = None
        self._try_load()

    def _try_load(self) -> None:
        try:
            import torch

            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            # Metric3D is distributed via torch.hub; requires network + weights.
            self._model = torch.hub.load(
                "YvanYin/Metric3D", "metric3d_vit_small", pretrain=True
            ).to(device).eval()
            self._device = device
            self._backend = "torch_hub"
            log.info(f"Loaded Metric3D on {device}")
        except Exception as e:  # noqa: BLE001
            log.warning(
                f"Metric3D unavailable ({e!r}); delegating to DepthAnythingEngine fallback."
            )
            self._fallback_engine = DepthAnythingEngine()
            self._backend = "fallback"

    @property
    def backend(self) -> str:
        return self._backend

    def infer(self, image: np.ndarray) -> np.ndarray:
        if self._backend == "torch_hub":
            return self._infer_real(image)
        assert self._fallback_engine is not None, "fallback engine is always set when backend != 'torch_hub'"
        return self._fallback_engine.infer(image)

    def _infer_real(self, image: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            x = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            x = x.to(self._device)
            pred = self._model.inference({"input": x})
            return pred["prediction"].squeeze().cpu().numpy().astype(np.float32)
