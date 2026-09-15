"""Metric monocular depth inference using Depth Anything V2.

Design notes
------------
Real inference requires `torch` + downloaded checkpoints (GBs, GPU strongly
recommended) — those are declared as the optional `depth` extra in
pyproject.toml (`pip install -e ".[depth]"`) and are NOT auto-installed,
since this repo must also import cleanly in CPU-only / CI environments.

This engine therefore:
  1. Tries to load the real model via `transformers` (Depth-Anything-V2
     checkpoints are mirrored on the HF Hub, e.g. "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf").
  2. Falls back to a lightweight classical estimator (edge/gradient based
     pseudo-depth) if torch/transformers/weights aren't available, so the
     rest of the pipeline (measurement, fusion, viewer) is still runnable
     and testable end-to-end without a GPU.

Swap `backend="transformers"` for `backend="onnx"` once you've exported
via `onnx_exporter.py` for the fast inference path described in Phase 1.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from depthcraft.utils.logger import get_logger

log = get_logger(__name__)

DEFAULT_CHECKPOINT = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"

# Hard wall-clock budget for loading the real model. huggingface_hub/requests
# calls involved in resolving + downloading a checkpoint don't reliably honor
# a read timeout, so a stalled connection (reachable host, no data) can hang
# far longer than any socket-level timeout would suggest. Loading is run in a
# background thread and abandoned (not killed -- Python threads can't be
# force-killed) if it blows this budget, so startup always falls back promptly
# instead of hanging indefinitely.
MODEL_LOAD_TIMEOUT_S = 20.0


class DepthAnythingEngine:
    """Wraps Depth Anything V2 (metric variant) for single-image inference."""

    def __init__(self, checkpoint: str = DEFAULT_CHECKPOINT, device: str | None = None):
        self.checkpoint = checkpoint
        self.device = device
        self._pipe: Any = None
        self._backend = "fallback"
        self._try_load_real_model()

    def _try_load_real_model(self) -> None:
        import queue
        import threading

        # A plain daemon thread (not concurrent.futures.ThreadPoolExecutor) is
        # deliberate: ThreadPoolExecutor registers an atexit hook that joins its
        # worker threads on interpreter shutdown, which would hang the whole
        # process anyway if the download thread never returns. A daemon thread
        # is simply abandoned by the interpreter at exit.
        result: queue.Queue = queue.Queue(maxsize=1)
        thread = threading.Thread(
            target=lambda: result.put(self._safe_load_pipeline()),
            daemon=True,
            name="depth-model-load",
        )
        thread.start()
        thread.join(timeout=MODEL_LOAD_TIMEOUT_S)

        if thread.is_alive():
            log.warning(
                f"Loading '{self.checkpoint}' exceeded {MODEL_LOAD_TIMEOUT_S:.0f}s "
                "(likely no/stalled network to Hugging Face Hub and no local cache). "
                "Falling back to classical pseudo-depth estimator. The download may "
                "still be proceeding in the background; retry once it's cached."
            )
            self._backend = "fallback"
            return

        outcome = result.get()
        if isinstance(outcome, Exception):
            log.warning(
                f"Real Depth Anything V2 model unavailable ({outcome!r}). "
                "Falling back to classical pseudo-depth estimator. "
                "Install the 'depth' extra and ensure network access to Hugging Face "
                "to use the real model."
            )
            self._backend = "fallback"
        else:
            self._pipe, device = outcome
            self._backend = "transformers"
            log.info(f"Loaded {self.checkpoint} on {device}")

    def _safe_load_pipeline(self):
        try:
            import torch
            from transformers import pipeline

            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            pipe = pipeline(task="depth-estimation", model=self.checkpoint, device=device)
            return pipe, device
        except Exception as e:  # noqa: BLE001 - reported back to the main thread
            return e

    @property
    def backend(self) -> str:
        return self._backend

    def infer(self, image: np.ndarray) -> np.ndarray:
        """Run depth inference. Returns an (H, W) float32 metric-ish depth map (meters)."""
        if self._backend == "transformers":
            return self._infer_real(image)
        return self._infer_fallback(image)

    # -- real backend --------------------------------------------------
    def _infer_real(self, image: np.ndarray) -> np.ndarray:
        from PIL import Image as PILImage

        pil_img = PILImage.fromarray(image)
        out = self._pipe(pil_img)
        depth = np.array(out["predicted_depth"] if "predicted_depth" in out else out["depth"])
        return depth.astype(np.float32)

    # -- fallback backend -------------------------------------------------
    def _infer_fallback(self, image: np.ndarray) -> np.ndarray:
        """Classical pseudo-depth: assumes bottom-of-frame = near, top = far,
        modulated by local contrast (a crude proxy for texture/occlusion edges).
        This is NOT metrically accurate -- it exists purely so the rest of the
        pipeline (fusion, measurement, viewer, API) has real numeric data to
        run against in environments without GPU/model weights.
        """
        import cv2

        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

        # vertical gradient prior: objects lower in frame tend to be closer
        vertical_prior = np.linspace(1.0, 6.0, h).reshape(h, 1) * np.ones((1, w))

        # local variance as a soft "detail = closer" cue
        blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=3)
        detail = np.abs(gray - blur)
        detail_norm = detail / (detail.max() + 1e-6)

        depth = vertical_prior * (1.2 - 0.4 * detail_norm)
        depth = cv2.GaussianBlur(depth, (0, 0), sigmaX=5)
        return depth.astype(np.float32)


def colorize_depth(depth: np.ndarray, colormap: str = "turbo") -> np.ndarray:
    """Map a depth array to an RGB visualization image using matplotlib colormaps."""
    import matplotlib as mpl

    d = depth.copy()
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    cmap = mpl.colormaps[colormap]
    rgb = (cmap(d)[..., :3] * 255).astype(np.uint8)
    return rgb
