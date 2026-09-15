"""IO helpers: image loading, depth map save/load, npy/exr fallback."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path: str | Path) -> np.ndarray:
    """Load an image as RGB uint8 array (H, W, 3)."""
    return np.array(Image.open(path).convert("RGB"))


def save_depth_npy(path: str | Path, depth: np.ndarray) -> None:
    np.save(str(Path(path).with_suffix(".npy")), depth.astype(np.float32))


def load_depth_npy(path: str | Path) -> np.ndarray:
    return np.load(str(Path(path).with_suffix(".npy")))


def save_depth_png16(path: str | Path, depth_m: np.ndarray, scale: float = 1000.0) -> None:
    """Save metric depth (meters) as a 16-bit PNG in millimetres, like Kinect/LiDAR dumps."""
    depth_mm = np.clip(depth_m * scale, 0, 65535).astype(np.uint16)
    Image.fromarray(depth_mm).save(path)


def load_depth_png16(path: str | Path, scale: float = 1000.0) -> np.ndarray:
    depth_mm = np.array(Image.open(path)).astype(np.float32)
    return depth_mm / scale


def save_json(path: str | Path, obj: dict) -> None:
    Path(path).write_text(json.dumps(obj, indent=2))


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
