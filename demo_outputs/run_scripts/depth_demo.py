"""Batch DepthCraft demo for README assets.

Per image: real Depth Anything V2 depth, TTA uncertainty, an RGB|depth|uncertainty panel,
the cleaned metric point cloud (.ply) with a rendered preview, and the same headless
measurement mvp_demo.py prints. Ends with a GPU latency benchmark and summary.json.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

import depthcraft.depth.depth_anything_engine as dae

dae.MODEL_LOAD_TIMEOUT_S = float(os.environ.get("DEPTH_LOAD_TIMEOUT_S", "600"))

from depthcraft.depth.uncertainty_estimator import estimate_uncertainty_tta  # noqa: E402
from depthcraft.measurement.measure_engine import measure_with_uncertainty  # noqa: E402
from depthcraft.reconstruction.pointcloud_processor import full_cleaning_pipeline, o3d_to_numpy  # noqa: E402
from depthcraft.utils.geometry_utils import default_intrinsics, depth_to_pointcloud  # noqa: E402


def fit_depth(depth, h: int, w: int) -> np.ndarray:
    """The HF pipeline may return depth at model resolution; back-projection needs image size."""
    import cv2

    depth = np.squeeze(np.asarray(depth, dtype=np.float32))
    if depth.shape != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
    return depth


def panel(images: list[Image.Image], height: int = 360) -> Image.Image:
    resized = [im.resize((int(im.width * height / im.height), height)) for im in images]
    out = Image.new("RGB", (sum(r.width for r in resized), height), "white")
    x = 0
    for r in resized:
        out.paste(r, (x, 0))
        x += r.width
    return out


def render_cloud(points: np.ndarray, colors: np.ndarray, path: Path, max_points: int = 40000) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = np.random.default_rng(0).choice(len(points), min(max_points, len(points)), replace=False)
    p, c = points[idx], colors[idx]
    if c.max() > 1.0:
        c = c / 255.0
    fig = plt.figure(figsize=(8, 6), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(p[:, 0], p[:, 2], -p[:, 1], c=np.clip(c, 0, 1), s=0.6, linewidths=0)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("depth (m)")
    ax.set_zlabel("height (m)")
    ax.view_init(elev=18, azim=-62)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def synced_ms(fn, *args):
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t = time.perf_counter()
    result = fn(*args)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return result, (time.perf_counter() - t) * 1000


def process_image(engine, img_path: Path, out: Path) -> dict:
    import open3d as o3d
    from scipy.spatial.distance import pdist, squareform

    d = out / img_path.stem
    d.mkdir(parents=True, exist_ok=True)
    pil = Image.open(img_path).convert("RGB")
    image = np.array(pil)
    h, w = image.shape[:2]

    engine.infer(image)  # warm-up for this resolution
    raw, infer_ms = synced_ms(engine.infer, image)
    raw_shape = list(np.squeeze(np.asarray(raw)).shape)
    depth = fit_depth(raw, h, w)
    np.save(d / "depth.npy", depth)
    depth_img = Image.fromarray(dae.colorize_depth(depth))
    depth_img.save(d / "depth_vis.png")

    _, uncertainty = estimate_uncertainty_tta(engine, image)
    uncertainty = fit_depth(uncertainty, h, w)
    unc_img = Image.fromarray(dae.colorize_depth(uncertainty, colormap="viridis"))
    unc_img.save(d / "uncertainty_vis.png")
    panel([pil, depth_img, unc_img]).save(d / "panel_rgb_depth_uncertainty.jpg", quality=90)

    intrinsics = default_intrinsics(w, h, fov_deg=60.0)
    points, colors = depth_to_pointcloud(depth, intrinsics, color=image)
    pcd = full_cleaning_pipeline(points, colors)
    o3d.io.write_point_cloud(str(d / "pointcloud.ply"), pcd)
    clean_pts, clean_cols = o3d_to_numpy(pcd)
    if clean_cols is None:
        clean_cols = np.full_like(clean_pts, 0.5)
    render_cloud(clean_pts, clean_cols, d / "pointcloud_preview.png")

    # Same smoke-test measurement mvp_demo.py does in --headless mode.
    rng = np.random.default_rng(0)
    sample = clean_pts[rng.choice(len(clean_pts), min(500, len(clean_pts)), replace=False)]
    i, j = np.unravel_index(np.argmax(squareform(pdist(sample))), (len(sample), len(sample)))
    p1, p2 = sample[i], sample[j]
    measurement = measure_with_uncertainty(p1, p2, np.abs(p1) * 0.02 + 0.005, np.abs(p2) * 0.02 + 0.005)

    return {
        "image": img_path.name,
        "size": [w, h],
        "raw_depth_shape": raw_shape,
        "infer_ms": round(infer_ms, 1),
        "depth_m": {k: round(float(f(depth)), 3) for k, f in (("min", np.min), ("median", np.median), ("max", np.max))},
        "uncertainty_mean": round(float(uncertainty.mean()), 4),
        "points_backprojected": int(len(points)),
        "points_after_cleaning": int(len(clean_pts)),
        "measurement": str(measurement),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bench-runs", type=int, default=30)
    ap.add_argument("--allow-fallback", action="store_true")
    args = ap.parse_args()

    import torch

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    engine = dae.DepthAnythingEngine()
    load_s = time.time() - t0
    print(f"backend={engine.backend} checkpoint={engine.checkpoint} load={load_s:.1f}s")
    if engine.backend != "transformers" and not args.allow_fallback:
        raise SystemExit("Real Depth Anything V2 model did not load; not producing fallback demo assets.")

    summary = {
        "backend": engine.backend,
        "checkpoint": engine.checkpoint,
        "model_load_s": round(load_s, 1),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "images": [],
        "errors": [],
    }
    images = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    for img_path in images:
        try:
            record = process_image(engine, img_path, out)
            print(json.dumps(record))
            summary["images"].append(record)
        except Exception:  # keep going; one bad image shouldn't lose the rest
            traceback.print_exc()
            summary["errors"].append({"image": img_path.name, "error": traceback.format_exc(limit=3)})

    if images and args.bench_runs:
        image = np.array(Image.open(images[0]).convert("RGB"))
        engine.infer(image)
        lat = np.array([synced_ms(engine.infer, image)[1] for _ in range(args.bench_runs)])
        summary["benchmark"] = {
            "image": images[0].name,
            "runs": args.bench_runs,
            "mean_ms": round(float(lat.mean()), 1),
            "p50_ms": round(float(np.percentile(lat, 50)), 1),
            "p95_ms": round(float(np.percentile(lat, 95)), 1),
            "fps": round(1000.0 / float(lat.mean()), 2),
        }
        print("benchmark:", summary["benchmark"])

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out / 'summary.json'}")
    if summary["errors"]:
        raise SystemExit(f"{len(summary['errors'])} image(s) failed")


if __name__ == "__main__":
    main()
