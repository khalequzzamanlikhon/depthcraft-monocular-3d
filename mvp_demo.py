#!/usr/bin/env python
"""Phase 0 MVP demo: image.jpg -> depth map -> metric point cloud -> click two
points -> distance in cm.

Usage:
    python mvp_demo.py --image room.jpg
    python mvp_demo.py --image room.jpg --headless   # skip interactive picking,
                                                       # just print pipeline stats
"""
from __future__ import annotations

import argparse

import numpy as np
from PIL import Image

from depthcraft.depth.depth_anything_engine import DepthAnythingEngine, colorize_depth
from depthcraft.measurement.measure_engine import measure_with_uncertainty
from depthcraft.reconstruction.pointcloud_processor import full_cleaning_pipeline, o3d_to_numpy
from depthcraft.utils.geometry_utils import default_intrinsics, depth_to_pointcloud
from depthcraft.utils.io_utils import ensure_dir
from depthcraft.utils.logger import get_logger

log = get_logger("mvp_demo")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to a single RGB image")
    parser.add_argument("--output-dir", default="outputs/mvp_demo")
    parser.add_argument("--fov-deg", type=float, default=60.0)
    parser.add_argument("--headless", action="store_true", help="Skip interactive Open3D picking")
    args = parser.parse_args()

    out_dir = ensure_dir(args.output_dir)

    log.info(f"Loading image: {args.image}")
    image = np.array(Image.open(args.image).convert("RGB"))

    log.info("Running depth inference...")
    engine = DepthAnythingEngine()
    depth = engine.infer(image)
    log.info(f"Depth engine backend: {engine.backend}")

    Image.fromarray(colorize_depth(depth)).save(out_dir / "depth_vis.png")
    log.info(f"Saved depth visualization to {out_dir / 'depth_vis.png'}")

    intrinsics = default_intrinsics(image.shape[1], image.shape[0], fov_deg=args.fov_deg)
    points, colors = depth_to_pointcloud(depth, intrinsics, color=image)
    log.info(f"Back-projected {len(points)} points.")

    pcd = full_cleaning_pipeline(points, colors)
    clean_points, _ = o3d_to_numpy(pcd)
    log.info(f"After cleaning: {len(clean_points)} points.")

    import open3d as o3d

    o3d.io.write_point_cloud(str(out_dir / "pointcloud.ply"), pcd)
    log.info(f"Saved point cloud to {out_dir / 'pointcloud.ply'}")

    if args.headless or len(clean_points) < 2:
        log.info(
            "Headless mode: measuring the two most distant points in the "
            "cloud as a smoke test instead of interactive picking."
        )
        from scipy.spatial.distance import pdist, squareform

        sample = clean_points[np.random.choice(len(clean_points), min(500, len(clean_points)), replace=False)]
        d = squareform(pdist(sample))
        i, j = np.unravel_index(np.argmax(d), d.shape)
        p1, p2 = sample[i], sample[j]
        sigma = np.abs(p1) * 0.02 + 0.005
        result = measure_with_uncertainty(p1, p2, sigma, sigma)
        log.info(str(result))
        return

    from depthcraft.visualization.open3d_viewer import Open3DViewer

    viewer = Open3DViewer()
    viewer.load_geometry(pcd)
    log.info("Shift+click two points in the viewer window, then close it.")
    picks = viewer.run_and_get_picks()

    if len(picks) < 2:
        log.warning("Fewer than 2 points picked -- nothing to measure.")
        return

    p1, p2 = picks[0]["coord"], picks[1]["coord"]
    sigma = np.abs(p1) * 0.02 + 0.005  # conservative default uncertainty
    result = measure_with_uncertainty(p1, p2, sigma, sigma)
    print(result)


if __name__ == "__main__":
    main()
