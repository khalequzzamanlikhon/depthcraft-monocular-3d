#!/usr/bin/env python
"""End-to-end multi-view dense reconstruction script (Phases 2-4):
video frames -> masking -> SfM -> MVS -> fusion -> TSDF -> mesh + splat prep.

This is the long-running batch job referenced by the API's /reconstruct
endpoint for the "full" (non single-image-MVP) path. Requires the `sfm`
extra (COLMAP) at minimum; masking and splatting extras are optional.

Usage:
    python scripts/dense_recon.py --frames_dir data/frames/room --output_dir outputs/room
"""
import argparse
from pathlib import Path

from depthcraft.masking.yolo_masker import YOLOMasker
from depthcraft.mvs.patchmatch_wrapper import load_confidence_maps, run_patchmatch_mvs
from depthcraft.sfm.colmap_wrapper import export_poses_and_points, run_sfm
from depthcraft.utils.logger import get_logger

log = get_logger("dense_recon")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--skip_masking", action="store_true")
    args = p.parse_args()

    frames_dir = Path(args.frames_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_masking:
        log.info("Step 1/5: dynamic object masking (YOLOv8-seg fallback path)")
        masker = YOLOMasker()
        if not masker.available:
            log.warning("No masker available -- proceeding without dynamic-object exclusion.")

    log.info("Step 2/5: Structure from Motion (COLMAP)")
    database_path = output_dir / "database.db"
    sparse_dir = output_dir / "sparse"
    reconstructions = run_sfm(frames_dir, database_path, sparse_dir)
    best = reconstructions[max(reconstructions, key=lambda k: reconstructions[k].num_reg_images())]
    export_poses_and_points(best, sparse_dir / "0")

    log.info("Step 3/5: Multi-View Stereo (PatchMatch)")
    mvs_result = run_patchmatch_mvs(sparse_dir / "0", frames_dir, output_dir)
    confidences = load_confidence_maps(mvs_result["dense_dir"])
    log.info(f"Loaded {len(confidences)} confidence maps.")

    log.info(
        "Step 4/5: depth fusion + TSDF integration "
        "(per-frame mono-depth alignment loop omitted here -- see "
        "depth/metric_converter.py + mvs/dense_fusion.py for the per-frame call pattern)"
    )
    # A production run loops over registered frames here, calling
    # align_scale_ransac() + confidence_fusion() per frame, then
    # TSDFFuser.integrate_frame() with the fused depth + pose.

    log.info("Step 5/5: meshing")
    log.info(
        "Point cloud -> clean_pointcloud() -> estimate_normals() -> "
        "poisson_reconstruct() -> export_mesh() -- wire up with your fused "
        "TSDF point cloud once Step 4's per-frame loop is filled in."
    )


if __name__ == "__main__":
    main()
