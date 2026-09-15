#!/usr/bin/env python
"""CLI: train 3D Gaussian Splatting from a COLMAP dataset.

Usage:
    python scripts/train_3dgs.py --colmap_path data/room --output outputs/room --backend nerfstudio
"""
import argparse

from depthcraft.reconstruction.gaussian_splatting import train_3dgs_gsplat, train_3dgs_nerfstudio

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--colmap_path", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--backend", choices=["nerfstudio", "gsplat"], default="nerfstudio")
    p.add_argument("--iterations", type=int, default=7000)
    args = p.parse_args()

    if args.backend == "nerfstudio":
        train_3dgs_nerfstudio(args.colmap_path, args.output, max_iterations=args.iterations)
    else:
        train_3dgs_gsplat(args.colmap_path, args.output, iterations=args.iterations)
