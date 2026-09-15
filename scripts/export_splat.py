#!/usr/bin/env python
"""CLI: trained 3DGS .ply -> compact .splat for web viewers (gsplat.js / super-splat)."""
import argparse

from depthcraft.visualization.web_exporter import export_splat

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to 3DGS .ply")
    p.add_argument("--output", required=True, help="Path to output .splat")
    args = p.parse_args()

    export_splat(args.input, args.output)
