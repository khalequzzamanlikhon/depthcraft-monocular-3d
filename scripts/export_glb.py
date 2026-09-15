#!/usr/bin/env python
"""CLI: mesh (.ply/.obj) -> .glb for web viewing."""
import argparse

from depthcraft.visualization.web_exporter import export_glb

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(args.input)
    export_glb(mesh, args.output)
