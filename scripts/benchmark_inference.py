#!/usr/bin/env python
"""CLI: benchmark PyTorch vs ONNX vs TensorRT depth inference latency.

Usage:
    python scripts/benchmark_inference.py --image data/sample_images/room.jpg
"""
import argparse

import numpy as np
from PIL import Image

from depthcraft.depth.depth_anything_engine import DepthAnythingEngine
from depthcraft.depth.onnx_exporter import benchmark_backend

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--n-runs", type=int, default=30)
    args = p.parse_args()

    image = np.array(Image.open(args.image).convert("RGB"))
    engine = DepthAnythingEngine()
    print(f"Backend under test: {engine.backend}")

    results = benchmark_backend(engine.infer, image, n_runs=args.n_runs)
    print(f"PyTorch/fallback backend: {results}")

    print(
        "\nTo compare against ONNX/TensorRT, export first with "
        "depthcraft/depth/onnx_exporter.py, then wrap an ONNXRuntime "
        "InferenceSession's .run() call in the same benchmark_backend() helper."
    )
