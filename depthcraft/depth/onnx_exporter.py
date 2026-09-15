"""Export a loaded torch depth model to ONNX and (optionally) build a TensorRT engine.

Requires the `depth` and `onnx` extras: `pip install -e ".[depth,onnx]"`.
Not imported by default anywhere else in the pipeline, so the base install
stays lightweight.
"""
from __future__ import annotations

from pathlib import Path

from depthcraft.utils.logger import get_logger

log = get_logger(__name__)


def export_to_onnx(
    torch_model,
    output_path: str | Path,
    input_shape: tuple[int, int, int, int] = (1, 3, 518, 518),
    opset_version: int = 17,
) -> Path:
    """Export a torch depth model to ONNX with dynamic H/W axes."""
    import torch

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(*input_shape)
    torch_model.eval()

    torch.onnx.export(
        torch_model,
        (dummy_input,),
        str(output_path),
        opset_version=opset_version,
        input_names=["image"],
        output_names=["depth"],
        dynamic_axes={
            "image": {0: "batch", 2: "height", 3: "width"},
            "depth": {0: "batch", 1: "height", 2: "width"},
        },
    )
    log.info(f"Exported ONNX model to {output_path}")
    return output_path


def build_tensorrt_engine(onnx_path: str | Path, engine_path: str | Path, fp16: bool = True):
    """Build a TensorRT engine from an ONNX file. Requires `tensorrt` + a CUDA GPU.

    This is a thin wrapper -- TensorRT's Python API is intentionally verbose
    and version-sensitive, so we keep the logic minimal and let the caller
    handle environment-specific builder config.
    """
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                log.error(parser.get_error(i))
            raise RuntimeError("Failed to parse ONNX model for TensorRT.")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    serialized_engine = builder.build_serialized_network(network, config)
    Path(engine_path).write_bytes(serialized_engine)
    log.info(f"Built TensorRT engine at {engine_path}")
    return engine_path


def benchmark_backend(infer_fn, sample_input, n_runs: int = 30) -> dict:
    """Simple wall-clock latency benchmark shared by PyTorch/ONNX/TensorRT paths."""
    import time

    # warmup
    for _ in range(3):
        infer_fn(sample_input)

    times: list[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        infer_fn(sample_input)
        times.append(time.perf_counter() - t0)

    import numpy as np

    times_arr = np.array(times)
    return {
        "mean_ms": float(times_arr.mean() * 1000),
        "p50_ms": float(np.percentile(times_arr, 50) * 1000),
        "p95_ms": float(np.percentile(times_arr, 95) * 1000),
        "fps": float(1.0 / times_arr.mean()),
    }
