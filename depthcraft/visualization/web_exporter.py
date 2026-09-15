"""Export reconstruction outputs to web-friendly formats (.glb for meshes, .splat for gaussians)."""
from __future__ import annotations

from pathlib import Path

from depthcraft.utils.logger import get_logger

log = get_logger(__name__)


def export_glb(mesh, output_path: str | Path) -> Path:
    import open3d as o3d

    output_path = Path(output_path)
    o3d.io.write_triangle_mesh(str(output_path), mesh, write_triangle_uvs=True)
    log.info(f"Exported GLB to {output_path}")
    return output_path


def export_splat(ply_path: str | Path, output_path: str | Path) -> Path:
    """Convert a 3DGS .ply into the compact .splat format used by web viewers
    (antimatter15/splat, gsplat.js). The .splat format packs each Gaussian into
    32 bytes: position(3xf32->downsampled), scale, rgba, rotation(quaternion as u8).
    """
    import numpy as np
    from plyfile import PlyData

    ply = PlyData.read(str(ply_path))
    v = ply["vertex"]
    n = len(v)

    positions = np.stack([v["x"], v["y"], v["z"]], axis=-1)
    scales = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1))
    rotations = np.stack([v[f"rot_{i}"] for i in range(4)], axis=-1)
    rotations /= np.linalg.norm(rotations, axis=-1, keepdims=True) + 1e-8
    colors = np.clip(
        0.5 + np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=-1) * 0.28209479177387814, 0, 1
    )
    opacities = 1 / (1 + np.exp(-v["opacity"]))  # sigmoid

    buffer = bytearray(n * 32)
    for i in range(n):
        off = i * 32
        buffer[off:off + 12] = positions[i].astype("<f4").tobytes()
        buffer[off + 12:off + 24] = scales[i].astype("<f4").tobytes()
        buffer[off + 24:off + 28] = (
            bytes([int(colors[i, 0] * 255), int(colors[i, 1] * 255),
                   int(colors[i, 2] * 255), int(opacities[i] * 255)])
        )
        rot_u8 = ((rotations[i] * 0.5 + 0.5) * 255).astype("uint8")
        buffer[off + 28:off + 32] = bytes(rot_u8.tolist())

    output_path = Path(output_path)
    output_path.write_bytes(bytes(buffer))
    log.info(f"Exported {n} Gaussians as .splat to {output_path}")
    return output_path
