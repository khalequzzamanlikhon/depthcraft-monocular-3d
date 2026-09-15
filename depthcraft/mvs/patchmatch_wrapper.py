"""PatchMatch MVS via pycolmap's dense stereo pipeline (Phase 3 step 1)."""
from __future__ import annotations

from pathlib import Path

from depthcraft.utils.logger import get_logger

log = get_logger(__name__)


def run_patchmatch_mvs(sparse_model_dir: str | Path, image_dir: str | Path, output_dir: str | Path):
    """Runs COLMAP's dense stereo pipeline: undistortion -> PatchMatch stereo -> fusion.

    Requires the `sfm` extra + COLMAP system install (this reuses the same
    binary as colmap_wrapper.py, just its dense-reconstruction commands).
    """
    try:
        import pycolmap
    except ImportError as e:
        raise ImportError("pycolmap required. Install with `pip install -e '.[sfm]'`.") from e

    sparse_model_dir, image_dir, output_dir = Path(sparse_model_dir), Path(image_dir), Path(output_dir)
    undistorted_dir = output_dir / "dense"
    undistorted_dir.mkdir(parents=True, exist_ok=True)

    log.info("Undistorting images for dense stereo")
    pycolmap.undistort_images(str(undistorted_dir), str(sparse_model_dir), str(image_dir))

    log.info("Running PatchMatch stereo (photometric + geometric consistency)")
    pycolmap.patch_match_stereo(str(undistorted_dir))

    log.info("Fusing depth maps into dense point cloud")
    dense_ply = output_dir / "dense_pointcloud.ply"
    pycolmap.stereo_fusion(str(dense_ply), str(undistorted_dir))

    return {"dense_pointcloud": str(dense_ply), "dense_dir": str(undistorted_dir)}


def load_confidence_maps(dense_dir: str | Path) -> dict:
    """PatchMatch writes per-pixel photometric-consistency confidence alongside
    each depth map under <dense_dir>/stereo/depth_maps/. This loads them as
    numpy arrays keyed by image filename.
    """

    dense_dir = Path(dense_dir)
    depth_dir = dense_dir / "stereo" / "depth_maps"
    confidences = {}
    for f in sorted(depth_dir.glob("*.geometric.bin")):
        confidences[f.stem] = _read_colmap_array(f)
    return confidences


def _read_colmap_array(path: Path):
    """Parses COLMAP's custom binary array format (width&height&channels&\\n + floats)."""
    import numpy as np

    with open(path, "rb") as f:
        header = b""
        while True:
            byte = f.read(1)
            header += byte
            if header.count(b"&") == 3:
                break
        w, h, c = map(int, header.decode().strip("&\n").split("&"))
        data = np.fromfile(f, dtype=np.float32).reshape(h, w, c)
    return data
