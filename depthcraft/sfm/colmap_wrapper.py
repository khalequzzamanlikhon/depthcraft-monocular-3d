"""COLMAP-based incremental Structure-from-Motion via pycolmap.

Requires the `sfm` extra (`pip install -e ".[sfm]"`) which needs the COLMAP
binary/library available on the system (apt install colmap, or build from
source with CUDA for GPU-accelerated feature extraction/matching).
"""
from __future__ import annotations

from pathlib import Path

from depthcraft.utils.logger import get_logger

log = get_logger(__name__)


def run_sfm(image_dir: str | Path, database_path: str | Path, output_dir: str | Path):
    """Run full incremental SfM: feature extraction -> exhaustive matching -> mapping.

    Returns a dict of {reconstruction_id: pycolmap.Reconstruction}.
    """
    try:
        import pycolmap
    except ImportError as e:
        raise ImportError(
            "pycolmap is required for SfM. Install with `pip install -e '.[sfm]'` "
            "and ensure the COLMAP system library is installed."
        ) from e

    image_dir, database_path, output_dir = Path(image_dir), Path(database_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    database_path.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"Extracting features from {image_dir}")
    pycolmap.extract_features(str(database_path), str(image_dir))

    log.info("Running exhaustive matching")
    pycolmap.match_exhaustive(str(database_path))

    log.info("Running incremental mapping")
    reconstructions = pycolmap.incremental_mapping(
        str(database_path), str(image_dir), str(output_dir)
    )

    if not reconstructions:
        raise RuntimeError("COLMAP produced no reconstruction -- check image overlap/texture.")

    best_id = max(reconstructions, key=lambda k: reconstructions[k].num_reg_images())
    best = reconstructions[best_id]
    log.info(
        f"Best reconstruction: {best.num_reg_images()} registered images, "
        f"{best.num_points3D()} points"
    )
    return reconstructions


def export_poses_and_points(reconstruction, output_dir: str | Path):
    """Write cameras.txt/images.txt/points3D.txt in COLMAP's text format."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reconstruction.write_text(str(output_dir))
    log.info(f"Exported COLMAP text model to {output_dir}")
