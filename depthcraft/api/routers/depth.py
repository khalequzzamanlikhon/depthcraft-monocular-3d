from __future__ import annotations

import numpy as np
from fastapi import APIRouter, UploadFile
from PIL import Image

from depthcraft.api.job_store import job_dir, new_job
from depthcraft.api.models import DepthResponse
from depthcraft.depth.depth_anything_engine import DepthAnythingEngine, colorize_depth
from depthcraft.depth.uncertainty_estimator import estimate_uncertainty_tta
from depthcraft.utils.io_utils import save_depth_npy

router = APIRouter(prefix="/api/v1", tags=["depth"])

_engine: DepthAnythingEngine | None = None


def get_engine() -> DepthAnythingEngine:
    global _engine
    if _engine is None:
        _engine = DepthAnythingEngine()
    return _engine


@router.post("/depth", response_model=DepthResponse)
async def estimate_depth(file: UploadFile, with_uncertainty: bool = False):
    job_id = new_job()
    out_dir = job_dir(job_id)

    image = np.array(Image.open(file.file).convert("RGB"))
    engine = get_engine()

    if with_uncertainty:
        depth, uncertainty = estimate_uncertainty_tta(engine, image)
    else:
        depth = engine.infer(image)
        uncertainty = None

    save_depth_npy(out_dir / "depth", depth)
    depth_vis_path = out_dir / "depth_vis.png"
    Image.fromarray(colorize_depth(depth)).save(depth_vis_path)

    uncertainty_url = None
    if uncertainty is not None:
        save_depth_npy(out_dir / "uncertainty", uncertainty)
        uncertainty_vis_path = out_dir / "uncertainty_vis.png"
        Image.fromarray(colorize_depth(uncertainty, colormap="viridis")).save(uncertainty_vis_path)
        uncertainty_url = f"/outputs/{job_id}/uncertainty_vis.png"

    h, w = depth.shape
    return DepthResponse(
        job_id=job_id,
        depth_map_url=f"/outputs/{job_id}/depth_vis.png",
        uncertainty_map_url=uncertainty_url,
        backend=engine.backend,
        width=w,
        height=h,
    )
