"""Reconstruction endpoint: single-image path builds a metric point cloud
from mono depth (fast MVP path). Full video->SfM->MVS->TSDF pipeline is
exposed via the same job store but orchestrated by scripts/dense_recon.py
for long-running jobs (kept out of the request/response cycle).
"""
from __future__ import annotations

import numpy as np
from fastapi import APIRouter, UploadFile
from PIL import Image

from depthcraft.api.job_store import job_dir, new_job, set_job_data
from depthcraft.api.models import ReconstructResponse
from depthcraft.api.routers.depth import get_engine
from depthcraft.reconstruction.pointcloud_processor import full_cleaning_pipeline, o3d_to_numpy
from depthcraft.utils.geometry_utils import default_intrinsics, depth_to_pointcloud

router = APIRouter(prefix="/api/v1", tags=["reconstruct"])


@router.post("/reconstruct", response_model=ReconstructResponse)
async def reconstruct_single_image(file: UploadFile, voxel_size: float = 0.01):
    """MVP single-image reconstruction: depth -> point cloud -> clean -> export.
    For multi-view dense reconstruction (SfM+MVS+TSDF+splats), see scripts/dense_recon.py.
    """
    import open3d as o3d

    job_id = new_job()
    out_dir = job_dir(job_id)

    image = np.array(Image.open(file.file).convert("RGB"))
    engine = get_engine()
    depth = engine.infer(image)

    intrinsics = default_intrinsics(image.shape[1], image.shape[0])
    points, colors = depth_to_pointcloud(depth, intrinsics, color=image)

    pcd = full_cleaning_pipeline(points, colors, voxel_size=voxel_size)
    set_job_data(job_id, "pointcloud", pcd)
    set_job_data(job_id, "intrinsics", intrinsics)

    ply_path = out_dir / "pointcloud.ply"
    o3d.io.write_point_cloud(str(ply_path), pcd)

    clean_pts, _ = o3d_to_numpy(pcd)
    return ReconstructResponse(
        job_id=job_id,
        pointcloud_url=f"/outputs/{job_id}/pointcloud.ply",
        num_points=len(clean_pts),
    )
