"""Measurement endpoint: two pixel clicks -> 3D distance + uncertainty."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from depthcraft.api.job_store import JobNotFoundError, get_job
from depthcraft.api.models import MeasureRequest, MeasureResponse
from depthcraft.measurement.measure_engine import click_to_measure
from depthcraft.utils.geometry_utils import default_intrinsics

router = APIRouter(prefix="/api/v1", tags=["measure"])


@router.post("/measure", response_model=MeasureResponse)
async def measure_distance(req: MeasureRequest):
    try:
        job = get_job(req.job_id)
    except JobNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    pcd = job.get("pointcloud")
    if pcd is None:
        raise HTTPException(404, "No reconstruction found for this job_id. Call /reconstruct first.")

    import numpy as np

    points = np.asarray(pcd.points)
    intrinsics = job.get("intrinsics")
    if intrinsics is None:
        intrinsics = default_intrinsics(1920, 1080)
    pose_w2c = np.eye(4)  # single-image MVP: camera at world origin

    try:
        result = click_to_measure(
            (req.point_a.x, req.point_a.y),
            (req.point_b.x, req.point_b.y),
            intrinsics,
            pose_w2c,
            points=points,
        )
    except RuntimeError as e:
        raise HTTPException(422, str(e)) from e

    return MeasureResponse(
        distance_m=result.distance_m,
        uncertainty_1sigma_m=result.uncertainty_1sigma_m,
        uncertainty_95ci_m=result.uncertainty_95ci_m,
    )
