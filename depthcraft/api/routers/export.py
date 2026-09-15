"""Export endpoint: hand back already-produced job artifacts in the requested format."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from depthcraft.api.job_store import job_dir
from depthcraft.api.models import ExportRequest, ExportResponse

router = APIRouter(prefix="/api/v1", tags=["export"])

_EXT_MAP = {"ply": "pointcloud.ply", "obj": "mesh.obj", "glb": "mesh.glb", "splat": "gaussians.splat"}


@router.post("/export/{format}", response_model=ExportResponse)
async def export_asset(format: str, req: ExportRequest):
    if format not in _EXT_MAP:
        raise HTTPException(400, f"Unsupported format: {format}")

    out_dir = job_dir(req.job_id)
    filename = _EXT_MAP[format]
    path = out_dir / filename
    if not path.exists():
        raise HTTPException(
            404,
            f"{filename} not yet generated for job {req.job_id}. "
            "Run the relevant reconstruction/meshing/splatting step first.",
        )
    return ExportResponse(file_url=f"/outputs/{req.job_id}/{filename}")
