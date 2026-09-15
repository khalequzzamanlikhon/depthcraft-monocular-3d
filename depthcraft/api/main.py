"""DepthCraft FastAPI backend entrypoint.

Run with:
    uvicorn depthcraft.api.main:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from depthcraft.api.job_store import OUTPUT_ROOT
from depthcraft.api.models import HealthResponse
from depthcraft.api.routers import depth, export, measure, reconstruct
from depthcraft.utils.logger import get_logger

log = get_logger(__name__)

app = FastAPI(
    title="DepthCraft API",
    description="Monocular depth estimation, 3D reconstruction, and virtual tape measure.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(depth.router)
app.include_router(reconstruct.router)
app.include_router(measure.router)
app.include_router(export.router)

app.mount("/outputs", StaticFiles(directory=str(OUTPUT_ROOT)), name="outputs")


@app.get("/api/v1/health", response_model=HealthResponse)
async def health():
    models_loaded = []
    gpu = False
    try:
        import torch

        gpu = torch.cuda.is_available()
    except ImportError:
        pass

    try:
        from depthcraft.api.routers.depth import _engine

        if _engine is not None:
            models_loaded.append(f"depth:{_engine.backend}")
    except Exception as e:  # noqa: BLE001
        log.debug(f"Depth engine not yet initialized: {e!r}")

    return HealthResponse(status="ok", gpu=gpu, models_loaded=models_loaded)


@app.get("/")
async def root():
    return {"message": "DepthCraft API is running. See /docs for interactive API docs."}
