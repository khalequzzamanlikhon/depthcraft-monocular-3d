"""Pydantic request/response schemas for the FastAPI backend."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Point2D(BaseModel):
    x: float
    y: float


class DepthResponse(BaseModel):
    job_id: str
    depth_map_url: str
    uncertainty_map_url: str | None = None
    backend: str
    width: int
    height: int


OutputFormat = Literal["ply", "obj", "glb", "splat"]


def _default_output_formats() -> list[OutputFormat]:
    return ["glb"]


class ReconstructRequest(BaseModel):
    job_id: str
    output_formats: list[OutputFormat] = Field(default_factory=_default_output_formats)


class ReconstructResponse(BaseModel):
    job_id: str
    pointcloud_url: str | None = None
    mesh_url: str | None = None
    splat_url: str | None = None
    num_points: int


class MeasureRequest(BaseModel):
    job_id: str
    point_a: Point2D
    point_b: Point2D


class MeasureResponse(BaseModel):
    distance_m: float
    uncertainty_1sigma_m: float
    uncertainty_95ci_m: float
    unit: str = "meters"


class ExportRequest(BaseModel):
    job_id: str
    format: Literal["ply", "obj", "glb", "splat"]


class ExportResponse(BaseModel):
    file_url: str


class HealthResponse(BaseModel):
    status: str
    gpu: bool
    models_loaded: list[str]
