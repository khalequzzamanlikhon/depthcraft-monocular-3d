"""In-memory job/session store for the API.

Swap for Redis/a database in a real multi-worker deployment -- kept simple
here since FastAPI + uvicorn's default single-process dev server is the
target for the Phase 5 MVP.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

OUTPUT_ROOT = Path("outputs")
OUTPUT_ROOT.mkdir(exist_ok=True)

_JOBS: dict[str, dict[str, Any]] = {}


def new_job() -> str:
    job_id = uuid.uuid4().hex[:12]
    job_dir = OUTPUT_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    _JOBS[job_id] = {"dir": job_dir}
    return job_id


class JobNotFoundError(Exception):
    """Raised when a job_id is not present in the store."""


def get_job(job_id: str) -> dict[str, Any]:
    if job_id not in _JOBS:
        raise JobNotFoundError(f"Unknown job_id: {job_id}")
    return _JOBS[job_id]


def set_job_data(job_id: str, key: str, value: Any) -> None:
    get_job(job_id)[key] = value


def job_dir(job_id: str) -> Path:
    return get_job(job_id)["dir"]
