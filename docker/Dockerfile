# Multi-stage build for production (GPU)
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS base

RUN apt-get update && apt-get install -y \
    python3.11 python3-pip python3.11-venv \
    libgl1 libglx-mesa0 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    git wget colmap \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY depthcraft/ ./depthcraft/
RUN pip install --no-cache-dir -e ".[dev]"

COPY config/ ./config/
COPY mvp_demo.py .
COPY scripts/ ./scripts/

EXPOSE 8000
CMD ["uvicorn", "depthcraft.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
