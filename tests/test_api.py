import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from depthcraft.api.main import app

client = TestClient(app)


def _sample_image_bytes():
    img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    buf.seek(0)
    return buf


def test_health_endpoint():
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"


def test_depth_endpoint_returns_valid_response():
    files = {"file": ("test.png", _sample_image_bytes(), "image/png")}
    resp = client.post("/api/v1/depth", files=files)
    assert resp.status_code == 200
    body = resp.json()
    assert "job_id" in body
    assert body["width"] == 64
    assert body["height"] == 64


def test_reconstruct_and_measure_flow():
    files = {"file": ("test.png", _sample_image_bytes(), "image/png")}
    recon_resp = client.post("/api/v1/reconstruct", files=files)
    assert recon_resp.status_code == 200
    job_id = recon_resp.json()["job_id"]
    assert recon_resp.json()["num_points"] > 0

    measure_resp = client.post(
        "/api/v1/measure",
        json={
            "job_id": job_id,
            "point_a": {"x": 5, "y": 5},
            "point_b": {"x": 50, "y": 50},
        },
    )
    # Either a valid measurement or a 422 (ray missed sparse cloud) is acceptable
    # for a random-noise test image; we only assert the endpoint behaves, not
    # that random noise produces a specific geometric answer.
    assert measure_resp.status_code in (200, 422)


def test_measure_unknown_job_returns_404():
    resp = client.post(
        "/api/v1/measure",
        json={"job_id": "doesnotexist", "point_a": {"x": 0, "y": 0}, "point_b": {"x": 1, "y": 1}},
    )
    assert resp.status_code == 404
