"""Gradio demo UI. Calls the FastAPI backend rather than running inference
directly, per Phase 5 guidance (keep heavy inference out of the Gradio process).

Run the API first:
    uvicorn depthcraft.api.main:app --port 8000
Then:
    python -m depthcraft.visualization.gradio_app
"""
from __future__ import annotations

import io

import gradio as gr
import requests

API_URL = "http://localhost:8000"


def process_image(image, mode):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    files = {"file": ("image.png", buf, "image/png")}

    if mode == "Depth Only":
        resp = requests.post(f"{API_URL}/api/v1/depth", files=files, params={"with_uncertainty": True})
        resp.raise_for_status()
        data = resp.json()
        return f"{API_URL}{data['depth_map_url']}", None, f"Backend: {data['backend']}"

    if mode == "Full Reconstruction":
        resp = requests.post(f"{API_URL}/api/v1/reconstruct", files=files)
        resp.raise_for_status()
        data = resp.json()
        return None, f"{API_URL}{data['pointcloud_url']}", f"{data['num_points']} points reconstructed. job_id={data['job_id']}"

    return None, None, "Unknown mode"


with gr.Blocks(title="DepthCraft") as demo:
    gr.Markdown("# DepthCraft — Depth Estimation & 3D Reconstruction Demo")
    with gr.Row():
        with gr.Column():
            image_in = gr.Image(label="Upload Image", type="pil")
            mode = gr.Radio(["Depth Only", "Full Reconstruction"], value="Depth Only", label="Mode")
            run_btn = gr.Button("Process", variant="primary")
        with gr.Column():
            depth_out = gr.Image(label="Depth Map")
            model_out = gr.Model3D(label="3D Result")
            status_out = gr.Textbox(label="Status")

    run_btn.click(process_image, inputs=[image_in, mode], outputs=[depth_out, model_out, status_out])

if __name__ == "__main__":
    demo.launch()
