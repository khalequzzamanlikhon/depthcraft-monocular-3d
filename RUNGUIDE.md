# DepthCraft — Complete Run Guide

Everything you need to go from zero to a running app, explained step by step.
No assumptions. No skipped steps.

---

## What this project actually does

You upload a photo → it estimates depth → builds a 3D point cloud → lets you
click two points and measure the real-world distance between them (e.g. "that
wall is 3.4 m wide"). There is also a web viewer, a REST API, and an optional
photorealistic 3D rendering path.

---

## Before you start — what you need

| Thing | Required? | Notes |
|---|---|---|
| Python 3.10 or newer | Yes | Check with `python3 --version` |
| pip | Yes | Comes with Python |
| A terminal / command prompt | Yes | |
| Internet connection | Yes | To download packages |
| A GPU (NVIDIA) | **No** | Makes depth quality better, not required to run |
| COLMAP installed | No | Only for multi-image 3D reconstruction |
| Node.js 18+ | No | Only if you want the React web viewer |

---

## Step 1 — Get the code

Unzip the file you downloaded:

```bash
unzip DepthCraft.zip
cd depthcraft
```

You should now be inside a folder that looks like this:

```
depthcraft/
├── depthcraft/        ← the actual Python package
├── frontend/          ← React web viewer
├── tests/             ← automated tests
├── scripts/           ← helper scripts
├── docker/            ← Docker files
├── mvp_demo.py        ← the simplest demo you can run
├── pyproject.toml     ← package config (where [dev], [depth] etc are defined)
└── Makefile           ← shortcuts for common commands
```

---

## Step 2 — Create a virtual environment (strongly recommended)

A virtual environment keeps this project's packages separate from your system
Python. Think of it as a clean isolated box.

```bash
# Create it (only do this once)
python3 -m venv .venv

# Activate it — do this EVERY time you open a new terminal
source .venv/bin/activate          # Mac / Linux
.venv\Scripts\activate             # Windows
```

You'll know it worked when your terminal prompt shows `(.venv)` at the start.

---

## Step 3 — Install the base packages

```bash
pip install -e ".[dev]"
```

**What this does:**
- `pip install` — install packages
- `-e .` — install *this project* (the `.` = current folder) in editable mode
  (editable = you can edit the code and changes take effect immediately, no
  reinstall needed)
- `".[dev]"` — also install the `dev` group defined in `pyproject.toml`,
  which adds `pytest`, `ruff`, and `mypy` (testing + linting tools)

This takes 2–5 minutes. It installs numpy, OpenCV, Open3D, FastAPI, Gradio,
scikit-learn, and everything else the project needs to run without a GPU.

**It does NOT install PyTorch** (which is 2–4 GB). That's intentional.
The project runs fine without it using a built-in fallback depth estimator.

---

## Step 4 — Verify everything installed correctly

```bash
pytest tests/ -v
```

Expected output (last few lines):

```
tests/test_api.py::test_health_endpoint PASSED
tests/test_api.py::test_depth_endpoint_returns_valid_response PASSED
tests/test_api.py::test_reconstruct_and_measure_flow PASSED
tests/test_api.py::test_measure_unknown_job_returns_404 PASSED
...
22 passed in 2.00s
```

If you see `22 passed` — you're good. Everything works.

---

## Step 5 — Run the MVP demo (simplest possible thing)

This is the Phase 0 demo: one image in → depth map + point cloud + distance
measurement out. **No server, no browser, just one command.**

```bash
python mvp_demo.py --image path/to/any/photo.jpg --headless
```

Use any `.jpg` or `.png` you have — a photo of your room, your desk, anything.

Example output:

```
INFO | Loading image: room.jpg
INFO | Running depth inference...
INFO | Depth engine backend: fallback
INFO | Back-projected 4096 points.
INFO | After cleaning: 3840 points.
INFO | Saved point cloud to outputs/mvp_demo/pointcloud.ply
INFO | Headless mode: measuring two distant points as a smoke test
INFO | Distance: 3.46 m +/- 0.06 m (95% CI: 3.46 +/- 0.11 m)
```

The line `backend: fallback` means it's using the built-in classical depth
estimator (no PyTorch needed). When you later install the real model it will
say `backend: transformers` instead — and depth quality will be much better.

**Without `--headless`** (needs a screen/display):

```bash
python mvp_demo.py --image room.jpg
```

This opens an interactive Open3D 3D viewer window. Shift+click two points in
the window, close it, and it prints the distance between those two points.

---

## Step 6 — Run the full-stack app (API + demo UI)

This is the "real" app: a REST API backend and a Gradio web UI you use in
your browser.

### 6a. Start the API backend

Open a terminal, activate the venv, and run:

```bash
uvicorn depthcraft.api.main:app --host 0.0.0.0 --port 8000 --reload
```

You should see:

```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

Leave this terminal running. The API is now live.

**Test it immediately** in your browser: open http://localhost:8000/docs

You'll see an interactive API page (Swagger UI) where you can try every
endpoint by clicking — no code needed.

Or test via curl:

```bash
curl http://localhost:8000/api/v1/health
# → {"status":"ok","gpu":false,"models_loaded":[]}
```

### 6b. Start the Gradio demo UI

Open a **second terminal**, activate the venv, then:

```bash
python -m depthcraft.visualization.gradio_app
```

Opens at http://localhost:7860

- Upload any image
- Choose "Depth Only" → see a colorized depth map
- Choose "Full Reconstruction" → get a 3D point cloud (downloadable `.ply` file)

---

## Step 7 — (Optional) Run the React web viewer

This is the fancy browser-based 3D viewer. Needs Node.js 18+.

```bash
# Check if you have Node
node --version    # needs to show v18 or higher

# Install and run
cd frontend
npm install
npm run dev
```

Opens at http://localhost:3000

Upload an image → see the 3D point cloud rendered live in the browser → use
the Measure panel to enter two pixel coordinates and get the real-world
distance.

> Note: the API (Step 6a) must be running for the React app to work.

---

## All the shortcuts (Makefile)

Instead of typing long commands, you can use `make`:

```bash
make install        # = pip install -e ".[dev]"
make test           # = pytest tests/ -v
make lint           # = ruff check + mypy
make api            # = start the FastAPI backend
make gradio         # = start the Gradio demo
make clean          # = delete outputs/ and cache files
```

```bash
# Run the MVP demo with make
make mvp IMAGE=path/to/photo.jpg
```

---

## Upgrading to the real depth model (optional, needs internet)

The default fallback depth estimator works but gives rougher results. To
use the actual state-of-the-art Depth Anything V2 model:

```bash
pip install -e ".[depth]"
```

This installs PyTorch and Hugging Face Transformers (~2–4 GB download).

The first time you run *anything* after this (demo, API, tests), it will
automatically download the model weights from Hugging Face (~1.3 GB).
Subsequent runs use the cached weights.

**How you know it worked:** the output will say `backend: transformers`
instead of `backend: fallback`.

```bash
python mvp_demo.py --image room.jpg --headless
# INFO | Depth engine backend: transformers   ← real model
```

You do NOT need a GPU for this. It will just be slower on CPU (~30 sec per
image vs ~0.5 sec on a GPU).

---

## Upgrading to COLMAP multi-view reconstruction (optional)

This unlocks the Phase 2–4 pipeline: multiple photos/video → real 3D
reconstruction. Needs COLMAP installed on your system.

```bash
# Install COLMAP system library first
sudo apt install colmap          # Ubuntu/Debian
brew install colmap              # macOS

# Then install the Python bindings
pip install -e ".[sfm]"

# Extract frames from a video (1 frame per second)
ffmpeg -i myvideo.mp4 -vf fps=1 data/frames/room/frame_%04d.jpg

# Run the full reconstruction pipeline
python scripts/dense_recon.py \
    --frames_dir data/frames/room \
    --output_dir outputs/room
```

---

## What each endpoint does (API reference)

| Endpoint | Method | What it does |
|---|---|---|
| `/api/v1/health` | GET | Check if API is running |
| `/api/v1/depth` | POST | Upload image → get depth map |
| `/api/v1/reconstruct` | POST | Upload image → get 3D point cloud |
| `/api/v1/measure` | POST | Two pixel coords → real-world distance |
| `/api/v1/export/{format}` | POST | Download result as `.ply`, `.obj`, `.glb` |

Full interactive docs: http://localhost:8000/docs (when API is running)

**Quick API test with curl:**

```bash
# Depth estimation
curl -X POST http://localhost:8000/api/v1/depth \
  -F "file=@room.jpg" | python3 -m json.tool

# 3D reconstruction
curl -X POST http://localhost:8000/api/v1/reconstruct \
  -F "file=@room.jpg" | python3 -m json.tool
```

---

## Running with Docker (everything in one command)

No Python setup needed — Docker handles everything.

```bash
# CPU-only (works on any machine)
docker build -t depthcraft:cpu -f docker/Dockerfile.cpu .
docker run -p 8000:8000 depthcraft:cpu

# GPU (needs NVIDIA Container Toolkit)
docker compose -f docker/docker-compose.yml up --build
```

API available at http://localhost:8000 either way.

---

## Project output files

Everything the app produces goes into the `outputs/` folder:

```
outputs/
└── abc123def456/          ← each job gets a unique ID
    ├── depth_vis.png      ← colorized depth map (viewable in any image viewer)
    ├── depth.npy          ← raw depth data (numpy array, meters)
    ├── pointcloud.ply     ← 3D point cloud (open with MeshLab, CloudCompare, or Open3D)
    ├── mesh.glb           ← 3D mesh for web viewing
    └── uncertainty_vis.png ← depth uncertainty map (if requested)
```

`.ply` files can be opened with:
- **MeshLab** (free, recommended) — meshlab.net
- **CloudCompare** (free) — cloudcompare.org
- The **Open3D viewer** built into this project

---

## Summary — the three paths

### Path A: Just want to try it right now (2 minutes)
```bash
pip install -e ".[dev]"
python mvp_demo.py --image photo.jpg --headless
```

### Path B: Use the full web app
```bash
pip install -e ".[dev]"
# Terminal 1:
uvicorn depthcraft.api.main:app --port 8000 --reload
# Terminal 2:
python -m depthcraft.visualization.gradio_app
# Open: http://localhost:7860
```

### Path C: Best depth quality (needs internet, GPU recommended)
```bash
pip install -e ".[dev,depth]"
uvicorn depthcraft.api.main:app --port 8000 --reload
# First run downloads ~1.3 GB model weights automatically
```

---

## Common problems and fixes

**"No module named 'depthcraft'"**
→ You forgot to install the package, or your venv isn't active.
```bash
source .venv/bin/activate
pip install -e ".[dev]"
```

**"Port 8000 already in use"**
→ Something else is using that port. Either stop it, or use a different port:
```bash
uvicorn depthcraft.api.main:app --port 8001 --reload
```

**"WARNING: backend: fallback"**
→ This is normal without the `depth` extra. The app still works, depth quality
is just lower. Install `pip install -e ".[depth]"` for the real model.

**"No display" error when running without `--headless`**
→ You're on a server or headless machine. Always add `--headless` in that case.

**Tests fail with import errors**
→ Make sure you're in the right folder and venv is active:
```bash
cd depthcraft          # must be in the project root
source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v
```

**"ModuleNotFoundError: No module named 'torch'"**
→ Expected if you haven't installed the `depth` extra. The app uses the
fallback estimator automatically — this is not an error, just an info warning.

---

## Quick reference card

```
SETUP (once):
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -e ".[dev]"
  pytest tests/ -v              ← should say 22 passed

EVERY DAY:
  source .venv/bin/activate     ← always do this first

RUN DEMO:
  python mvp_demo.py --image photo.jpg --headless

RUN APP:
  uvicorn depthcraft.api.main:app --port 8000 --reload   (terminal 1)
  python -m depthcraft.visualization.gradio_app           (terminal 2)
  → open http://localhost:7860

UPGRADE DEPTH QUALITY:
  pip install -e ".[depth]"     ← downloads ~3 GB, then runs better
```
