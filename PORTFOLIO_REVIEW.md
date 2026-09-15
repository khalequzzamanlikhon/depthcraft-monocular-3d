# DepthCraft — Portfolio Review

## Short answer: yes, this is a strong portfolio project

It's not a toy. It implements real computer-vision and 3D-reconstruction
algorithms (not stubs pretending to work), is honest in its own docs about
what needs a GPU/internet vs. what runs anywhere, ships a tested FastAPI
backend + React/Three.js frontend + Docker + CI, and has 22 passing tests.
That combination — breadth (depth estimation → SfM/MVS → mesh/Gaussian
splatting → measurement → API → web viewer) plus honesty about what's real —
is more than most portfolio projects show, and is exactly the kind of thing
that reads well to a technical interviewer who actually opens the code.

What makes it land well specifically:
- **Correct math, not hand-waving.** E.g. the virtual-tape-measure uncertainty
  propagation in [measure_engine.py](depthcraft/measurement/measure_engine.py)
  implements first-order error propagation properly (`sigma_d^2 = sum[(p1-p2)/d]^2 * (sigma1^2+sigma2^2)`),
  not just a placeholder distance calculation.
- **Graceful degradation as a design pattern, applied consistently.** Every
  GPU/heavy-dependency component (Depth Anything V2, SAM 2, COLMAP, 3DGS)
  has a real fallback path so the whole pipeline runs end-to-end on a laptop
  with no GPU — and the code says so honestly instead of silently mocking
  results.
- **It's a full stack, not just a notebook.** FastAPI backend with routers,
  Pydantic schemas, an in-memory job store, a Gradio demo, a React/Three.js
  frontend, Dockerfiles (GPU and CPU), and a GitHub Actions CI pipeline.

## What I fixed before setting this up for GitHub

While reviewing I ran the actual test/lint/type-check suite (not just read
the code) and found a few real issues, now fixed:

1. **Startup could hang forever with no internet.** [depth_anything_engine.py](depthcraft/depth/depth_anything_engine.py)
   tries to load the real Depth Anything V2 model via Hugging Face when
   `torch`/`transformers` are installed. If the network is reachable but the
   download stalls (common on restricted networks/sandboxes), the original
   code had no timeout and blocked indefinitely — I measured this hanging
   for 7+ minutes in testing before I traced it. Fixed by running the load in
   a background daemon thread with a 20s wall-clock budget; if it's not done
   in time, the engine falls back to the classical estimator and logs why.
   (A plain `ThreadPoolExecutor` doesn't work for this — it registers an
   atexit hook that joins its worker threads on interpreter shutdown, so the
   process would still hang at exit even after "giving up" on the timeout.)

2. **A real bug in `FeatureMatcher.__init__`** ([matcher.py](depthcraft/sfm/matcher.py)):
   `self._superglue = None` ran *after* `self._resolve_backend(...)`, which
   is exactly the call that sets `self._superglue` to the real SuperGlue
   model when available. So even when SuperGlue loaded successfully, it was
   immediately overwritten with `None` — the `superglue` backend would have
   crashed with `'NoneType' object is not callable` the first time it was
   used. Fixed by reordering the two lines.

3. **Lint/type-check were not actually clean.** `ruff check` had 7 findings
   (unused variables, an unused import, a bare except) and `mypy` had 14
   errors (mostly the classic "`self._x = None` in `__init__`, then called
   as if it were the real object elsewhere" pattern, which mypy correctly
   flags because it can't see the cross-method invariant). Since
   `.github/workflows/ci.yml` runs both `ruff check` and `mypy` as blocking
   CI steps, **CI would have gone red on first push.** Both are now clean;
   verified locally: `ruff check depthcraft/` → *All checks passed!*,
   `mypy depthcraft/ --ignore-missing-imports` → *Success: no issues found*,
   `pytest tests/ -v` → *22 passed*.

4. **No git repo, no LICENSE, no visuals in the README.** The project lived
   as a bare folder with no version control. Initialized git, added an MIT
   `LICENSE`, embedded the point-cloud screenshot you already had into the
   README, and added CI/license/Python badges (the CI badge needs your
   GitHub username filled in once you push — see README §9).

5. **Stray tool artifacts.** A `.freebuff/` directory containing opaque
   internal project-id UUIDs (unrelated to the codebase, likely left by
   whatever tool scaffolded this project) was excluded via `.gitignore`
   rather than committed.

## Remaining suggestions (not blocking, ordered by payoff)

These are worth doing before/soon after you publish, but none of them are
"the project is broken" — they're polish that will make it read even better.

1. **Frontend UI is functional but bare.** [App.tsx](frontend/src/App.tsx) is
   inline-styled with no layout/visual design. For a portfolio piece where
   people will actually click around, 30–60 minutes of basic styling (even
   just a CSS framework like Tailwind via a `<link>` or a couple of flexbox
   rules, a header, spacing) would make a much better first impression than
   the code quality alone can carry. This matters more than any backend
   polish because it's the first thing a non-technical reviewer sees.
2. **`SplatViewer.tsx` is an intentional stub** (clearly commented as such,
   which is good practice) — but consider either wiring up
   `@mkkellogg/gaussian-splats-3d` for a real demo of the 3DGS path, or
   removing it from the default UI until a `.splat` export flow exists, so
   a visitor doesn't wonder why there's a dead component in the shipped app.
3. **README `[✅ Fully real]` claims are accurate for CPU-only paths, but
   double check them once more after a clean-machine run** — I verified
   tests/lint/types locally in *this* environment (which happened to already
   have `torch`+`transformers` installed), but I have not exercised the GPU
   paths (3DGS training, TensorRT export, real COLMAP SfM) myself. Worth a
   one-time end-to-end run on a CUDA box before you claim those work, if you
   haven't already.
4. **Add a short demo GIF/video** (even a 10-second screen recording of the
   Gradio app) alongside the static screenshot — portfolio reviewers weigh
   "I can see it work in 10 seconds" heavily, more than reading a features
   table.
5. **Pin dependency versions more tightly for reproducibility**, or add a
   lockfile (`pip-compile` / `uv.lock`). Current `pyproject.toml` uses `>=`
   ranges everywhere, which is fine for a library but means "works today"
   isn't guaranteed to mean "works in a year" without re-testing.
6. **Minor:** the `.venv/` in your working folder is currently ~1.9 GB
   (includes `torch`, `transformers`, `open3d`, `gradio`, etc. — more than
   the base `[dev]` install pulls in). It's correctly excluded by
   `.gitignore`, so this doesn't affect the repo, just noting it so you're
   not surprised by install time on a fresh clone with the `[dev,depth]`
   extras.

## What NOT to worry about

- The classical-fallback depth estimator being "not metrically accurate" —
  it's clearly labeled as such in code comments and the README's "what's
  real" table. This kind of honesty is a *strength* in a portfolio piece,
  not a weakness to hide.
- `SAM2Masker.mask_moving_objects()` returning placeholder all-ones masks
  until prompts are seeded — also clearly commented, matches the documented
  scope (SAM 2 needs interactive/automatic prompt seeding that's out of
  scope for an MVP).
- Empty `calibration/`, `data/ground_truth_measurements/` directories —
  these already have `.gitkeep` files and are legitimately meant to be
  filled in by whoever runs the calibration workflow.

## Bottom line

Ship it. Fix the frontend polish (#1 above) if you have an extra hour, add
a demo GIF, then push using the guide now in [README.md §9](README.md#9-pushing-to-github).
The engineering underneath is already good enough to be the centerpiece of
a portfolio — the main thing standing between "good code" and "good
portfolio piece" was presentation (visuals, a working CI badge, a LICENSE)
and a couple of real bugs that would have embarrassed a first CI run, both
now handled.
