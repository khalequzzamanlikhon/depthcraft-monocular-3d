"""Run a DepthCraft script or module with the model-load timeout raised.

depth_anything_engine.MODEL_LOAD_TIMEOUT_S is 20 s; a cold GPU load of the 1.3 GB
Metric-Indoor-Large checkpoint can exceed that, and the engine then silently switches
to the classical fallback estimator. Patching the module global keeps the repo untouched.

    python patched_run.py mvp_demo.py --image x.jpg --headless
    python patched_run.py -m uvicorn depthcraft.api.main:app --port 8765
"""
import os
import runpy
import sys

import depthcraft.depth.depth_anything_engine as engine_module

engine_module.MODEL_LOAD_TIMEOUT_S = float(os.environ.get("DEPTH_LOAD_TIMEOUT_S", "600"))
sys.path.insert(0, os.getcwd())

if sys.argv[1] == "-m":
    module = sys.argv[2]
    sys.argv = [module] + sys.argv[3:]
    runpy.run_module(module, run_name="__main__", alter_sys=True)
else:
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(script, run_name="__main__")
