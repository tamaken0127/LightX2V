__version__ = "0.1.0"
__author__ = "LightX2V Contributors"
__license__ = "Apache 2.0"

import time
_t0 = time.time()

import lightx2v_platform.set_ai_device
print(f"[Timing] lightx2v_platform.set_ai_device: {time.time()-_t0:.2f}s", flush=True)
_t0 = time.time()

from lightx2v import common
print(f"[Timing] lightx2v.common: {time.time()-_t0:.2f}s", flush=True)
_t0 = time.time()

from lightx2v import models
print(f"[Timing] lightx2v.models: {time.time()-_t0:.2f}s", flush=True)
_t0 = time.time()

from lightx2v import utils
print(f"[Timing] lightx2v.utils: {time.time()-_t0:.2f}s", flush=True)
_t0 = time.time()

from lightx2v.pipeline import LightX2VPipeline
print(f"[Timing] lightx2v.pipeline: {time.time()-_t0:.2f}s", flush=True)

__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "models",
    "common",
    "utils",
    "LightX2VPipeline",
]
