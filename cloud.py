"""Modal deployment. Disabled schedule by default; no credentials in source."""
import os
from pathlib import Path

import modal

ROOT = Path(__file__).parent
ENABLED = os.environ.get("PAPERLAB_SCHEDULE") == "1"
FULL_FLY = os.environ.get("PAPERLAB_FLY", "1") == "1"
PRODUCT = os.environ.get("PAPERLAB_PRODUCT", "BTC-USD")
if PRODUCT not in ("BTC-USD", "ETH-USD"):
    raise ValueError("Unsupported product")

app = modal.App("fly-paper-lab")
volume = modal.Volume.from_name("fly-paper-lab-state", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("g++")
         .pip_install("numpy==2.5.3", "torch==2.14.0", "requests==2.34.2", "feedparser==6.0.14", "defusedxml==0.7.1", "pillow==12.3.0", "pandas==3.0.5", "pyarrow==25.0.1")
         .add_local_dir(ROOT / "paperlab", "/opt/paperlab/paperlab", copy=True, ignore=["__pycache__/"])
         .add_local_dir(ROOT / "vendor", "/opt/paperlab/vendor", copy=True, ignore=["__pycache__/"])
         .env({"PYTHONPATH": "/opt/paperlab", "OMP_NUM_THREADS": "2", "PAPERLAB_PRODUCT": PRODUCT, "PAPERLAB_FLY": "1" if FULL_FLY else "0"}))


@app.function(image=image, volumes={"/state": volume}, cpu=2, memory=16384 if FULL_FLY else 4096,
              max_containers=1, min_containers=0, scaledown_window=2, timeout=3300,
              schedule=modal.Cron("*/15 * * * *") if ENABLED else None, retries=0)
def worker(prepare: bool = False):
    import os
    import sys
    sys.path.insert(0, "/opt/paperlab")
    from paperlab.runtime import cycle
    full = os.environ["PAPERLAB_FLY"] == "1"
    volume.reload()
    if prepare:
        if full:
            from paperlab.fly import prepare as prepare_graph
            prepare_graph("/state/fly-data")
        volume.commit()
        return {"status": "prepared", "full_fly": full}
    result = cycle("/state", os.environ["PAPERLAB_PRODUCT"], full, train_daily=True)
    volume.commit()
    return result


@app.local_entrypoint()
def main(prepare: bool = False):
    print(worker.remote(prepare))
