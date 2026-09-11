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
         .pip_install("torch==2.14.0", index_url="https://download.pytorch.org/whl/cpu")
         .pip_install("numpy==2.5.3", "requests==2.34.2", "feedparser==6.0.14", "defusedxml==0.7.1", "pillow==12.3.0", "pandas==3.0.5", "pyarrow==25.0.1", "transformers==5.17.0")
         .add_local_dir(ROOT / "paperlab", "/opt/paperlab/paperlab", copy=True, ignore=["__pycache__/"])
         .add_local_dir(ROOT / "vendor", "/opt/paperlab/vendor", copy=True, ignore=["__pycache__/"])
         .env({"PYTHONPATH": "/opt/paperlab", "OMP_NUM_THREADS": "2", "HF_HOME": "/state/huggingface", "PAPERLAB_FINBERT_REVISION": "4556d13015211d73dccd3fdd39d39232506f3e43", "PAPERLAB_PRODUCT": PRODUCT, "PAPERLAB_FLY": "1" if FULL_FLY else "0"}))


@app.function(image=image, volumes={"/state": volume}, cpu=2, memory=16384 if FULL_FLY else 4096,
              max_containers=1, min_containers=0, scaledown_window=2, timeout=600,
              schedule=modal.Cron("*/5 * * * *") if ENABLED else None, retries=0)
def worker(prepare: bool = False, probe: bool = False, diagnostics: bool = False):
    import os
    import sys
    import time
    sys.path.insert(0, "/opt/paperlab")
    from paperlab.runtime import cycle
    from paperlab.budget import reserve, settle
    from paperlab.core import atomic_json
    from paperlab.telemetry import emit
    full = os.environ["PAPERLAB_FLY"] == "1"
    if sum((prepare, probe, diagnostics)) > 1:
        raise ValueError("Prepare and probe are separate bounded invocations")
    volume.reload()
    # Match the explicitly saved provider usage cap; never rely on credits to expand it.
    # Measured cold starts were ~3 seconds. Ten seconds plus the unchanged 2x
    # rate margin accommodates faster scheduling without a 30-second floor per tick.
    reservation = reserve("/state/budget.json", full, limit_override=40, startup_seconds=10)
    if reservation is None:
        emit("budget_stopped", action="Disable schedule; reserved compute limit reached")
        return {"status": "budget_stopped", "action": "Remove the schedule and redeploy. Compute estimate reached its reserved limit."}
    volume.commit()  # Persist worst-case reservation before expensive work; crashes retain it.
    if diagnostics:
        from paperlab.diagnostics import run
        try:
            result = run("/state/fast-5m", "/state/fly-data", full, os.environ["PAPERLAB_PRODUCT"])
            result["budget"] = settle("/state/budget.json", reservation, time.time() - reservation["started"])
            atomic_json("/state/fast-5m/diagnostics/last-call.json", result)
            volume.commit()
            return result
        except Exception as exc:
            emit("diagnostics_failed", error_type=type(exc).__name__, error=str(exc)[:300])
            volume.commit()  # Retain the reservation and any completed diagnostic artifacts.
            raise
    if prepare:
        if full:
            from paperlab.fly import prepare as prepare_graph
            prepare_graph("/state/fly-data")
        # Bootstrap only the compact policy on closed historical candles. Its news features
        # remain empty for past dates; daily retraining later uses observed forward data.
        from paperlab.data import historical
        from paperlab.core import load_ticks, digest
        from paperlab.compact import train
        from paperlab.news import News
        historical("/state/bootstrap.jsonl", os.environ["PAPERLAB_PRODUCT"])
        news = News("/state/news.db")
        try:
            train(load_ticks("/state/bootstrap.jsonl"), news, "/state/bootstrap", dataset_sha=digest("/state/bootstrap.jsonl"))
        finally:
            news.db.close()
        from pathlib import Path
        active = Path("/state/active-policy.pt")
        active.with_suffix(".partial").write_bytes(Path("/state/bootstrap/policy.pt").read_bytes())
        active.with_suffix(".partial").replace(active)
        budget = settle("/state/budget.json", reservation, time.time() - reservation["started"])
        result = {"status": "prepared", "full_fly": full, "budget": budget}
        atomic_json("/state/bootstrap-status.json", result)
        volume.commit()
        return result
    if probe:
        if not full:
            raise ValueError("The native fly probe requires PAPERLAB_FLY=1")
        from paperlab.core import load_ticks
        from paperlab.fly import Fly, replay
        from paperlab.news import News
        ticks = load_ticks("/state/bootstrap.jsonl")
        news = News("/state/news.db")
        try:
            result = replay(ticks, news, "/state/fly-data", "/state/probe", steps=8)
            restored = Fly("/state/fly-data", learning=False, checkpoint="/state/probe/fly.npz")
            before = restored.controller.brain.memory()["sha256"]
            observation = restored.observe(ticks, 72, news, .1)
            assert restored.controller.brain.memory()["sha256"] == before
            result["frozen_restore_verified"] = True
            result["restored_observation"] = observation
        finally:
            news.db.close()
        result["budget"] = settle("/state/budget.json", reservation, time.time() - reservation["started"])
        atomic_json("/state/probe/validation.json", result)
        volume.commit()
        return result
    started = time.time()
    try:
        # New cadence has its own accounts and archive. Original 15-minute state
        # and policies remain in /state; only immutable full-graph data is shared.
        result = cycle("/state/fast-5m", os.environ["PAPERLAB_PRODUCT"], full, train_daily=True,
                       interval_seconds=300, train_every_seconds=21600, news_every_seconds=900,
                       fly_data="/state/fly-data", historical_warmup=True)
    except Exception as exc:
        emit("cycle_failed", error_type=type(exc).__name__, error=str(exc)[:300])
        volume.commit()
        raise
    result["cycle_seconds"] = time.time() - started
    result["budget"] = settle("/state/budget.json", reservation, time.time() - reservation["started"])
    atomic_json("/state/latest.json", result)
    atomic_json("/state/fast-5m/latest.json", result)
    emit("worker_completed", status=result["status"], cycle_seconds=result["cycle_seconds"],
         estimated_compute_usd=result["budget"]["estimated_compute_usd"],
         monthly_reserved_usd=result["budget"]["monthly_reserved_usd"])
    volume.commit()
    return result


@app.local_entrypoint()
def main(prepare: bool = False, probe: bool = False, diagnostics: bool = False):
    print(worker.remote(prepare, probe, diagnostics))
