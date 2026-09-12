"""Modal deployment. Disabled schedule by default; no credentials in source."""
import os
from pathlib import Path

import modal

ROOT = Path(__file__).parent
ENABLED = os.environ.get("PAPERLAB_SCHEDULE") == "1"
FULL_FLY = os.environ.get("PAPERLAB_FLY", "1") == "1"
UNIVERSE = os.environ.get("PAPERLAB_UNIVERSE", "0") == "1"
PAPER_STUDY = os.environ.get('PAPERLAB_PAPER_STUDY') == '1'
ONLINE_STUDY = os.environ.get('PAPERLAB_ONLINE_STUDY') == '1'
if ONLINE_STUDY and not (PAPER_STUDY and FULL_FLY and UNIVERSE):
    raise ValueError('Online study requires paper study, full fly and universe flags')
PRODUCT = os.environ.get("PAPERLAB_PRODUCT", "BTC-USD")
if PRODUCT not in ("BTC-USD", "ETH-USD"):
    raise ValueError("Unsupported product")

app = modal.App("fly-paper-lab")
volume = modal.Volume.from_name("fly-paper-lab-state", create_if_missing=True)
discovery_volume = modal.Volume.from_name("fly-paper-lab-universe", create_if_missing=True)
writers = modal.Dict.from_name("fly-paper-lab-writers", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("g++")
         .pip_install("torch==2.14.0", index_url="https://download.pytorch.org/whl/cpu")
         .pip_install("numpy==2.5.3", "requests==2.34.2", "feedparser==6.0.14", "defusedxml==0.7.1", "pillow==12.3.0", "pandas==3.0.5", "pyarrow==25.0.1", "transformers==5.17.0")
         .add_local_dir(ROOT / "paperlab", "/opt/paperlab/paperlab", copy=True, ignore=["__pycache__/"])
         .add_local_dir(ROOT / "vendor", "/opt/paperlab/vendor", copy=True, ignore=["__pycache__/"])
         .add_local_file(ROOT / "reports/fly-market-study-08-preregistration.json", "/opt/paperlab/registered/fly-market-study-08-preregistration.json", copy=True)
         .add_local_file(ROOT / "reports/fly-market-study-07-plan.json", "/opt/paperlab/registered/fly-market-study-07-plan.json", copy=True)
         .env({"PYTHONPATH": "/opt/paperlab", "OMP_NUM_THREADS": "2", "HF_HOME": "/state/huggingface", "PAPERLAB_FINBERT_REVISION": "4556d13015211d73dccd3fdd39d39232506f3e43", "PAPERLAB_PRODUCT": PRODUCT, "PAPERLAB_FLY": "1" if FULL_FLY else "0", "PAPERLAB_UNIVERSE":"1" if UNIVERSE else "0"}))

if PAPER_STUDY:
    for name in ('fly-market-study-09-preregistration.json','fly-market-study-08-plan.json','fly-market-study-08.json',
                 'fly-market-study-10-preregistration.json','fly-market-study-09-plan.json','fly-market-study-09.json',
                 'fly-paper-stimulation-study-01.json','fly-paper-stimulation-audit-01.json'):
        image=image.add_local_file(ROOT/'reports'/name,'/opt/paperlab/registered/'+name,copy=True)
    image=image.add_local_file(ROOT/'reports/fly-paper-memory-audit-01.json','/opt/paperlab/paper-memory-01/audit.json',copy=True)
    for i in range(2):
        image=image.add_local_file(ROOT/f'runs/reward-exposure-01/verified-export/pool{i}-memory.npz',f'/opt/paperlab/paper-memory-01/pool{i}-memory.npz',copy=True)
    image=image.env({'PAPERLAB_PAPER_STUDY':'1'})

if ONLINE_STUDY:
    for name in ('fly-market-study-11-preregistration.json','fly-market-study-10-plan.json',
                 'fly-market-study-10.json','fly-credit-reset-study-01.json','fly-credit-reset-audit-01.json'):
        image=image.add_local_file(ROOT/'reports'/name,'/opt/paperlab/registered/'+name,copy=True)
    image=image.add_local_file(ROOT/'cloud.py','/opt/paperlab/registered/cloud-source.py',copy=True)
    image=image.env({'PAPERLAB_ONLINE_STUDY':'1'})


def exclusive(name, call, *args):
    """Atomic cloud ownership also covers overlapping deployment versions.

    A hard-killed owner leaves a fail-closed lock. Reclaim only after verifying
    that its Modal input has ended; never expire a possibly live SQLite writer.
    """
    import time
    owner={"started":time.time(),"call_id":modal.current_function_call_id(),"input_id":modal.current_input_id()}
    if not writers.put(name,owner,skip_if_exists=True):
        print(f"Writer {name} already owned; skipped overlapping invocation",flush=True)
        return {"status":"writer_busy","writer":name}
    try:
        return call(*args)
    finally:
        writers.pop(name)


WORKER_MEMORY=8192 if FULL_FLY and UNIVERSE else 16384 if FULL_FLY else 4096


# Native libraries and model readers can retain volume handles after a call.
# Fresh containers make the next reload safe, including back-to-back diagnostic jobs.
@app.function(image=image, volumes={"/state": volume,"/discovery":discovery_volume}, cpu=(2,2), memory=(WORKER_MEMORY,WORKER_MEMORY),
              max_containers=1, min_containers=0, scaledown_window=2, timeout=600, single_use_containers=True,
              schedule=modal.Cron("*/5 * * * *") if ENABLED else None, retries=0)
def worker(prepare: bool = False, probe: bool = False, diagnostics: bool = False, debug: dict | None = None):
    return exclusive("worker",_worker,prepare,probe,diagnostics,debug)


def _worker(prepare=False,probe=False,diagnostics=False,debug=None):
    import os
    import sys
    import time
    worker_started=time.monotonic()
    sys.path.insert(0, "/opt/paperlab")
    from paperlab.runtime import cycle
    from paperlab.budget import reserve, settle
    from paperlab.core import atomic_json
    from paperlab.telemetry import emit
    full = os.environ["PAPERLAB_FLY"] == "1"
    if sum((prepare, probe, diagnostics, debug is not None)) > 1:
        raise ValueError("Prepare and probe are separate bounded invocations")
    if debug is not None:
        from paperlab.cloud_debug import validate_request
        validate_request(debug)
        if not full:
            raise ValueError("Fly diagnostics require the full retained graph")
    volume.reload()
    # Match the explicitly saved provider usage cap; never rely on credits to expand it.
    # Measured cold starts were ~3 seconds. Ten seconds plus the unchanged 2x
    # rate margin accommodates faster scheduling without a 30-second floor per tick.
    universe = os.environ.get("PAPERLAB_UNIVERSE") == "1"
    reservation = reserve("/state/budget.json", full, limit_override=25 if universe else 40, startup_seconds=10,
                          memory_gib=8 if universe and full else 16 if full else 4)
    if reservation is None:
        emit("budget_stopped", action="Disable schedule; reserved compute limit reached")
        return {"status": "budget_stopped", "action": "Remove the schedule and redeploy. Compute estimate reached its reserved limit."}
    volume.commit()  # Persist worst-case reservation before expensive work; crashes retain it.
    if full and universe and not any((prepare, probe, diagnostics, debug is not None)):
        from paperlab.fly_market_schedule import execute_due
        from paperlab.cloud_debug import run
        discovery_volume.reload()
        scheduled = execute_due('/state', '/discovery', '/opt/paperlab/registered',
            call_id=modal.current_function_call_id(), input_id=modal.current_input_id(),
            commit=volume.commit, run=run,
            settle_budget=lambda: settle('/state/budget.json', reservation, time.time() - reservation['started']))
        if scheduled is not None:
            return scheduled  # Give the bounded study this entire worker invocation.
        if os.environ.get('PAPERLAB_PAPER_STUDY')=='1':
            from paperlab.fly_paper_schedule import execute_due as paper_due
            from paperlab.fly_paper_study import run as paper_run
            for study in ('09','10'):
                scheduled=paper_due('/state','/discovery','/opt/paperlab/registered','/opt/paperlab/paper-memory-01',
                    call_id=modal.current_function_call_id(),input_id=modal.current_input_id(),commit=volume.commit,
                    run=paper_run,settle_budget=lambda:settle('/state/budget.json',reservation,time.time()-reservation['started']),study=study)
                if scheduled is not None:return scheduled
    if debug is not None:
        from paperlab.cloud_debug import run
        try:
            result = run(debug, "/state/fly-debugger", "/state/fly-data")
            result["budget"] = settle("/state/budget.json", reservation, time.time() - reservation["started"])
            atomic_json(Path("/state/fly-debugger") / debug["run_id"] / "cloud-result.json", result)
            volume.commit()
            return result
        except Exception as exc:
            emit("fly_debug_failed", error_type=type(exc).__name__, error=str(exc)[:300])
            volume.commit()
            raise
    if diagnostics:
        try:
            if universe:
                from paperlab.universe_probe import run
                result=run("/state/meme-probes","/state/fly-data")
            else:
                from paperlab.diagnostics import run
                result = run("/state/fast-5m", "/state/fly-data", full, os.environ["PAPERLAB_PRODUCT"])
            result["budget"] = settle("/state/budget.json", reservation, time.time() - reservation["started"])
            atomic_json("/state/meme-probes/last-call.json" if universe else "/state/fast-5m/diagnostics/last-call.json", result)
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
    output_root="/state/meme-pools-v1" if universe else "/state/fast-5m"
    try:
        # New cadence has its own accounts and archive. Original 15-minute state
        # and policies remain in /state; only immutable full-graph data is shared.
        if universe:
            from paperlab.multi import cycle as multi_cycle
            discovery_volume.reload()
            if not Path("/discovery/universe-snapshot.db").exists():
                result={"status":"waiting_for_universe_collector"}
            else:
                result=multi_cycle(output_root,"/discovery/universe-snapshot.db","/state/fly-data",use_fly=full)
        else:
            result = cycle("/state/fast-5m", os.environ["PAPERLAB_PRODUCT"], full, train_daily=True,
                           interval_seconds=300, train_every_seconds=21600, news_every_seconds=900,
                           fly_data="/state/fly-data", historical_warmup=True)
    except Exception as exc:
        emit("cycle_failed", error_type=type(exc).__name__, error=str(exc)[:300])
        volume.commit()
        raise
    result["cycle_seconds"] = time.time() - started
    if full and universe and os.environ.get('PAPERLAB_ONLINE_STUDY')=='1':
        # The normal account/checkpoint transaction becomes durable before a
        # diagnostic chunk. A failed or killed study cannot erase that cycle.
        if result['status']!='duplicate_or_old_slot':
            atomic_json('/state/latest.json',result)
            atomic_json(Path(output_root)/'latest.json',result)
        volume.commit()
        from paperlab.fly_online_schedule import execute_due as online_due
        from paperlab.fly_online_study import run_chunk
        try:
            result['study_11']=online_due('/state','/discovery','/opt/paperlab/registered','/opt/paperlab/paper-memory-01',
                call_id=modal.current_function_call_id(),input_id=modal.current_input_id(),commit=volume.commit,
                run=run_chunk,deadline=worker_started+570,
                allow_compute=result['status']=='paper_research' and result['cycle_seconds']<=90)
        except Exception as exc:
            emit('paper_online_failed',error_type=type(exc).__name__,error=str(exc)[:300])
            volume.commit()
            raise
    result["budget"] = settle("/state/budget.json", reservation, time.time() - reservation["started"])
    atomic_json("/state/last-call.json", result)
    atomic_json(Path(output_root)/"last-call.json", result)
    if result["status"] != "duplicate_or_old_slot":
        atomic_json("/state/latest.json", result)
        atomic_json(Path(output_root)/"latest.json", result)
    emit("worker_completed", status=result["status"], cycle_seconds=result["cycle_seconds"],
         estimated_compute_usd=result["budget"]["estimated_compute_usd"],
         monthly_reserved_usd=result["budget"]["monthly_reserved_usd"])
    volume.commit()
    return result


# The collector has its own volume and budget file, so no second writer can touch
# the trader's SQLite/checkpoints. No torch, transformer, or fly graph is loaded here.
collector_image=(modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy==2.5.3","requests==2.34.2","websockets==16.0")
    .add_local_dir(ROOT/"paperlab","/opt/paperlab/paperlab",copy=True,ignore=["__pycache__/"])
    .env({"PYTHONPATH":"/opt/paperlab","OPENBLAS_NUM_THREADS":"1"}))


@app.function(image=collector_image,volumes={"/discovery":discovery_volume,"/state":volume},
              cpu=(.125,.125),memory=(256,256),max_containers=1,min_containers=0,
              scaledown_window=2,timeout=360,retries=0,single_use_containers=True,
              schedule=modal.Cron("*/5 * * * *") if ENABLED and UNIVERSE else None)
def universe_collector():
    return exclusive("collector",_universe_collector)


def _universe_collector():
    import asyncio
    import json
    import sys
    import time
    sys.path.insert(0,"/opt/paperlab")
    from paperlab.universe import collect_window
    from paperlab.budget import reserve,settle
    from paperlab.core import atomic_json
    discovery_volume.reload()
    volume.reload()
    reservation=reserve("/discovery/budget.json",False,seconds=360,limit_override=15,
                        startup_seconds=10,cpu=.125,memory_gib=.25)
    if reservation is None:
        return {"status":"collector_budget_stopped"}
    discovery_volume.commit()
    def priorities():
        path=Path("/state/meme-pools-v1/watch.json")
        return json.loads(path.read_text()) if path.exists() else []
    try:
        result=asyncio.run(collect_window("/discovery",seconds=240,publish=discovery_volume.commit.aio,priorities=priorities))
    except Exception:
        discovery_volume.commit()  # Retain worst-case charge and received launch events.
        raise
    result["budget"]=settle("/discovery/budget.json",reservation,time.time()-reservation["started"])
    atomic_json("/discovery/last-window.json",result)
    discovery_volume.commit()
    return result


@app.local_entrypoint()
def main(prepare: bool = False, probe: bool = False, diagnostics: bool = False):
    print(worker.remote(prepare, probe, diagnostics))
