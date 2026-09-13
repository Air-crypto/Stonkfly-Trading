"""Modal scheduled paper service: light coordinator, bounded full-fly windows."""
import os
from pathlib import Path
import modal
from cloud import exclusive, volume
from solana_cloud import image as base_image

ROOT = Path(__file__).resolve().parent if modal.is_local() else Path('/opt/paperlab')
app = modal.App('fly-paper-solana-online')
image = base_image.add_local_file(ROOT/'solana_online_cloud.py', '/opt/paperlab/solana_online_cloud.py', copy=True)
MODE = os.environ.get('PAPERLAB_ONLINE_MODE', 'paced')
ENABLED = os.environ.get('PAPERLAB_ONLINE_SCHEDULE') == '1'


@app.function(image=image, volumes={'/state':volume}, cpu=(2,2), memory=(8192,8192),
              timeout=1200, max_containers=1, min_containers=0, retries=0,
              single_use_containers=True, nonpreemptible=True)
def worker(run_id: str, parent: str):
    return exclusive('worker', _run, run_id, parent)


def _run(run_id, parent):
    import json, re, time
    from paperlab.core import atomic_json, digest
    from paperlab.budget import reserve, settle
    from paperlab.solana_online import run
    if not re.fullmatch(r'solana-online-[0-9-]+',run_id) or not re.fullmatch(r'solana-[a-z0-9-]{1,60}',parent):
        raise ValueError('Invalid immutable run or parent id')
    volume.reload(); state=Path('/state'); root=state/'solana-live'/run_id
    if root.exists(): raise ValueError('Never restart an existing window')
    reservation=reserve(state/'budget.json',True,seconds=3600,startup_seconds=30,memory_gib=8,limit_override=25)
    if reservation is None:return dict(status='budget_stopped')
    reservation['rate']*=3;reservation['startup_seconds']=10
    root.mkdir(parents=True)
    atomic_json(root/'owner.json',dict(call_id=modal.current_function_call_id(),parent=parent,
        run_id=run_id,at=time.time(),reservation=reservation,paper_only=True,source_sha256={
            n:digest('/opt/paperlab/'+n) for n in ('solana_online_cloud.py','paperlab/solana_online.py',
                                                'paperlab/solana_events.py','paperlab/solana_service.py')}))
    volume.commit()
    try:
        result=run(root,'/state/fly-data',state/'solana-live'/parent,seconds=900,commit=volume.commit)
        result['budget']=settle(state/'budget.json',reservation,time.time()-reservation['started'])
        atomic_json(root/'completed.json',result);volume.commit();return result
    except BaseException as exc:
        atomic_json(root/'failed.json',dict(at=time.time(),error_type=type(exc).__name__,
            message=str(exc)[:300],reservation_retained=True));volume.commit();raise


@app.function(image=image, volumes={'/state':volume}, cpu=(.125,.125), memory=(512,512),
              timeout=60, max_containers=1, min_containers=0, retries=0, single_use_containers=True,
              schedule=modal.Cron('*/15 * * * *') if ENABLED else None)
def coordinator():
    return exclusive('solana-online-coordinator', _coordinate)


def _coordinate():
    import json
    from paperlab.solana_service import service
    volume.reload()
    def poll(call_id):
        try:return modal.FunctionCall.from_id(call_id).get(timeout=0)
        except TimeoutError:return None
        except Exception as exc:return dict(status='failed',error_type=type(exc).__name__)
    result=service('/state',dispatch=lambda run_id,parent:worker.spawn(run_id,parent).object_id,
                   poll=poll,commit=volume.commit,mode=MODE)
    print(json.dumps(result),flush=True);return result


@app.function(image=image, volumes={'/state':volume}, cpu=(.125,.125), memory=(512,512),
              timeout=60, max_containers=1, min_containers=0, retries=0, single_use_containers=True)
def set_cadence(mode: str, start_now: bool=False):
    """Explicit operator action, serialized with scheduled dispatch."""
    return exclusive('solana-online-coordinator', _set_cadence, mode, start_now)


def _set_cadence(mode, start_now):
    from paperlab.solana_service import configure_cadence
    volume.reload()
    configure_cadence('/state', mode, commit=volume.commit, start_now=start_now)
    return _coordinate()
