"""Independent, prepaid, three-day frozen forward paper experiment on Modal."""
import os
from pathlib import Path
import modal
from cloud import exclusive, volume
from solana_cloud import image as base_image

ROOT=Path(__file__).resolve().parent if modal.is_local() else Path('/opt/paperlab')
app=modal.App('fly-paper-frozen-forward')
forward_volume=modal.Volume.from_name('fly-paper-frozen-forward',create_if_missing=True)
image=base_image
for filename in ('frozen_forward.py','forward_service.py','frozen_forward_cloud.py'):
    image=image.add_local_file(ROOT/filename,'/opt/paperlab/'+filename,copy=True)
image=image.env({'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
ENABLED=os.environ.get('PAPERLAB_FORWARD_SCHEDULE')=='1'


@app.function(image=image,volumes={'/state':volume,'/forward':forward_volume},
              cpu=(1,1),memory=(8192,8192),timeout=1200,max_containers=1,min_containers=0,
              single_use_containers=True,retries=0,nonpreemptible=False)
def worker(session: str, seconds: int):
    return exclusive('frozen-forward-worker',_run,session,seconds)


def _run(session,seconds):
    import json,re,time
    from paperlab.core import atomic_json
    from frozen_forward import run
    if not re.fullmatch(r'session-[0-9]{5}',session):raise ValueError('Invalid session')
    forward_volume.reload();root=Path('/forward');control=json.loads((root/'control.json').read_text())
    if not control['enabled'] or not control.get('pending') or control['pending']['session']!=session:
        raise ValueError('No funded dispatch for this session')
    manifest=json.loads((root/'manifest.json').read_text());saved=json.loads((root/'state.json').read_text())
    if time.time()>=manifest['ends_at'] or control['spent_usd']>control['allowance_usd']:
        raise ValueError('Forward time or budget envelope ended')
    out=root/'sessions'/session
    try:
        result=run(out,'/state/fly-data',manifest,saved,seconds=seconds,commit=forward_volume.commit)
        # Raw SQLite is closed by the runner. Compress it losslessly to bound storage.
        import gzip,shutil
        raw=out/'events.db'
        if raw.exists():
            with raw.open('rb') as src,gzip.open(out/'events.db.gz','wb',compresslevel=1) as dst:
                shutil.copyfileobj(src,dst)
            raw.unlink()
        atomic_json(out/'completed.json',result);forward_volume.commit();return result
    except BaseException as exc:
        atomic_json(out/'failed.json',dict(at=time.time(),error_type=type(exc).__name__,message=str(exc)[:400],
                                         reservation_retained=True,paper_only=True))
        forward_volume.commit();raise


@app.function(image=image,volumes={'/forward':forward_volume},cpu=(.125,.125),memory=(512,512),
              timeout=60,max_containers=1,min_containers=0,single_use_containers=True,retries=0,
              schedule=modal.Period(minutes=1) if ENABLED else None)
def coordinator():
    result=exclusive('frozen-forward-coordinator',_coordinate)
    if result.get('enabled') is False:
        _stop_app()
    return result


def _coordinate():
    import json
    from forward_service import coordinate
    forward_volume.reload()
    def poll(call_id):
        if not call_id:return dict(status='missing_call_id')
        try:result=modal.FunctionCall.from_id(call_id).get(timeout=0)
        except TimeoutError:return None
        except Exception as exc:return dict(status='failed',error_type=type(exc).__name__)
        forward_volume.reload()  # Completion can race the initial volume snapshot.
        return result
    result=coordinate('/forward',dispatch=lambda session,seconds:worker.spawn(session,seconds).object_id,
                      poll=poll,commit=forward_volume.commit)
    print(json.dumps(result),flush=True);return result


@app.function(image=image,volumes={'/state':volume,'/forward':forward_volume},
              cpu=(.125,.125),memory=(512,512),timeout=120,max_containers=1,min_containers=0,
              single_use_containers=True,retries=0)
def start():
    return exclusive('frozen-forward-coordinator',lambda:exclusive('worker',_start))


def _start():
    from forward_service import fund
    volume.reload();forward_volume.reload()
    manifest=fund('/state','/forward',commit=volume.commit)
    forward_volume.commit()
    return dict(manifest=manifest,coordinator=_coordinate())


def _stop_app():
    """Stop this experiment's schedule after its durable terminal result is saved."""
    import asyncio
    from modal.client import _Client
    from modal_proto import api_pb2
    async def stop():
        client=await _Client.from_env()
        found=await client.stub.AppGetByDeploymentName(api_pb2.AppGetByDeploymentNameRequest(
            name='fly-paper-frozen-forward',environment_name='main'))
        if found.app_id:
            await client.stub.AppStop(api_pb2.AppStopRequest(app_id=found.app_id,
                source=api_pb2.APP_STOP_SOURCE_PYTHON_CLIENT))
    asyncio.run(stop())
