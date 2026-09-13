"""Detached, bounded cloud-only Solana live paper pilot; no recurring spend by default."""
from pathlib import Path
import modal
from cloud import image as base_image, volume, exclusive

ROOT=Path(__file__).resolve().parent if modal.is_local() else Path('/opt/paperlab')
app=modal.App('fly-paper-solana-5s')
image=(base_image.pip_install('websockets==16.0','solders==0.29.0')
       .add_local_file(ROOT/'cloud.py','/opt/paperlab/cloud.py',copy=True)
       .add_local_file(ROOT/'solana_cloud.py','/opt/paperlab/solana_cloud.py',copy=True))


def _verify_import():
    from paperlab.solana_events import SCHEMA, PROGRAM
    from paperlab.solana_paper import Readout, FEATURES
    assert PROGRAM==SCHEMA['address'] and FEATURES==22
    Readout()
    print('Solana collector and trainable readout imports verified; no native brain constructed',flush=True)


image=image.run_function(_verify_import,cpu=.125,memory=1024,timeout=90)


@app.function(image=image,volumes={'/state':volume},cpu=(2,2),memory=(8192,8192),
              max_containers=1,min_containers=0,scaledown_window=2,timeout=1200,
              single_use_containers=True,retries=0,nonpreemptible=True)
def worker(run_id: str, seconds: int=900, probe: bool=False, resume_id: str | None=None, remaining_windows: int=0):
    return exclusive('worker',_window,run_id,seconds,probe,resume_id,remaining_windows)


def _window(run_id,seconds,probe,resume_id,remaining_windows):
    if type(remaining_windows) is not int or not 0<=remaining_windows<=2:
        raise ValueError('At most three bounded windows per dispatch')
    result=_run(run_id,seconds,probe,resume_id)
    if not probe and result.get('status')=='completed' and remaining_windows:
        # The single-container function queues its successor; it cannot run until
        # this call returns and releases the writer lease. Do not retry failures.
        next_id=run_id+'-next'
        call=worker.spawn(next_id,seconds=seconds,resume_id=run_id,remaining_windows=remaining_windows-1)
        from paperlab.core import atomic_json
        result['next_call_id']=call.object_id;result['next_run_id']=next_id
        atomic_json(Path('/state/solana-live')/run_id/'next.json',
                    {'call_id':call.object_id,'run_id':next_id,'remaining_windows':remaining_windows-1})
        volume.commit()
    return result


def _run(run_id,seconds,probe,resume_id=None):
    import json,re,time
    from paperlab.budget import reserve,settle
    from paperlab.core import atomic_json,digest
    from paperlab.solana_paper import run
    from paperlab.solana_events import Feed
    if not re.fullmatch(r'solana-[a-z0-9-]{1,60}',run_id): raise ValueError('Invalid run id')
    if not 60<=seconds<=900: raise ValueError('Bound pilot duration')
    if resume_id is not None and not re.fullmatch(r'solana-[a-z0-9-]{1,60}',resume_id):raise ValueError('Invalid parent run')
    if probe and resume_id:raise ValueError('A feed probe cannot resume a trader')
    volume.reload();state=Path('/state');root=state/'solana-live'/run_id
    if root.exists(): raise ValueError('Preserve existing pilot; do not restart run id')
    # Reserve the FULL function timeout at 3x nonpreemptible rates, plus a 2x
    # uncertainty margin. A killed call retains this worst-case reservation.
    reservation=reserve(state/'budget.json',True,seconds=3600,startup_seconds=30,
                        memory_gib=8,limit_override=25)
    if reservation is None:return {'status':'budget_stopped'}
    reservation['rate']*=3;reservation['startup_seconds']=10
    root.mkdir(parents=True)
    atomic_json(root/'owner.json',dict(call_id=modal.current_function_call_id(),input_id=modal.current_input_id(),
        run_id=run_id,probe=probe,at=time.time(),source_sha256={
            n:digest('/opt/paperlab/'+n) for n in ('solana_cloud.py','paperlab/solana_events.py',
                    'paperlab/solana_paper.py','paperlab/solana_schema/pump_events.json',
                    'paperlab/solana_schema/pump_amm_events.json')},reservation=reservation,resume_id=resume_id))
    volume.commit()
    try:
        if probe:
            feed=Feed(root);feed.start();time.sleep(min(seconds,60));feed.close()
            result={'status':'feed_probe_completed','health':feed.health(),'launches':len(feed.snapshot()),'native_observations':0}
            atomic_json(root/'result.json',result)
        else:
            result=run(root,'/state/fly-data',seconds=seconds,commit=volume.commit,
                       resume=state/'solana-live'/resume_id if resume_id else None)
        result['budget']=settle(state/'budget.json',reservation,time.time()-reservation['started'])
        result['budget']['provider_invoice']=False
        atomic_json(root/'completed.json',result);volume.commit();return result
    except BaseException as exc:
        atomic_json(root/'failed.json',{'error_type':type(exc).__name__,'at':time.time(),
                    'reservation_retained':True,'message':str(exc)[:300]})
        volume.commit();raise
