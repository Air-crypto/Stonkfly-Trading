"""One-minute lightweight dispatcher calling the unchanged sealed Modal worker."""
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parent if modal.is_local() else Path("/opt/paperlab")
app = modal.App('fly-paper-checkpoint-dispatch')
volume = modal.Volume.from_name('fly-paper-lab-state')
writers = modal.Dict.from_name('fly-paper-lab-writers')
# No native runtime or model load in this scheduling container.
image = (modal.Image.debian_slim(python_version='3.12')
         .add_local_file(ROOT/'checkpoint_eval_dispatch.py', '/opt/paperlab/checkpoint_eval_dispatch.py', copy=True)
         .env({'PYTHONPATH': '/opt/paperlab'}))


def poll_completed(call_id):
    try:
        result = modal.FunctionCall.from_id(call_id).get(timeout=0)
    except TimeoutError:
        return None
    # Completion can race the reload at invocation startup.
    volume.reload()
    return result


@app.function(image=image, volumes={'/state': volume}, cpu=(.125,.125), memory=(512,512),
              timeout=60, max_containers=1, min_containers=0, retries=0,
              single_use_containers=True, schedule=modal.Cron('* * * * *'))
def coordinator():
    import time
    from checkpoint_eval_dispatch import dispatch, reserve_dispatch, settle_dispatch
    key = 'checkpoint-eval-coordinator'
    owner = dict(started=time.time(), call_id=modal.current_function_call_id(),
                 input_id=modal.current_input_id(), dispatcher='priority_frozen_dispatch_v1')
    if not writers.put(key, owner, skip_if_exists=True):
        return dict(status='writer_busy', writer=key)
    try:
        volume.reload()
        reservation = reserve_dispatch('/state', time.time())
        if reservation is None:
            return dict(status='dispatcher_budget_stopped')
        volume.commit()
        def refresh_training():
            modal.Function.from_name('fly-paper-solana-online', 'coordinator', environment_name='main').remote()
            volume.reload()
            from checkpoint_eval_dispatch import read
            return read('/state/solana-online/control.json')
        def spawn(batch, policy):
            return modal.Function.from_name('fly-paper-checkpoint-eval', 'worker', environment_name='main').spawn(batch, policy).object_id
        result = dispatch('/state', poll=poll_completed, spawn=spawn, commit=volume.commit,
                          refresh_training=refresh_training, worker_busy=lambda: bool(writers.get('worker')))
        result = dict(result, dispatch_budget=settle_dispatch('/state', reservation, time.time()))
        volume.commit()
        print({k: result.get(k) for k in ('status','batch','pending','enabled')}, flush=True)
        return result
    finally:
        writers.pop(key)


@app.function(image=image, volumes={'/state': volume}, cpu=(.125,.125), memory=(512,512),
              timeout=60, max_containers=1, min_containers=0, retries=0, single_use_containers=True)
def recover_result(expected_call_id):
    import time
    from checkpoint_eval_dispatch import recover_result as recover, reserve_dispatch, settle_dispatch
    key='checkpoint-eval-coordinator'
    owner=dict(started=time.time(),call_id=modal.current_function_call_id(),recovery=True)
    if not writers.put(key,owner,skip_if_exists=True):return dict(status='writer_busy',writer=key)
    reservation=None
    try:
        volume.reload()
        reservation=reserve_dispatch('/state',time.time())
        if reservation is None:return dict(status='dispatcher_budget_stopped')
        volume.commit()
        return recover('/state',expected_call_id,poll=poll_completed,commit=volume.commit)
    finally:
        try:
            if reservation is not None:
                settle_dispatch('/state',reservation,time.time());volume.commit()
        finally:
            writers.pop(key)
