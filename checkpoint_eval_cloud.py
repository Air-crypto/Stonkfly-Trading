"""Budgeted frozen-checkpoint evaluation, independent of the live trainer."""
from pathlib import Path
import os
import modal
from cloud import exclusive,volume
from solana_cloud import image as base_image
ROOT=Path(__file__).resolve().parent if modal.is_local() else Path('/opt/paperlab')
image=base_image.add_local_file(ROOT/'checkpoint_eval_cloud.py','/opt/paperlab/checkpoint_eval_cloud.py',copy=True).env(
    {'PAPERLAB_ALL_PUMP':os.environ.get('PAPERLAB_ALL_PUMP','0')})
app=modal.App('fly-paper-checkpoint-eval')
ALL_OBSERVED=os.environ.get('PAPERLAB_ALL_PUMP')=='1'

@app.function(image=image,volumes={'/state':volume},cpu=(2,2),memory=(8192,8192),timeout=1200,
              max_containers=1,min_containers=0,retries=0,single_use_containers=True,nonpreemptible=True)
def worker(batch,policy):
    return exclusive('worker',_evaluate,batch,policy)

def _evaluate(batch,policy):
    import time,re
    from paperlab.checkpoint_eval import read,evaluate,source_fingerprint
    from paperlab.core import atomic_json,digest
    from paperlab.budget import reserve,settle
    if not re.fullmatch(r'evaluation-[0-9]+',batch):raise ValueError('Invalid batch')
    volume.reload();state=Path('/state');base=state/'checkpoint-eval'/batch;plan=read(base/'plan.json')
    if policy not in plan['policies']:raise ValueError('Unsealed policy')
    if source_fingerprint('/opt/paperlab')!=plan['source_hashes']:
        raise ValueError('Evaluation source changed after seal')
    output=base/policy
    if output.exists():raise ValueError('Never overwrite evaluation attempt')
    reservation=reserve(state/'budget.json',True,seconds=3600,startup_seconds=30,memory_gib=8,limit_override=85,authorized_monthly_limit=100)
    if reservation is None:return dict(status='budget_stopped')
    reservation['rate']*=3;reservation['startup_seconds']=10
    output.mkdir();atomic_json(output/'owner.json',dict(call_id=modal.current_function_call_id(),reservation=reservation));volume.commit()
    try:
        result=evaluate(plan,policy,output,'/state/fly-data',commit=volume.commit)
        result['budget']=settle(state/'budget.json',reservation,time.time()-reservation['started'])
        atomic_json(output/'completed.json',result);volume.commit();return result
    except BaseException as exc:
        atomic_json(output/'failed.json',dict(error_type=type(exc).__name__,message=str(exc)[:300],reservation_retained=True));volume.commit();raise

@app.function(image=image,volumes={'/state':volume},cpu=(.125,.125),memory=(512,512),timeout=120,
              max_containers=1,min_containers=0,retries=0,single_use_containers=True,
              schedule=modal.Cron('7,22,37,52 * * * *'))
def coordinator():
    return exclusive('checkpoint-eval-coordinator',_coordinate)

def _coordinate():
    import time
    from paperlab.checkpoint_eval import read,register,seal_plan,seal_tape,source_fingerprint
    from paperlab.core import atomic_json,digest
    volume.reload();state=Path('/state');root=state/'checkpoint-eval';root.mkdir(exist_ok=True)
    control_path=root/'control.json';c=read(control_path) if control_path.exists() else dict(enabled=True,next_batch_at=0,batch=None,pending=None)
    now=time.time()
    def save(status):
        c.update(status=status,checked_at=now);atomic_json(control_path,c);volume.commit();return c
    if not c['enabled']:return save('disabled_requires_review')
    checkpoints=register(state);c['retained_checkpoints']=len(checkpoints)
    live=read(state/'solana-online/control.json')
    if not live['enabled'] or 'paused' in live.get('status','') or 'stopped' in live.get('status',''):return save('respecting_training_pause')
    if c['pending']:
        try:result=modal.FunctionCall.from_id(c['pending']['call_id']).get(timeout=0)
        except TimeoutError:return save('evaluation_running')
        except Exception as exc:c['enabled']=False;c['error']=type(exc).__name__;return save('evaluation_failed')
        if result.get('status')=='writer_busy':c['pending']=None;return save('worker_busy_wait')
        if result.get('status')=='budget_stopped':c['pending']=None;c['enabled']=False;return save('budget_paused_requires_review')
        if result.get('status')!='completed':c['enabled']=False;return save('evaluation_failed')
        c['pending']=None
    if c['batch'] is None:
        if now<c['next_batch_at']:return save('waiting_for_daily_cutoff')
        if not checkpoints:return save('waiting_for_checkpoints')
        plan=seal_plan(state,checkpoints,now,all_observed=ALL_OBSERVED)
        plan['source_hashes']=source_fingerprint('/opt/paperlab')
        base=root/plan['id'];base.mkdir();atomic_json(base/'plan.json',plan);c['batch']=plan['id']
        c['next_batch_at']=(int(now)//86400+1)*86400
        save('waiting_for_unseen_market_window')
    base=root/c['batch'];plan=read(base/'plan.json')
    if plan['tape'] is None:
        tape=seal_tape(state,plan)
        if tape is None:return save('waiting_for_unseen_market_window')
        plan['tape']=tape;atomic_json(base/'plan.json',plan);save('tape_sealed')
    todo=[p for p in plan['policies'] if not (base/p/'completed.json').exists()]
    if not todo:
        results=[read(base/p/'completed.json') for p in plan['policies']]
        atomic_json(base/'comparison.json',dict(results=results,scope=plan['scope'],tape=plan['tape'],profitable_learning_proven=False))
        c['last_completed_batch']=c['batch'];c['batch']=None;return save('comparison_completed')
    # Never consume the live trainer's reserved launch interval.
    if live.get('pending') or live.get('next_at',0)-now<1250:return save('waiting_for_training_gap')
    if (base/todo[0]).exists():c['enabled']=False;return save('ambiguous_attempt_requires_review')
    c['pending']=dict(policy=todo[0],call_id=None);save('dispatch_intent')
    call=worker.spawn(c['batch'],todo[0]);c['pending']['call_id']=call.object_id
    return save('evaluation_dispatched')
