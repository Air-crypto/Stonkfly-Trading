"""Unscheduled budgeted 1x/4x replay and full-native cadence experiment."""
from pathlib import Path
import modal
from cloud import exclusive,volume,writers
from solana_cloud import image as base_image
ROOT=Path(__file__).resolve().parent if modal.is_local() else Path('/opt/paperlab')
image=base_image
for name in ('replay_readout.py','accelerated_online.py','acceleration_study.py','acceleration_study_cloud.py'):
    image=image.add_local_file(ROOT/name,'/opt/paperlab/'+name,copy=True)
app=modal.App('fly-paper-acceleration-study')

@app.function(image=image,volumes={'/state':volume},cpu=(2,2),memory=(8192,8192),timeout=540,
              max_containers=1,min_containers=0,retries=0,single_use_containers=True,nonpreemptible=True)
def worker(run_id: str):
    return exclusive('worker',_study,run_id)

def _study(run_id):
    import re,time,json
    from paperlab.core import atomic_json,digest
    from paperlab.budget import reserve,settle
    from acceleration_study import select_data,replay_comparison,native_benchmark,read
    if not re.fullmatch('acceleration-[0-9-]+',run_id):raise ValueError('Invalid study ID')
    volume.reload();state=Path('/state');out=state/'acceleration-studies'/run_id
    if out.exists():raise ValueError('Never overwrite a study')
    live=read(state/'solana-online/control.json')
    if not live['enabled'] or live.get('audit_pause') or live.get('error') or live.get('budget_paused_until',0)>time.time():
        return dict(status='respecting_training_pause')
    if live.get('pending') or live['next_at']-time.time()<590:return dict(status='waiting_for_training_gap')
    reservation=reserve(state/'budget.json',True,seconds=1620,startup_seconds=30,memory_gib=8,limit_override=85,authorized_monthly_limit=100)
    if reservation is None:return dict(status='budget_stopped')
    reservation['rate']*=3;reservation['startup_seconds']=10;out.mkdir(parents=True)
    atomic_json(out/'owner.json',dict(call_id=modal.current_function_call_id(),reservation=reservation));volume.commit()
    try:
        plan=select_data(state,time.time())
        # Same starting weights in every arm; precedes all training/development logs.
        parent='solana-online-20260914-152719'
        if read(state/'solana-live'/parent/'completed.json')['ended']>=plan['training'][0]['started']:
            raise ValueError('Initialization does not precede training')
        plan.update(parent=parent,head_checkpoint=dict(path=str(state/'solana-live'/parent/'head-final.pt'),sha256=digest(state/'solana-live'/parent/'head-final.pt')),
                    native_checkpoint=dict(path=str(state/'solana-live'/parent/'fly-final.npz'),sha256=digest(state/'solana-live'/parent/'fly-final.npz')),
                    source_hashes={n:digest('/opt/paperlab/'+n) for n in ('replay_readout.py','accelerated_online.py','acceleration_study.py','acceleration_study_cloud.py')},
                    ratios=[1,4],capacity=10000,batch_size=32,target_every=100,learning_rate=.0003,
                    synthetic_cadence_arms=[[1,5],[4,5],[4,2.5]],native_arm_seconds=120,
                    scope='Controlled training and synthetic throughput diagnostics; no promotion or profitability claim')
        atomic_json(out/'plan.json',plan);volume.commit()
        replay=replay_comparison(state,out,plan,volume.commit);atomic_json(out/'replay-results.json',replay);volume.commit()
        native=native_benchmark(state,out,plan,volume.commit)
        result=dict(status='completed',replay=replay,native=native,baseline_unchanged=True,paper_only=True,
                    budget=settle(state/'budget.json',reservation,time.time()-reservation['started']))
        atomic_json(out/'completed.json',result);volume.commit();return result
    except BaseException as exc:
        atomic_json(out/'failed.json',dict(error_type=type(exc).__name__,message=str(exc)[:500],reservation_retained=True));volume.commit();raise
