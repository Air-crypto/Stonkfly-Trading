"""Unscheduled, budgeted public-data coverage probe; no trading or learning."""
from pathlib import Path
import modal
from cloud import exclusive,volume
from solana_cloud import image as base_image
ROOT=Path(__file__).resolve().parent if modal.is_local() else Path('/opt/paperlab')
image=base_image.add_local_file(ROOT/'universe_probe_cloud.py','/opt/paperlab/universe_probe_cloud.py',copy=True)
app=modal.App('fly-paper-universe-probe')

@app.function(image=image,volumes={'/state':volume},cpu=(1,1),memory=(2048,2048),timeout=180,
              max_containers=1,min_containers=0,retries=0,single_use_containers=True,nonpreemptible=True)
def worker(run_id: str):
    return exclusive('worker',_probe,run_id)

def _probe(run_id):
    import time,json,re,requests
    from collections import Counter
    from paperlab.solana_universe import AllObservedFeed,prepare_snapshots,USDC
    from paperlab.solana_quotes import quote_for
    from paperlab.budget import reserve,settle
    from paperlab.core import atomic_json
    if not re.fullmatch('universe-probe-[0-9-]+',run_id):raise ValueError('Invalid probe ID')
    volume.reload();root=Path('/state/universe-probes')/run_id
    if root.exists():raise ValueError('Never overwrite a probe')
    reservation=reserve('/state/budget.json',False,seconds=540,startup_seconds=30,cpu=1,memory_gib=2,
                        limit_override=85,authorized_monthly_limit=100)
    if reservation is None:return dict(status='budget_stopped')
    reservation['rate']*=3;reservation['startup_seconds']=10;root.mkdir(parents=True)
    atomic_json(root/'owner.json',dict(call_id=modal.current_function_call_id(),reservation=reservation));volume.commit()
    feed=AllObservedFeed(root);feed.start();start=time.time();samples=[]
    try:
        def price(symbol):
            r=requests.get('https://api.coinbase.com/v2/prices/'+symbol+'-USD/spot',timeout=4)
            r.raise_for_status();return float(r.json()['data']['amount'])
        sol=price('SOL');usdc=price('USDC');seen=time.time()
        while time.time()-start<80:
            snap,now,health=feed.snapshot_at();prepared=prepare_snapshots(snap,now,sol,seen,{USDC:usdc})
            quotes=[quote_for(t,now,sol,seen,unrestricted=True) for t in prepared.values()]
            s=dict(at=now,health=health,observed_tokens=len(snap),entry_eligible=sum(q.entry_tick is not None for q in quotes),
                reasons=dict(Counter(q.observation_reason or q.entry_reason or 'eligible' for q in quotes)))
            samples.append(s);atomic_json(root/'latest.json',s);print(json.dumps(s),flush=True)
            time.sleep(10)
        feed.close()
        result=dict(status='completed',paper_only=True,trading=False,all_tokens_guaranteed=False,
                    samples=samples,health=feed.health(),observed_tokens=len(feed.tokens),
                    budget=settle('/state/budget.json',reservation,time.time()-reservation['started']))
        atomic_json(root/'completed.json',result);volume.commit();return result
    except BaseException as exc:
        feed.close();atomic_json(root/'failed.json',dict(error_type=type(exc).__name__,reservation_retained=True));volume.commit();raise
