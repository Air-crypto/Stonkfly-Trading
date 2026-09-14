"""Unscheduled synthetic native smoke; never a performance evaluation."""
from pathlib import Path
import modal
from cloud import exclusive,volume
from solana_cloud import image as base_image
ROOT=Path(__file__).resolve().parent if modal.is_local() else Path('/opt/paperlab')
image=base_image.add_local_file(ROOT/'checkpoint_eval_smoke_cloud.py','/opt/paperlab/checkpoint_eval_smoke_cloud.py',copy=True)
app=modal.App('fly-paper-checkpoint-smoke')

@app.function(image=image,volumes={'/state':volume},cpu=(2,2),memory=(8192,8192),timeout=600,
              max_containers=1,min_containers=0,retries=0,single_use_containers=True,nonpreemptible=True)
def worker(run_id: str):
    return exclusive('worker',_smoke,run_id)

def _smoke(run_id):
    import json,time,sqlite3,re
    from paperlab.core import atomic_json,digest
    from paperlab.budget import reserve,settle
    from paperlab.checkpoint_eval import register,evaluate,PROTOCOL
    from paperlab.solana_quotes import QUOTE_PROTOCOL
    from paperlab.solana_events import SOL,ZERO,TOKEN
    if not re.fullmatch(r'audit-smoke-[0-9-]+',run_id):raise ValueError('Invalid immutable smoke ID')
    volume.reload();state=Path('/state');out=state/'checkpoint-eval-audit'/run_id
    if out.exists():raise ValueError('Never overwrite smoke')
    reservation=reserve(state/'budget.json',True,seconds=1800,startup_seconds=30,memory_gib=8,limit_override=85,authorized_monthly_limit=100)
    if reservation is None:return dict(status='budget_stopped')
    reservation['rate']*=3;reservation['startup_seconds']=10
    out.mkdir(parents=True);atomic_json(out/'owner.json',dict(call_id=modal.current_function_call_id(),reservation=reservation,synthetic=True));volume.commit()
    try:
        checkpoints=register(state);cp=max(checkpoints,key=lambda c:c['ended']);t0=time.time()+100
        tape=out/'synthetic';tape.mkdir();db=sqlite3.connect(tape/'events.db');db.execute('create table events(received real,log_index integer,body text)')
        mint='CeSFzoAqSMXMgodeLzrTMe3V5MdV5Nhinehedxohpump'
        created=dict(kind='CreateEvent',mint=mint,quote_mint=ZERO,token_program=TOKEN,is_mayhem_mode=False,
                     received=t0-60,timestamp=int(t0-60),slot=1,signature='synthetic-create',log_index=0)
        db.execute('insert into events values(?,?,?)',(created['received'],0,json.dumps(created)))
        rows=[]
        for i in range(18):
            obs=t0+i*5;received=obs-.1
            trade=dict(kind='TradeEvent',mint=mint,quote_mint=ZERO,mayhem_mode=False,received=received,timestamp=int(received),slot=2+i,
                       signature='synthetic-'+str(i),log_index=i+1,virtual_sol_reserves=60_000_000_000+i*50_000_000,
                       virtual_token_reserves=500_000_000_000_000,real_sol_reserves=30_000_000_000,sol_amount=10_000_000,is_buy=i%2==0)
            db.execute('insert into events values(?,?,?)',(received,i+1,json.dumps(trade)))
            opportunity=i>=12
            rows.append(dict(at=obs+.5,observation_at=obs,event_cursor=i+2,fx_state=dict(sol_usd=100,seen_at=t0-1),quote_protocol=QUOTE_PROTOCOL,admissions=[mint] if i==0 else [],retired=[],
                feed=dict(status='connected',last_message=received),neural=dict(mint=mint,tick=dict(received_at=received)) if opportunity else None,
                decision=dict(mint=mint,issued=obs+.25) if opportunity else None))
        db.commit();db.close()
        (tape/'fx.jsonl').write_text(json.dumps(dict(at=t0-1,sol_usd=100))+'\n')
        (tape/'decisions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        plan=dict(protocol=PROTOCOL,quote_protocol=QUOTE_PROTOCOL,cutoff=t0-1,policies=['untrained',cp['id']],checkpoints=[cp],
            synthetic=True,scope='Synthetic native mechanics smoke only; no performance or future-market claim.',
            tape=dict(id='synthetic-native-smoke',started=t0,files={n:dict(path=str(tape/n),sha256=digest(tape/n)) for n in ['events.db','fx.jsonl','decisions.jsonl']}))
        atomic_json(out/'plan.json',plan);results=[]
        for policy in plan['policies']:
            r=evaluate(plan,policy,out/policy,'/state/fly-data',commit=volume.commit)
            if r['native_observations']!=6 or not r['weights_unchanged']:raise ValueError('Incomplete native smoke coverage')
            atomic_json(out/policy/'completed.json',r);results.append(r);volume.commit()
        result=dict(status='completed',synthetic=True,results=results,budget=settle(state/'budget.json',reservation,time.time()-reservation['started']))
        atomic_json(out/'completed.json',result);volume.commit();return result
    except BaseException as exc:
        atomic_json(out/'failed.json',dict(error_type=type(exc).__name__,message=str(exc)[:500],reservation_retained=True));volume.commit();raise
