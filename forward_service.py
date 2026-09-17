"""Bounded, fail-closed scheduling and funding for the frozen paper account."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from paperlab.core import atomic_json, digest
from frozen_forward import initial_state, validate_state, PROTOCOL

CHECKPOINT = 'solana-online-20260916-005626'
WORKER_RATE = 2 * (.0000131 + 8 * .00000222)  # 1 CPU, 8 GiB, standard; 2x margin.
WORKER_RESERVATION = (1200 + 60) * WORKER_RATE
COORDINATOR_RATE = 2 * (.0000131 * .125 + .00000222 * .5)


def fund(shared, dest, allowance=17.5, now=None, commit=lambda:None):
    """Called under the existing shared worker lock before any independent spend."""
    now=time.time() if now is None else now
    shared=Path(shared);dest=Path(dest);manifest_path=dest/'manifest.json'
    if manifest_path.exists():return json.loads(manifest_path.read_text())
    if allowance != 17.5:raise ValueError('This protocol authorizes a fixed $17.50 envelope')
    month=datetime.fromtimestamp(now,timezone.utc).strftime('%Y-%m')
    budget=json.loads((shared/'budget.json').read_text())
    receipt=shared/'forward-reservations'/('frozen-'+CHECKPOINT+'.json')
    # Refuse ambiguous partial funding rather than charging twice or resuming with free cash.
    if receipt.exists():raise ValueError('Funding receipt exists without manifest; reconcile before retry')
    if budget['months'].get(month,0)+allowance>85*.75:raise ValueError('Shared monthly worker guard would be exceeded')
    parent=shared/'solana-live'/CHECKPOINT
    completed=json.loads((parent/'completed.json').read_text())
    state=json.loads((parent/'online-state.json').read_text())
    if completed['status']!='completed' or not completed['paper_only']:raise ValueError('Invalid checkpoint')
    import shutil
    dest.mkdir(parents=True,exist_ok=True);model=dest/'model';model.mkdir(exist_ok=True)
    manifest=dict(protocol=PROTOCOL,checkpoint=CHECKPOINT,started=now,ends_at=now+3*86400,
        days=3,paper_only=True,initial_cash_usd=1000,allowance_usd=allowance,
        frozen=True,native_learning=False,readout_learning=False,epsilon=0,
        max_token_acquisition_usd=250,liquidity_fraction=.01,news_enabled=False,
        worker_cpu=1,worker_memory_gib=8,nonpreemptible=False,
        selection='Selected after an earlier marked-return comparison; this new forward period is unseen',
        controls=['always_long','cash'],coverage='All received Pump/PumpSwap events, no guarantee of all tokens or gap-free data')
    for key,source_key,filename in [('native','native_checkpoint','fly.npz'),('head','head_checkpoint','head.pt')]:
        source=Path(state[source_key]);target=model/filename;shutil.copyfile(source,target)
        manifest[key]=dict(path=str(target),sha256=digest(target))
        if digest(source)!=manifest[key]['sha256']:raise ValueError('Checkpoint copy differs')
    # Persist the prepaid debit and receipt first. No refund is inferred from worker failure.
    atomic_json(receipt,dict(at=now,month=month,allowance_usd=allowance,checkpoint=CHECKPOINT,
                            prepaid=True,provider_invoice=False))
    budget['months'][month]=budget['months'].get(month,0)+allowance
    atomic_json(shared/'budget.json',budget);commit()
    atomic_json(manifest_path,manifest)
    atomic_json(dest/'state.json',initial_state())
    atomic_json(dest/'control.json',dict(enabled=True,status='ready',pending=None,sequence=0,
        completed_sessions=0,spent_usd=0.,allowance_usd=allowance,ends_at=manifest['ends_at'],
        checkpoint=CHECKPOINT,checked_at=now,paper_only=True,archive_limit_bytes=64*1024**3))
    return manifest


def coordinate(root, dispatch, poll, *, now=None, commit=lambda:None):
    now=time.time() if now is None else now;root=Path(root);path=root/'control.json'
    if not path.exists():return dict(status='not_started')
    c=json.loads(path.read_text())
    if not c['enabled']:return c
    c['checked_at']=now
    # Reserve the full 60-second small coordinator timeout before other work.
    c['spent_usd'] += 60*COORDINATOR_RATE
    atomic_json(path,c);commit()
    pending=c.get('pending')
    if pending:
        result=poll(pending['call_id'])
        if result is None:return c
        commit_path=root/'sessions'/pending['session']/'completed.json'
        if result.get('status')!='completed' or not commit_path.exists():
            c.update(enabled=False,status='paused_worker_failure',error=result,pending=pending)
        else:
            durable=json.loads(commit_path.read_text())
            if durable!=result:raise ValueError('Cloud result differs from durable session')
            if not result.get('weights_unchanged') or result.get('new_backprop_updates')!=0 or result.get('checkpoint')!=CHECKPOINT:
                raise ValueError('Result does not belong to the frozen experiment')
            state_path=commit_path.parent/'continuation.json'
            if digest(state_path)!=result['continuation_sha256']:raise ValueError('Account continuation hash differs')
            next_state=json.loads(state_path.read_text());validate_state(next_state)
            atomic_json(root/'state.json',next_state)
            # Startup is inside the prepaid envelope; crashes retain the full reservation.
            actual=min(WORKER_RESERVATION,(max(0,result['ended']-pending['reserved_at'])+60)*WORKER_RATE)
            c['spent_usd'] += actual-WORKER_RESERVATION
            c.update(pending=None,completed_sessions=c['completed_sessions']+1,last_result=result,status='ready')
    if c['enabled'] and not c['pending']:
        if now+60>=c['ends_at']:c.update(enabled=False,status='completed_three_day_test')
        elif c['spent_usd']+WORKER_RESERVATION+22*60*COORDINATOR_RATE>c['allowance_usd']:
            c.update(enabled=False,status='budget_stopped')
        else:
            size=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
            c['archive_bytes']=size
            if size>c['archive_limit_bytes']-2*1024**3:c.update(enabled=False,status='archive_stopped')
            else:
                c['sequence']+=1;session=f"session-{c['sequence']:05d}"
                c['spent_usd']+=WORKER_RESERVATION
                c['pending']=dict(session=session,call_id=None,reserved_at=now)
                c['status']='dispatching';atomic_json(path,c);commit()
                try:
                    c['pending']['call_id']=dispatch(session,min(180 if c['completed_sessions']==0 else 900,int(c['ends_at']-now)))
                    c['status']='running'
                except BaseException:
                    c.update(enabled=False,status='paused_dispatch_uncertain')
                    atomic_json(path,c);commit();raise
    atomic_json(path,c);commit();return c
