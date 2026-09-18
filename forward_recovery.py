"""Audit and retain an interrupted frozen paper segment without replaying trades."""
import json
from pathlib import Path
import sqlite3
import time

from paperlab.core import atomic_json, digest
from paperlab.solana_universe import AllObservedFeed
from frozen_forward import PROTOCOL, audit_fills, validate_state


def recover_segment(root, manifest, *, expected_sha=None, now=None):
    root=Path(root);now=time.time() if now is None else now
    for key in ('native','head'):
        if digest(manifest[key]['path'])!=manifest[key]['sha256']:
            raise ValueError('Frozen checkpoint file changed')
    source_sha=digest(root/'decisions.jsonl')
    if expected_sha is not None and expected_sha!=source_sha:
        raise ValueError('Partial ledger differs from the reviewed ledger')
    report_path=root/'recovery.json'
    if report_path.exists():
        report=json.loads(report_path.read_text())
        if report['source_hashes']['decisions.jsonl']!=source_sha or digest(root/'recovered-state.json')!=report['continuation_sha256']:
            raise ValueError('Prior recovery evidence changed')
        return report
    if (root/'completed.json').exists():raise ValueError('Never relabel a completed session')
    started=json.loads((root/'started.json').read_text())
    opening=json.loads((root/'opening.json').read_text())
    validate_state(opening)
    if (started['checkpoint']!=manifest['checkpoint'] or started['protocol']!=PROTOCOL
            or not started['paper_only'] or started['native_learning'] or started['readout_learning'] or started['epsilon']!=0):
        raise ValueError('Not the selected frozen paper session')
    raw=(root/'decisions.jsonl').read_bytes();cut=raw.rfind(b'\n')+1
    rows=[json.loads(line) for line in raw[:cut].splitlines()]
    if not rows:raise ValueError('No complete durable observations to recover')
    latest=json.loads((root/'latest.json').read_text())
    if not any(r==latest for r in rows[-2:]):raise ValueError('Latest snapshot differs from the durable ledger')
    previous=started['at'];cursor=0;updates=set();previous_observations=opening['observations']
    for r in rows:
        if not r['paper_only'] or r['protocol']!=PROTOCOL or r['at']<previous or r['observation_at']>r['at']:
            raise ValueError('Invalid recovery chronology')
        if type(r['event_cursor']) is not int or r['event_cursor']<cursor:
            raise ValueError('Invalid recovery event cursor')
        if not previous_observations<=r['observations']<=previous_observations+1:
            raise ValueError('Native observation count changed unexpectedly')
        previous=r['at'];cursor=r['event_cursor'];previous_observations=r['observations']
        d=r.get('diagnostics')
        if d:
            if d['native_learning'] or d['readout_learning'] or d['native_weight_delta_l2']!=0 or d['epsilon']!=0:
                raise ValueError('Partial segment did not remain frozen')
            updates.add(d['inherited_backprop_updates'])
    if len(updates)>1:raise ValueError('Readout update count changed')
    audits={name:audit_fills(opening,rows,name) for name in opening['portfolios']}
    last=rows[-1]
    for name,a in audits.items():
        if last['fills'][name]!=opening['fills'][name]+a['fills']:
            raise ValueError('Cumulative fill count does not reconstruct')
    feed=AllObservedFeed(root)
    for c in opening['watched'].values():feed.accept(c)
    for c in opening['pools']:feed.accept(c)
    # Metadata must have been observed by the final retained decision, not later.
    db=sqlite3.connect('file:'+str(root/'events.db')+'?mode=ro',uri=True)
    events=0
    try:
        if db.execute('select coalesce(max(rowid),0) from events').fetchone()[0]<cursor:
            raise ValueError('Raw tape is shorter than the decision cursor')
        for event_id,body in db.execute('select rowid,body from events where rowid<=? order by rowid',(cursor,)):
            body=json.loads(body)
            if body['received']>last['observation_at']:raise ValueError('Future event in recovered observation')
            feed.accept(body,event_cursor=event_id);events+=1
    finally:db.close()
    snapshots=feed.snapshot()
    held={m for p in last['portfolios'].values() for m,v in p['positions'].items() if float(v['qty'])>0}
    if not held<=snapshots.keys():raise ValueError('Missing metadata for retained inventory')
    state=dict(portfolios=last['portfolios'],watched={m:t['created'] for m,t in snapshots.items()},
        pools=list(feed.pool_info.values()),contexts={},visits={},steps=opening['steps']+len(rows),
        observations=last['observations'],fills=last['fills'],last_at=last['at'],
        interruption=dict(session=root.name,last_durable_at=last['at'],recovered_at=now,
            pending_orders='expired; no fills across unobserved gap',
            features='price history and neural transient state warm up again after interruption'))
    validate_state(state);atomic_json(root/'recovered-state.json',state)
    report=dict(status='interrupted_recovered',partial_session=True,paper_only=True,
        checkpoint=manifest['checkpoint'],protocol=PROTOCOL,started=started['at'],ended=last['at'],recovered_at=now,
        observations=last['observations']-opening['observations'],cumulative_observations=last['observations'],
        fills=last['fills'],metrics=last['metrics'],account_audits=audits,
        checkpoint_hashes_verified=True,observed_native_deltas_zero=True,
        weights_unchanged=None,weight_verification='Checkpoint files and observed diagnostics verified; interrupted process has no final in-memory weight comparison',
        new_backprop_updates=0,inherited_backprop_updates=next(iter(updates),None),
        raw_events_used=events,event_cursor=cursor,incomplete_trailing_bytes=len(raw)-cut,
        continuation_sha256=digest(root/'recovered-state.json'),
        source_hashes={n:digest(root/n) for n in ('started.json','opening.json','decisions.jsonl','latest.json','events.db')},
        profitable_learning_proven=False)
    atomic_json(report_path,report);return report


def resume_control(root, expected_session, expected_call_id, expected_sha, *, now=None, commit=lambda:None):
    """Operator recovery after the exact pending cloud call has ended and both leases are held."""
    from forward_service import COORDINATOR_RATE, WORKER_RESERVATION
    root=Path(root);now=time.time() if now is None else now
    receipt=root/f'operator-recovery-{expected_session}.json'
    if receipt.exists():
        old=json.loads(receipt.read_text())
        if old['call_id']!=expected_call_id or old['ledger_sha256']!=expected_sha:raise ValueError('Different recovery request')
        return old
    c=json.loads((root/'control.json').read_text());manifest=json.loads((root/'manifest.json').read_text())
    pending=c.get('pending') or {}
    if c['enabled'] or c['status']!='paused_worker_failure' or pending.get('session')!=expected_session or pending.get('call_id')!=expected_call_id:
        raise ValueError('Not the expected paused session')
    if c['ends_at']!=manifest['ends_at'] or now+240>=c['ends_at']:
        raise ValueError('Original experiment deadline has ended')
    # Charge a conservative bound for the old unmetered paused wakeups, plus recovery/probe overhead.
    overhead=max(0,now-c['checked_at']+120)*COORDINATOR_RATE+.10
    if c['spent_usd']+overhead+WORKER_RESERVATION+.02>c['allowance_usd']:
        raise ValueError('No prepaid allowance remains for recovery')
    segment=root/'sessions'/expected_session
    opening=json.loads((segment/'opening.json').read_text())
    if json.loads((root/'state.json').read_text())!=opening:
        raise ValueError('Account moved after the interrupted session began')
    report=recover_segment(segment,manifest,expected_sha=expected_sha,now=now)
    backup=root/'recovery-archives'/expected_session;backup.mkdir(parents=True,exist_ok=True)
    if not (backup/'control-before.json').exists():
        atomic_json(backup/'control-before.json',c);atomic_json(backup/'state-before.json',opening)
    state=json.loads((segment/'recovered-state.json').read_text())
    receipt_value=dict(session=expected_session,call_id=expected_call_id,ledger_sha256=expected_sha,
        recovered_at=now,continuation_sha256=report['continuation_sha256'],paper_only=True,
        overhead_reserved_usd=overhead,deadline_unchanged=c['ends_at'],allowance_unchanged=c['allowance_usd'],
        retained_fills=state['fills'],retained_cash={k:p['cash'] for k,p in state['portfolios'].items()})
    c.update(enabled=True,status='ready_after_interruption',pending=None,next_session_seconds=180,
        spent_usd=c['spent_usd']+overhead,interrupted_sessions=c.get('interrupted_sessions',0)+1,
        last_result=report,checked_at=now)
    c.pop('error',None)
    c.setdefault('recoveries',[]).append(receipt_value)
    atomic_json(root/'state.json',state);atomic_json(root/'control.json',c)
    atomic_json(receipt,receipt_value);commit();return receipt_value
