"""Versioned scheduling repair; sealed evaluator sources remain unchanged."""
# Baseline SHA256: 610cd8ecbaa13202a79f1cf4e0747d77e7362e0cfced3fa72392f38311d32cc2
"""Durable single-flight dispatch for budget-paced paper training windows."""
from datetime import datetime, timezone
from pathlib import Path
import json
import time

SEED = 'solana-live-14-next-next'
ARCHIVE_LIMIT_BYTES = 64*1024**3


def next_delay(mode):
    delays = {'paced': 12*3600, 'six_hour': 6*3600, 'hourly': 3600, 'consecutive': 0}
    if mode not in delays: raise ValueError('Unknown cadence')
    return delays[mode]


def configure_cadence(root, mode, *, commit, now=None, start_now=False):
    """Caller must hold the coordinator lease; preserve all safety pauses/state."""
    from paperlab.core import atomic_json
    delay = next_delay(mode)
    now = time.time() if now is None else now
    path = Path(root)/'solana-online/control.json'
    control = json.loads(path.read_text())
    previous_mode = control['mode']
    previous_next_at = control['next_at']
    if not control['enabled']:
        raise ValueError('Disabled service requires review before cadence changes')
    # A budget pause is sticky even after a scheduled tick replaces its status.
    budget_pause = control.get('budget_paused_until', 0)
    if control.get('status') == 'budget_paused_until_next_month':
        budget_pause = max(budget_pause, previous_next_at)
    if budget_pause > now:
        raise ValueError('Cadence change cannot bypass a budget pause')
    if not control.get('pending'):
        if control.get('status') != 'waiting_for_budget_paced_window':
            raise ValueError('Idle service must be reconciled before cadence changes')
        control['next_at'] = now if start_now else max(
            now, previous_next_at-next_delay(previous_mode)+delay)
    control['mode'] = mode
    control.setdefault('cadence_changes', []).append(dict(
        at=now, previous_mode=previous_mode, mode=mode,
        previous_next_at=previous_next_at, next_at=control['next_at'],
        start_now=start_now, pending_preserved=bool(control.get('pending'))))
    atomic_json(path, control); commit()
    return control


def enable_training_episodes(root, expected_parent, *, commit, now=None):
    """Explicit operator-authorized conversion of an exhausted paper account.

    The caller holds the coordinator lease. Budget, storage, pending calls and
    operational failures remain blocking; only account-risk exhaustion is reclassified.
    """
    from paperlab.core import atomic_json, digest
    now=time.time() if now is None else now
    root=Path(root);path=root/'solana-online/control.json'
    control=json.loads(path.read_text())
    if control.get('pending') or control['parent']!=expected_parent:
        raise ValueError('Parent changed or call pending; reconcile first')
    if control.get('account_mode')=='fresh_training_episode':
        raise ValueError('Training episodes already configured')
    if control.get('budget_paused_until',0)>now or control.get('status')=='budget_paused_until_next_month':
        raise ValueError('Cannot bypass a budget pause')
    used=sum(p.stat().st_size for p in (root/'solana-live').rglob('*') if p.is_file())
    if used>ARCHIVE_LIMIT_BYTES:
        raise ValueError('Cannot bypass storage cap')
    parent=root/'solana-live'/expected_parent
    result=json.loads((parent/'completed.json').read_text())
    if (result.get('status')!='completed' or not result.get('paper_only')
            or result.get('training_health')!='risk_capacity_exhausted'
            or control.get('last_result',{}).get('training_health')!='risk_capacity_exhausted'
            or control.get('status') not in ('disabled','training_risk_capacity_requires_review')):
        raise ValueError('Only the reviewed account-risk pause can enable fresh episodes')
    archive=root/'solana-online'/f'pre-episodes-{expected_parent}.json'
    if archive.exists():
        raise ValueError('Preserve original account-mode transition')
    atomic_json(archive,control)
    control.update(account_mode='fresh_training_episode',enabled=True,next_at=now,
        status='waiting_for_budget_paced_window',
        continuous_account_archive=dict(run_id=expected_parent,completed_sha256=digest(parent/'completed.json'),
                                        control_archive=str(archive)),
        episode_totals=dict(episodes=0,sum_episode_pnl_usd=0.,sum_episode_fees_usd=0.,
                            scope='Independent training episodes; not continuous portfolio performance'))
    control.setdefault('account_mode_changes',[]).append(dict(at=now,from_mode='continuous',
        to_mode='fresh_training_episode',parent=expected_parent,initial_cash_per_episode_usd=1000.,
        authorization='User requested resetting paper capital per training run; learned weights continue'))
    atomic_json(path,control);commit();return control


def service(root, *, dispatch, poll, commit, now=None, mode='paced', worker_busy=lambda:False):
    from paperlab.core import atomic_json
    now = time.time() if now is None else now
    next_delay(mode)
    root = Path(root); control_path = root/'solana-online/control.json'
    control = json.loads(control_path.read_text()) if control_path.exists() else dict(
        enabled=True, mode=mode, parent=SEED, pending=None, next_at=0., completed_windows=0,
        paper_only=True, scheduler_monthly_compute_upper_bound_usd=.99)
    def save(status):
        if not control.get('enabled'):
            reason = control.get('pause_reason') or (status if status!='disabled' else control.get('status','disabled'))
            control.setdefault('pause_reason', reason)
            control.setdefault('paused_at', now)
            status = reason
        control['archive_limit_bytes'] = ARCHIVE_LIMIT_BYTES
        control['scheduler_protocol'] = 'retained_archive_scheduler_v2'
        control['status'] = status; control['checked_at'] = now
        atomic_json(control_path, control); commit(); return control
    if not control['enabled']: return save('disabled')
    pending = control.get('pending')
    if pending:
        completed = root/'solana-live'/pending['run_id']/'completed.json'
        # Always confirm the call ended, including volume commits and lease release.
        if not pending.get('call_id'):
            control['enabled'] = False; return save('ambiguous_dispatch_requires_review')
        outcome = poll(pending['call_id'])
        if outcome is None: return save('running_or_queued')
        if outcome.get('status') == 'writer_busy':
            control['pending'] = None; return save('writer_busy_retry_next_schedule')
        if outcome.get('status') == 'budget_stopped':
            control['pending'] = None
            dt = datetime.fromtimestamp(now, timezone.utc)
            month = datetime(dt.year+int(dt.month==12), dt.month%12+1, 1, tzinfo=timezone.utc)
            control['next_at'] = month.timestamp()
            control['budget_paused_until'] = month.timestamp()
            return save('budget_paused_until_next_month')
        if outcome.get('status') != 'completed' or not completed.exists():
            control['enabled'] = False; return save('failed_requires_review')
        result = json.loads(completed.read_text())
        if result['status'] != 'completed' or not result.get('paper_only'):
            control['enabled'] = False; return save('invalid_completion')
        episodic=pending.get('account_mode','continuous')=='fresh_training_episode'
        if episodic:
            episode=result.get('episode') or {}
            pnl=result.get('episode_pnl_usd')
            import math
            if (result.get('account_mode')!='fresh_training_episode' or not episode.get('training_only')
                    or episode.get('episode_id')!=pending['run_id'] or episode.get('initial_cash_usd')!=1000
                    or not isinstance(pnl,(int,float)) or not math.isfinite(pnl)
                    or abs(pnl-(result['equity_stress_usd']-1000))>1e-7):
                control['enabled']=False;return save('invalid_episode_completion')
            summary=dict(run_id=pending['run_id'],initial_cash_usd=1000.,episode_pnl_usd=pnl,
                ended=result['ended'],end_equity_stress_usd=result['equity_stress_usd'],
                end_portfolio=result['portfolio'],training_only=True,
                training_health=result.get('training_health'),account_audit=result.get('account_audit'))
            atomic_json(root/'solana-online/episode-results'/f"{pending['run_id']}.json",summary)
            totals=control['episode_totals'];totals['episodes']+=1
            totals['sum_episode_pnl_usd']+=pnl
            totals['sum_episode_fees_usd']+=float(result['portfolio']['fees'])
        control['parent'] = pending['run_id']; control['pending'] = None
        control['completed_windows'] += 1
        control['last_result'] = {k:result.get(k) for k in
            ('ended','neural_observations','new_readout_updates','nonzero_reward_updates','fills',
             'equity_stress_usd','entry_budget_usd','training_health','account_audit','budget')}
        control['last_result'].update(account_mode=result.get('account_mode','continuous'),
                                      episode_pnl_usd=result.get('episode_pnl_usd'))
        if result.get('portfolio', {}).get('halted') and not episodic:
            control['enabled'] = False; return save('paper_loss_stop_requires_review')
        if result.get('training_health')=='risk_capacity_exhausted' and not episodic:
            control['enabled']=False; return save('training_risk_capacity_requires_review')
        control['next_at'] = pending['dispatched_at'] + next_delay(control['mode'])
        save('completed')
    if now < control['next_at']: return save('waiting_for_budget_paced_window')
    if worker_busy(): return save('waiting_for_shared_worker')
    # Retain all evidence, stop rather than silently delete trajectories or grow indefinitely.
    used = sum(p.stat().st_size for p in (root/'solana-live').rglob('*') if p.is_file())
    control['archive_bytes'] = used
    if used > ARCHIVE_LIMIT_BYTES:
        control['enabled'] = False; return save('storage_cap_requires_review')
    run_id = 'solana-online-' + datetime.fromtimestamp(now, timezone.utc).strftime('%Y%m%d-%H%M%S')
    account_mode=control.get('account_mode','continuous')
    if account_mode not in ('continuous','fresh_training_episode'):
        control['enabled']=False;return save('invalid_account_mode')
    control['pending'] = dict(run_id=run_id, call_id=None, dispatched_at=now,account_mode=account_mode)
    save('dispatch_intent')  # A crash here cannot silently cause a duplicate submission.
    call_id = (dispatch(run_id, control['parent'], account_mode) if account_mode=='fresh_training_episode'
               else dispatch(run_id, control['parent']))
    control['pending']['call_id'] = call_id
    return save('dispatched')


def recover_storage(root, expected_parent, *, commit, now=None):
    """Reviewed storage-only recovery; other pauses and account history stay intact."""
    from paperlab.core import atomic_json, digest
    now=time.time() if now is None else now
    root=Path(root);path=root/'solana-online/control.json'
    c=json.loads(path.read_text())
    if (c.get('enabled') or c.get('pending') or c['parent']!=expected_parent
        or c.get('error') or c.get('audit_pause') or c.get('budget_paused_until',0)>now
        or c.get('status') not in ('disabled','storage_cap_requires_review')
        or c.get('pause_reason') not in (None,'storage_cap_requires_review')
        or c.get('archive_bytes',0)<=8*1024**3):
        raise ValueError('Not the verified idle storage pause')
    used=sum(p.stat().st_size for p in (root/'solana-live').rglob('*') if p.is_file())
    if used>=ARCHIVE_LIMIT_BYTES:raise ValueError('Archive still exceeds bounded allowance')
    parent=json.loads((root/'solana-live'/expected_parent/'completed.json').read_text())
    if parent.get('status')!='completed' or not parent.get('paper_only'):raise ValueError('Invalid parent')
    archive=root/'solana-online'/('storage-recovery-'+expected_parent+'.json')
    if archive.exists():raise ValueError('Preserve prior recovery record')
    atomic_json(archive,dict(before=c,at=now,measured_archive_bytes=used,
        evidence='Original coordinator log: storage_cap_requires_review at 1789447632.6025057',
        previous_limit_bytes=8*1024**3,new_limit_bytes=ARCHIVE_LIMIT_BYTES,
        parent_sha256=digest(root/'solana-live'/expected_parent/'completed.json'),
        authorization='User requested diagnosis and repair; preserve evidence and $100 monthly authorization'))
    c.update(enabled=True,status='waiting_for_budget_paced_window',archive_bytes=used,
        archive_limit_bytes=ARCHIVE_LIMIT_BYTES,next_at=now+1800,
        scheduler_protocol='retained_archive_scheduler_v2')
    c.pop('pause_reason',None);c.pop('paused_at',None)
    c.setdefault('recovery_history',[]).append(dict(at=now,reason='storage_cap',archive=str(archive),
        next_at=c['next_at'],note='Bounded 30-minute recovery gap for evaluation preparation; resume hourly training after verification'))
    atomic_json(path,c);commit();return c
