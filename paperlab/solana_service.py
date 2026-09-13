"""Durable single-flight dispatch for budget-paced paper training windows."""
from datetime import datetime, timezone
from pathlib import Path
import json
import time

SEED = 'solana-live-14-next-next'


def next_delay(mode):
    delays = {'paced': 12*3600, 'six_hour': 6*3600, 'consecutive': 0}
    if mode not in delays: raise ValueError('Unknown cadence')
    return delays[mode]


def configure_cadence(root, mode, *, commit, now=None, start_now=False):
    """Caller must hold the coordinator lease; preserve all safety pauses/state."""
    from .core import atomic_json
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


def service(root, *, dispatch, poll, commit, now=None, mode='paced'):
    from .core import atomic_json
    now = time.time() if now is None else now
    next_delay(mode)
    root = Path(root); control_path = root/'solana-online/control.json'
    control = json.loads(control_path.read_text()) if control_path.exists() else dict(
        enabled=True, mode=mode, parent=SEED, pending=None, next_at=0., completed_windows=0,
        paper_only=True, scheduler_monthly_compute_upper_bound_usd=.99)
    def save(status):
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
        control['parent'] = pending['run_id']; control['pending'] = None
        control['completed_windows'] += 1
        control['last_result'] = {k:result.get(k) for k in
            ('ended','neural_observations','new_readout_updates','fills','equity_stress_usd','budget')}
        if result.get('portfolio', {}).get('halted'):
            control['enabled'] = False; return save('paper_loss_stop_requires_review')
        control['next_at'] = pending['dispatched_at'] + next_delay(control['mode'])
        save('completed')
    if now < control['next_at']: return save('waiting_for_budget_paced_window')
    # Retain all evidence, stop rather than silently delete trajectories or grow indefinitely.
    used = sum(p.stat().st_size for p in (root/'solana-live').rglob('*') if p.is_file())
    control['archive_bytes'] = used
    if used > 8*1024**3:
        control['enabled'] = False; return save('storage_cap_requires_review')
    run_id = 'solana-online-' + datetime.fromtimestamp(now, timezone.utc).strftime('%Y%m%d-%H%M%S')
    control['pending'] = dict(run_id=run_id, call_id=None, dispatched_at=now)
    save('dispatch_intent')  # A crash here cannot silently cause a duplicate submission.
    call_id = dispatch(run_id, control['parent'])
    control['pending']['call_id'] = call_id
    return save('dispatched')
