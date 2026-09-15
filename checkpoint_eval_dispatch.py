"""Scheduling only: never imported by, or substituted for, the sealed evaluator."""
from pathlib import Path
import math
import time
import json
import hashlib


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    partial = path.with_suffix(path.suffix + ".dispatch-partial")
    partial.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    partial.replace(path)

PROTOCOL = 'priority_frozen_dispatch_v1'
# The unchanged worker has a 1200-second timeout. Preserve startup/headroom.
REQUIRED_GAP_SECONDS = 1250


def priority_order(plan):
    """Newest pre-cutoff checkpoint, controls, then remaining newest first.

    Membership, cutoff and outcomes never change. Ranking never uses returns.
    """
    policies = plan['policies']
    checkpoints = plan['checkpoints']
    ids = [c['id'] for c in checkpoints]
    if (len(set(policies)) != len(policies) or len(set(ids)) != len(ids)
            or set(policies) != set(ids) | {'untrained', 'cash', 'always_long'}
            or set(ids) & {'untrained', 'cash', 'always_long'}):
        raise ValueError('Invalid sealed policy membership')
    if any(not math.isfinite(c['ended']) or c['ended'] >= plan['cutoff'] for c in checkpoints):
        raise ValueError('Checkpoint does not precede sealed cutoff')
    newest = [c['id'] for c in sorted(checkpoints, key=lambda c: (c['ended'], c['id']), reverse=True)]
    if not newest:
        raise ValueError('No sealed trained checkpoint')
    return newest[:1] + ['untrained', 'cash', 'always_long'] + newest[1:]


def paused(control, now):
    return (not control.get('enabled') or control.get('audit_pause') or control.get('error')
            or control.get('budget_paused_until', 0) > now
            or any(s in control.get('status', '') for s in ('paused', 'stopped', 'requires_review')))


def recover_result(root, expected_call_id, *, poll, commit, clock=time.time):
    """Reconcile only the verified result of a missing-file pause, under lease."""
    root=Path(root);path=root/'checkpoint-eval/control.json';c=read(path)
    pending=c.get('pending') or {}
    if (c.get('enabled') or c.get('status')!='evaluation_failed'
            or c.get('error')!='MissingDurableEvaluationResult' or c.get('audit_pause')
            or c.get('budget_paused_until',0)>clock() or pending.get('call_id')!=expected_call_id):
        raise ValueError('Not the expected missing-result pause')
    result=poll(expected_call_id)  # Caller refreshes the mounted volume after completion.
    if read(path)!=c:raise ValueError('Evaluation control changed during verification')
    base=root/'checkpoint-eval'/c['batch'];policy=pending['policy']
    plan=read(base/'plan.json');output=base/policy
    if (not result or result.get('status')!='completed' or result.get('policy')!=policy
            or policy not in plan['policies'] or result.get('tape')!=plan['tape']['id']
            or result.get('weights_unchanged') is not True
            or read(output/'owner.json').get('call_id')!=expected_call_id
            or read(output/'completed.json')!=result):
        raise ValueError('Completed call and durable result must match exactly')
    record=base/'reconciliation-recovery'/f'{expected_call_id}.json'
    if record.exists():raise ValueError('Recovery already recorded; inspect current state')
    record.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(record,dict(at=clock(),before=c,result_sha256=digest(output/'completed.json'),
        action='Resume after exact completed call and persisted result match'))
    c.update(enabled=True,pending=None,status='verified_result_reconciled',checked_at=clock())
    c.pop('error')
    atomic_json(path,c);commit()
    return dict(status=c['status'],batch=c['batch'],policy=policy)


def dispatch(root, *, poll, spawn, commit, refresh_training, worker_busy, clock=time.time):
    """Caller holds the same coordinator lease as the original coordinator.

    Durable dispatch intent remains fail-closed after uncertain spawn outcomes.
    The original coordinator continues registry/new-batch work on its old schedule.
    """
    root = Path(root)
    path = root / 'checkpoint-eval/control.json'
    c = read(path)
    now = clock()
    if paused(c, now):
        return dict(status='respecting_evaluation_pause')

    def save(status):
        c.update(status=status, checked_at=clock(), dispatcher=PROTOCOL)
        atomic_json(path, c)
        commit()
        return c

    pending = c.get('pending')
    if pending:
        if not pending.get('call_id'):
            c['enabled'] = False
            return save('ambiguous_dispatch_requires_review')
        try:
            result = poll(pending['call_id'])
        except Exception as exc:
            c.update(enabled=False, error=type(exc).__name__)
            return save('evaluation_failed')
        if result is None:
            return save('evaluation_running')
        if result.get('status') == 'writer_busy':
            c['pending'] = None
            return save('worker_busy_wait')
        if result.get('status') == 'budget_stopped':
            c.update(pending=None, enabled=False)
            return save('budget_paused_requires_review')
        if result.get('status') != 'completed' or result.get('policy') != pending['policy']:
            c.update(enabled=False, error='UnexpectedEvaluationOutcome')
            return save('evaluation_failed')
        result_path = root / 'checkpoint-eval' / c['batch'] / pending['policy'] / 'completed.json'
        if not result_path.exists() or read(result_path).get('status') != 'completed':
            c.update(enabled=False, error='MissingDurableEvaluationResult')
            return save('evaluation_failed')
        c['pending'] = None
        save('evaluation_reconciled')

    if not c.get('batch'):
        return save('waiting_for_original_coordinator')
    base = root / 'checkpoint-eval' / c['batch']
    plan_path = base / 'plan.json'
    plan = read(plan_path)
    if not plan.get('tape'):
        return save('waiting_for_unseen_market_window')
    order = priority_order(plan)
    record = dict(protocol=PROTOCOL, plan_sha256=digest(plan_path), policies=order,
                  selection='Newest sealed pre-cutoff checkpoint first; never rank by returns')
    manifest = base / 'dispatch-policy-v1.json'
    if manifest.exists():
        if read(manifest) != record:
            c.update(enabled=False, error='DispatchManifestMismatch')
            return save('dispatch_manifest_requires_review')
    else:
        atomic_json(manifest, record)
        commit()
    todo = [p for p in order if not (base / p / 'completed.json').exists()]
    if not todo:
        # Leave final comparison aggregation to the unchanged coordinator.
        return save('awaiting_comparison_aggregation')

    live = read(root / 'solana-online/control.json')
    if paused(live, clock()):
        return save('respecting_training_pause')
    if live.get('pending'):
        if not live['pending'].get('call_id'):
            return save('respecting_training_dispatch_intent')
        try:
            outcome = poll(live['pending']['call_id'])
        except Exception:
            # Only the training coordinator owns failure/account/cadence transitions.
            outcome = dict(status='failed')
        if outcome is None:
            return save('waiting_for_training_gap')
        live = refresh_training()
    elif live.get('next_at', 0) <= clock():
        # A due training window retains priority, including a missed cron boundary.
        live = refresh_training()
    if paused(live, clock()):
        return save('respecting_training_pause')
    if live.get('pending') or live.get('next_at', 0) - clock() < REQUIRED_GAP_SECONDS:
        return save('waiting_for_training_gap')
    if worker_busy():
        return save('worker_busy_wait')
    policy = todo[0]
    if (base / policy).exists():
        c['enabled'] = False
        return save('ambiguous_attempt_requires_review')
    c['pending'] = dict(policy=policy, call_id=None)
    c.setdefault('priority_dispatches', []).append(dict(at=clock(), policy=policy, protocol=PROTOCOL))
    save('dispatch_intent')
    # An exception leaves intent on disk. Do not retry an uncertain dispatch.
    call_id = spawn(c['batch'], policy)
    if not isinstance(call_id, str) or not call_id:
        raise ValueError('Missing spawned call ID')
    c['pending']['call_id'] = call_id
    return save('evaluation_dispatched')


# Separate serialized ledger: never race the shared worker budget file.
# $5 is carved out of the existing $15 collector/overhead allocation.
DISPATCH_MONTHLY_LIMIT = 5.
DISPATCH_RATE = 2 * (.0000131 * .125 + .00000222 * .5)


def reserve_dispatch(root, now):
    from datetime import datetime, timezone
    path = Path(root) / 'checkpoint-eval/dispatch-budget.json'
    state = read(path) if path.exists() else dict(months={})
    month = datetime.fromtimestamp(now, timezone.utc).strftime('%Y-%m')
    charge = (60 + 30) * DISPATCH_RATE
    if state['months'].get(month, 0) + charge > .75 * DISPATCH_MONTHLY_LIMIT:
        return None
    state['months'][month] = state['months'].get(month, 0) + charge
    state.update(monthly_limit_usd=DISPATCH_MONTHLY_LIMIT, allocation='Within existing $15 overhead', provider_bill=False)
    atomic_json(path, state)
    return dict(month=month, reserved=charge, started=now)


def settle_dispatch(root, reservation, now):
    path = Path(root) / 'checkpoint-eval/dispatch-budget.json'
    state = read(path)
    charge = (max(0., now-reservation['started']) + 30) * DISPATCH_RATE
    state['months'][reservation['month']] += charge-reservation['reserved']
    atomic_json(path, state)
    return dict(estimated_compute_usd=charge, monthly_estimated_usd=state['months'][reservation['month']],
                monthly_limit_usd=DISPATCH_MONTHLY_LIMIT, provider_bill=False)
