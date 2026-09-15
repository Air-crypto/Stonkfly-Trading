"""Bounded evaluation preparation, separate from the immutable policy evaluator."""
from pathlib import Path
import time
from paperlab.core import atomic_json
from paperlab.checkpoint_eval import read, register, seal_plan, seal_tape, source_fingerprint
from paperlab.budget import reserve, settle


def request_prospective(root, source_root, *, expected_last_batch, commit, clock=time.time):
    """Advance one daily cutoff under both coordinator and worker leases."""
    root=Path(root);path=root/'checkpoint-eval/control.json';c=read(path);now=clock()
    live=read(root/'solana-online/control.json')
    for control in (c,live):
        if (not control.get('enabled') or control.get('error') or control.get('audit_pause')
                or control.get('budget_paused_until',0)>now
                or any(s in control.get('status','') for s in ('paused','stopped','requires_review'))):
            raise ValueError('Prospective request cannot override a pause')
    if c.get('last_completed_batch')!=expected_last_batch:
        raise ValueError('Completed evaluation changed; inspect before requesting')
    if c.get('batch') or c.get('pending'):
        return dict(status='existing_evaluation_preserved',batch=c.get('batch'))
    request=c.get('prospective_request')
    if not request or request.get('after_batch')!=expected_last_batch:
        c['prospective_request']=dict(after_batch=expected_last_batch,requested_at=now,
            previous_next_batch_at=c['next_batch_at'],selection='Latest completed checkpoint before cutoff; never select by returns')
        c['next_batch_at']=min(c['next_batch_at'],now)
        atomic_json(path,c);commit()
    return prepare(root,source_root,commit=commit,clock=clock)


def prepare(root, source_root, *, commit, all_observed=True, clock=time.time):
    root=Path(root);base=root/'checkpoint-eval';path=base/'control.json';c=read(path);now=clock()
    def save(status):
        c.update(status=status,checked_at=clock(),preparation_protocol='budgeted_preparation_v2')
        atomic_json(path,c);commit();return c
    if not c.get('enabled') or c.get('error') or c.get('audit_pause'):return dict(status='respecting_evaluation_pause')
    live=read(root/'solana-online/control.json')
    if not live.get('enabled') or live.get('audit_pause') or live.get('error') or live.get('budget_paused_until',0)>now:
        return dict(status='respecting_training_pause')
    if c.get('pending'):return dict(status='evaluation_pending')
    plan=read(base/c['batch']/'plan.json') if c.get('batch') else None
    if plan and plan.get('tape'):
        todo=[p for p in plan['policies'] if not (base/c['batch']/p/'completed.json').exists()]
        if todo:return dict(status='ready_for_dispatch',remaining=len(todo))
        results=[read(base/c['batch']/p/'completed.json') for p in plan['policies']]
        atomic_json(base/c['batch']/'comparison.json',dict(results=results,scope=plan['scope'],tape=plan['tape'],profitable_learning_proven=False))
        c.update(last_completed_batch=c['batch'],batch=None)
        return save('comparison_completed')
    if not plan and now<c['next_batch_at']:return save('waiting_for_daily_cutoff')
    reservation=reserve(root/'budget.json',True,seconds=1800,startup_seconds=30,memory_gib=8,
                        limit_override=85,authorized_monthly_limit=100)
    if reservation is None:return save('preparation_budget_stopped')
    reservation['rate']*=3;reservation['startup_seconds']=10
    progress=dict(started=now,batch=c.get('batch'),reservation=reservation)
    def stage(name):
        progress.update(stage=name,at=clock());atomic_json(base/'preparation-latest.json',progress);commit()
    try:
        stage('register_checkpoints')
        checkpoints=register(root);c['retained_checkpoints']=len(checkpoints)
        if plan is None:
            if not checkpoints:return save('waiting_for_checkpoints')
            plan=seal_plan(root,checkpoints,now,all_observed=all_observed)
            primary=max(plan['checkpoints'],key=lambda cp:(cp['ended'],cp['id']))
            plan.update(primary_policy=primary['id'],
                primary_selection='Latest completed checkpoint before cutoff; never select by returns',
                tape_selection='First completed training episode that starts strictly after cutoff')
            plan['source_hashes']=source_fingerprint(source_root)
            target=base/plan['id']
            if target.exists():raise ValueError('Never overwrite a sealed batch')
            target.mkdir();atomic_json(target/'plan.json',plan)
            c.update(batch=plan['id'],next_batch_at=(int(now)//86400+1)*86400)
            save('waiting_for_unseen_market_window')
        elif source_fingerprint(source_root)!=plan['source_hashes']:
            raise ValueError('Sealed evaluation source changed')
        stage('seal_unseen_tape')
        tape=seal_tape(root,plan)
        if tape is None:return save('waiting_for_unseen_market_window')
        plan['tape']=tape;atomic_json(base/c['batch']/'plan.json',plan)
        return save('tape_sealed')
    except Exception as exc:
        c.update(enabled=False,error=type(exc).__name__)
        progress.update(error_type=type(exc).__name__,message=str(exc)[:500])
        save('preparation_failed_requires_review');raise
    finally:
        progress.update(ended=clock(),stage='failed' if progress.get('error_type') else 'completed',
                        budget=settle(root/'budget.json',reservation,clock()-reservation['started']))
        atomic_json(base/'preparation-latest.json',progress);commit()
