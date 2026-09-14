"""Modal scheduled paper service: light coordinator, bounded full-fly windows."""
import os
from pathlib import Path
import modal
from cloud import exclusive, volume
from solana_cloud import image as base_image

ROOT = Path(__file__).resolve().parent if modal.is_local() else Path('/opt/paperlab')
app = modal.App('fly-paper-solana-online')
image = base_image.add_local_file(ROOT/'solana_online_cloud.py', '/opt/paperlab/solana_online_cloud.py', copy=True).env(
    {'PAPERLAB_ALL_PUMP':os.environ.get('PAPERLAB_ALL_PUMP','0')})
MODE = os.environ.get('PAPERLAB_ONLINE_MODE', 'paced')
ENABLED = os.environ.get('PAPERLAB_ONLINE_SCHEDULE') == '1'
ALL_OBSERVED = os.environ.get('PAPERLAB_ALL_PUMP') == '1'


@app.function(image=image, volumes={'/state':volume}, cpu=(2,2), memory=(8192,8192),
              timeout=1200, max_containers=1, min_containers=0, retries=0,
              single_use_containers=True, nonpreemptible=True)
def worker(run_id: str, parent: str, account_mode: str='continuous'):
    return exclusive('worker', _run, run_id, parent, account_mode)


def _run(run_id, parent, account_mode='continuous'):
    import json, re, time
    from paperlab.core import atomic_json, digest
    from paperlab.budget import reserve, settle
    from paperlab.solana_online import run, ACCOUNT_MODES
    if account_mode not in ACCOUNT_MODES:raise ValueError('Unknown account mode')
    if not re.fullmatch(r'solana-online-[0-9-]+',run_id) or not re.fullmatch(r'solana-[a-z0-9-]{1,60}',parent):
        raise ValueError('Invalid immutable run or parent id')
    volume.reload(); state=Path('/state'); root=state/'solana-live'/run_id
    if root.exists(): raise ValueError('Never restart an existing window')
    # $85 shared worker allocation + $15 for collector/overhead; retain 25% margin.
    reservation=reserve(state/'budget.json',True,seconds=3600,startup_seconds=30,memory_gib=8,
                        limit_override=85,authorized_monthly_limit=100)
    if reservation is None:return dict(status='budget_stopped')
    reservation['rate']*=3;reservation['startup_seconds']=10
    root.mkdir(parents=True)
    atomic_json(root/'owner.json',dict(call_id=modal.current_function_call_id(),parent=parent,
        account_mode=account_mode,
        run_id=run_id,at=time.time(),reservation=reservation,paper_only=True,source_sha256={
            n:digest('/opt/paperlab/'+n) for n in ('solana_online_cloud.py','paperlab/solana_online.py',
                                                'paperlab/solana_events.py','paperlab/solana_service.py',
                                                'paperlab/solana_paper.py','paperlab/solana_online_audit.py','paperlab/solana_quotes.py',
                                                'paperlab/solana_execution_audit.py','paperlab/solana_universe.py',
                                                'paperlab/solana_schema/pump_pool_account.json','paperlab/budget.py')}))
    volume.commit()
    try:
        result=run(root,'/state/fly-data',state/'solana-live'/parent,seconds=900,commit=volume.commit,
                   account_mode=account_mode,all_observed=ALL_OBSERVED)
        from paperlab.solana_online_audit import audit
        result['account_audit']=audit(json.loads((root/'opening.json').read_text()),
            [json.loads(line) for line in (root/'decisions.jsonl').read_text().splitlines()])
        result['training_health']=('learning_from_paper_execution' if result['fills'] and result['nonzero_reward_updates']
            else 'updates_without_execution_rewards' if result['new_readout_updates'] else 'no_learning_updates')
        from decimal import Decimal
        if Decimal(result['entry_budget_usd'])<Decimal('1.05') and not result['tradable_positions']:
            result['training_health']='risk_capacity_exhausted'
        result['budget']=settle(state/'budget.json',reservation,time.time()-reservation['started'])
        atomic_json(root/'completed.json',result);volume.commit();return result
    except BaseException as exc:
        atomic_json(root/'failed.json',dict(at=time.time(),error_type=type(exc).__name__,
            message=str(exc)[:300],reservation_retained=True));volume.commit();raise


@app.function(image=image, volumes={'/state':volume}, cpu=(.125,.125), memory=(512,512),
              timeout=60, max_containers=1, min_containers=0, retries=0, single_use_containers=True,
              # Avoid simultaneous starts with the legacy worker's */5 schedule.
              schedule=modal.Cron('2,17,32,47 * * * *') if ENABLED else None)
def coordinator():
    return exclusive('solana-online-coordinator', _coordinate)


def _coordinate():
    import json
    from paperlab.solana_service import service
    volume.reload()
    def poll(call_id):
        try:return modal.FunctionCall.from_id(call_id).get(timeout=0)
        except TimeoutError:return None
        except Exception as exc:return dict(status='failed',error_type=type(exc).__name__)
    result=service('/state',dispatch=lambda run_id,parent,account_mode='continuous':worker.spawn(run_id,parent,account_mode).object_id,
                   poll=poll,commit=volume.commit,mode=MODE)
    print(json.dumps(result),flush=True);return result


@app.function(image=image, volumes={'/state':volume}, cpu=(.125,.125), memory=(512,512),
              timeout=60, max_containers=1, min_containers=0, retries=0, single_use_containers=True)
def set_cadence(mode: str, start_now: bool=False):
    """Explicit operator action, serialized with scheduled dispatch."""
    return exclusive('solana-online-coordinator', _set_cadence, mode, start_now)


def _set_cadence(mode, start_now):
    from paperlab.solana_service import configure_cadence
    volume.reload()
    configure_cadence('/state', mode, commit=volume.commit, start_now=start_now)
    return _coordinate()


@app.function(image=image, volumes={'/state':volume}, cpu=(.125,.125), memory=(512,512),
              timeout=60, max_containers=1, min_containers=0, retries=0, single_use_containers=True)
def enable_episodes(expected_parent: str):
    """Explicit authorized account-mode transition; leaves old ledgers/checkpoints intact."""
    return exclusive('solana-online-coordinator', _enable_episodes, expected_parent)


def _enable_episodes(expected_parent):
    from paperlab.solana_service import enable_training_episodes
    volume.reload()
    enable_training_episodes('/state', expected_parent, commit=volume.commit)
    return _coordinate()
