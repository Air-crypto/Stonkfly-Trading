"""Unscheduled, detached, bounded paper replay. Never resumes the live portfolio."""
from pathlib import Path
import modal
from cloud import exclusive, volume
from solana_cloud import image as base_image

ROOT = Path(__file__).parent
image = base_image.add_local_file(ROOT/'group_replay_cloud.py', '/opt/paperlab/group_replay_cloud.py', copy=True)
app = modal.App('fly-paper-group-replay')


@app.function(image=image, volumes={'/state': volume}, cpu=(2, 2), memory=(8192, 8192),
              max_containers=1, min_containers=0, scaledown_window=2, timeout=1200,
              single_use_containers=True, retries=0, nonpreemptible=True)
def worker(run_id: str):
    return exclusive('worker', _run, run_id)


def _run(run_id):
    import re
    import time
    from paperlab.budget import reserve, settle
    from paperlab.core import atomic_json, digest
    from paperlab.group_replay import experiment
    if not re.fullmatch(r'group-replay-[a-z0-9-]{1,50}', run_id):
        raise ValueError('Invalid immutable replay run id')
    volume.reload()
    state = Path('/state')
    output = state/'group-replay'/run_id
    if output.exists():
        raise ValueError('Preserve prior replay output')
    # 1200s at 3x nonpreemptible pricing, plus existing 2x estimation margin.
    reservation = reserve(state/'budget.json', True, seconds=3600, startup_seconds=30,
                          memory_gib=8, limit_override=85, authorized_monthly_limit=100)
    if reservation is None:
        return dict(status='budget_stopped')
    reservation['rate'] *= 3
    reservation['startup_seconds'] = 10
    output.mkdir(parents=True)
    atomic_json(output/'owner.json', dict(call_id=modal.current_function_call_id(),
        input_id=modal.current_input_id(), at=time.time(), reservation=reservation,
        source_sha256={name: digest('/opt/paperlab/'+name) for name in
            ('group_replay_cloud.py','paperlab/group_replay.py','paperlab/core.py',
             'paperlab/solana_events.py','paperlab/solana_paper.py','paperlab/fly.py')}))
    volume.commit()
    try:
        result = experiment(state, output, volume.commit)
        result['budget'] = settle(state/'budget.json', reservation, time.time()-reservation['started'])
        atomic_json(output/'completed.json', result)
        volume.commit()
        return result
    except BaseException as exc:
        atomic_json(output/'failed.json', dict(error_type=type(exc).__name__, message=str(exc)[:300],
                                              at=time.time(), reservation_retained=True))
        volume.commit()
        raise
