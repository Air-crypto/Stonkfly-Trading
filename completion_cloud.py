"""Temporary non-preemptible completion; preserve the old main/recovery apps."""
from pathlib import Path
import modal

from cloud import volume, exclusive
from recovery_cloud import image

ROOT = Path(__file__).parent
app = modal.App('fly-paper-completion-11')
image = (image.add_local_file(ROOT/'completion_cloud.py', '/opt/paperlab/completion_cloud.py', copy=True)
    .add_local_file(ROOT/'reports/fly-online-completion-protocol-11.json',
                    '/opt/paperlab/recovery/completion-protocol.json', copy=True)
    .add_local_file(ROOT/'reports/fly-online-recovery-preemption-11.json',
                    '/opt/paperlab/recovery/fly-online-recovery-preemption-11.json', copy=True))
for name in ('development-pool0-pristine_frozen', 'development-pool0-trained_frozen',
        'development-pool0-trained_online_carry', 'development-pool0-trained_online_reset_rates',
        'development-pool1-pristine_frozen', 'development-pool1-trained_frozen'):
    image = image.add_local_file(ROOT/'reports/online-completion-retained'/f'{name}.json',
        '/opt/paperlab/recovery/retained/'+name+'.json', copy=True)


def _verify_import():
    import importlib
    importlib.import_module('completion_cloud')
    print('Non-preemptible completion entrypoint import passed', flush=True)


image = image.run_function(_verify_import, cpu=.125, memory=256, timeout=60)


@app.function(image=image, volumes={'/state': volume}, cpu=.125, memory=512,
    timeout=60, retries=0, single_use_containers=True)
def check_reference():
    """Read-only cloud preflight; no claim, budget mutation or model construction."""
    import json
    from paperlab.fly_online_completion import reference
    volume.reload()
    p, parent, envelope, sources, news = reference('/state', '/opt/paperlab/registered',
        '/opt/paperlab/recovery', '/opt/paperlab')
    result = {'status': 'completion_reference_verified', 'plan_sha256': envelope['sha256'],
        'original_sources': len(sources), 'retained_conditions': len(p['retained']), 'news': news,
        'new_neural_observations': 0, 'archive_writes': 0,
        'call_id': modal.current_function_call_id(), 'input_id': modal.current_input_id()}
    print(json.dumps(result), flush=True)
    return result


@app.function(image=image, volumes={'/state': volume}, cpu=(2, 2), memory=(8192, 8192),
    max_containers=1, min_containers=0, scaledown_window=2, timeout=600,
    single_use_containers=True, retries=0, nonpreemptible=True,
    schedule=modal.Cron('2-59/5 * * * *'))
def worker():
    return exclusive('worker', _run)


def _run():
    import time
    from paperlab.core import atomic_json
    from paperlab.fly_online_completion import DIRECTORY, execute
    volume.reload()
    try:
        return execute('/state', '/opt/paperlab/registered', '/opt/paperlab/recovery', '/opt/paperlab',
            call_id=modal.current_function_call_id(), input_id=modal.current_input_id(), commit=volume.commit)
    except BaseException as exc:
        path = Path('/state')/DIRECTORY/'halt.json'
        if not path.exists():
            atomic_json(path, {'status': 'halted', 'call_id': modal.current_function_call_id(),
                'input_id': modal.current_input_id(), 'at': time.time(),
                'error_type': type(exc).__name__, 'error': str(exc)[:300]})
            volume.commit()
        raise
