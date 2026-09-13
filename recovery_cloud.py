"""Separate, temporary study 11 recovery; the deployed main app is unchanged."""
from pathlib import Path
import modal

from cloud import image, volume, exclusive

ROOT = Path(__file__).parent
app = modal.App('fly-paper-recovery-11')


def _verify_import():
    # Modal supplies its SDK to function containers, including build functions.
    # A plain Docker RUN command does not have that injected runtime.
    import importlib
    importlib.import_module('recovery_cloud')
    print('Recovery entrypoint import passed', flush=True)


image = (image.env({'PYTHONHASHSEED':'0'})
    .add_local_file(ROOT/'cloud.py', '/opt/paperlab/cloud.py', copy=True)
    .add_local_file(ROOT/'recovery_cloud.py', '/opt/paperlab/recovery_cloud.py', copy=True)
    .add_local_file(ROOT/'reports/fly-online-recovery-protocol-11.json', '/opt/paperlab/recovery/protocol.json', copy=True))
for name in ('fly-online-failure-11.json', 'fly-online-control-first-11.json', 'fly-online-control-second-11.json',
             'fly-news-process-probe-11.json', 'fly-online-recovery-preflight-failure-11.json'):
    image = image.add_local_file(ROOT/'reports'/name, '/opt/paperlab/recovery/'+name, copy=True)
image = image.run_function(_verify_import, cpu=.125, memory=256, timeout=60)


@app.function(image=image, volumes={'/state':volume}, cpu=(2,2), memory=(8192,8192),
    max_containers=1, min_containers=0, scaledown_window=2, timeout=600,
    single_use_containers=True, retries=0, schedule=modal.Cron('2-59/5 * * * *'))
def worker():
    return exclusive('worker', _run)


def _run():
    import time
    from paperlab.core import atomic_json
    from paperlab.fly_online_recovery import DIRECTORY, execute
    volume.reload()
    try:
        return execute('/state', '/opt/paperlab/registered', '/opt/paperlab/recovery', '/opt/paperlab',
            call_id=modal.current_function_call_id(), input_id=modal.current_input_id(), commit=volume.commit)
    except Exception as exc:
        path = Path('/state')/DIRECTORY/'halt.json'
        if not path.exists():
            atomic_json(path, {'status':'halted', 'call_id':modal.current_function_call_id(),
                'input_id':modal.current_input_id(), 'at':time.time(),
                'error_type':type(exc).__name__, 'error':str(exc)[:300]})
            volume.commit()
        raise
