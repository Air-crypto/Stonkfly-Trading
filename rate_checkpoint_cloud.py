"""Unscheduled checkpoint export for the next prospective rate comparison."""
from pathlib import Path
import modal
from cloud import image, volume, discovery_volume, exclusive, FULL_FLY, UNIVERSE

if not (FULL_FLY and UNIVERSE):
    raise ValueError('Checkpoint export requires the full fly and universe settings')

ROOT = Path(__file__).parent
app = modal.App('fly-paper-rate-checkpoint-12')
image = (image.add_local_file(ROOT/'cloud.py', '/opt/paperlab/cloud.py', copy=True)
         .add_local_file(ROOT/'rate_checkpoint_cloud.py', '/opt/paperlab/rate_checkpoint_cloud.py', copy=True))


def _verify_import():
    import importlib
    importlib.import_module('rate_checkpoint_cloud')
    from paperlab.fly_rate_checkpoint import source_hashes
    source_hashes()
    print('Checkpoint export entrypoint and source manifest verified', flush=True)


image = image.run_function(_verify_import, cpu=.125, memory=256, timeout=60)


@app.function(image=image, volumes={'/state': volume, '/discovery': discovery_volume},
              cpu=(2, 2), memory=(8192, 8192), max_containers=1, min_containers=0,
              scaledown_window=2, timeout=600, single_use_containers=True,
              retries=0, nonpreemptible=True)
def worker(request: dict):
    return exclusive('worker', _run, request)


def _run(request):
    from paperlab.fly_rate_checkpoint import execute
    volume.reload(); discovery_volume.reload()
    return execute(request, '/state', '/discovery', call_id=modal.current_function_call_id(),
                   input_id=modal.current_input_id(), commit=volume.commit)
