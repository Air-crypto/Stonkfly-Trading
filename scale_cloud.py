"""One unscheduled, non-preemptible assay under the shared paper-worker budget."""
from pathlib import Path
import modal

from cloud import image, volume, exclusive, FULL_FLY, UNIVERSE

if not (FULL_FLY and UNIVERSE):
    raise ValueError('Learning-rate assay requires PAPERLAB_FLY=1 and PAPERLAB_UNIVERSE=1')

ROOT = Path(__file__).parent
app = modal.App('fly-paper-scale-01')
image = (image.add_local_file(ROOT/'cloud.py', '/opt/paperlab/cloud.py', copy=True)
    .add_local_file(ROOT/'selective_cloud.py', '/opt/paperlab/selective_cloud.py', copy=True)
    .add_local_file(ROOT/'scale_cloud.py', '/opt/paperlab/scale_cloud.py', copy=True))
for name in ('fly-first-drive-01.json', 'fly-market-study-11-preregistration.json', 'fly-learning-scale-protocol-01.json',
             'fly-selective-audit-01.json', 'fly-selective-memory-01.json'):
    image = image.add_local_file(ROOT/'reports'/name, '/opt/paperlab/reports/'+name, copy=True)


def _verify_import():
    import importlib
    importlib.import_module('scale_cloud')
    from paperlab.fly_learning_scale_cloud import run_budgeted_request
    from paperlab.fly_learning_scale import source_hashes
    source_hashes()  # Fail the build if any execution or audit dependency is missing.
    print('Learning-rate assay worker import passed', flush=True)


image = image.run_function(_verify_import, cpu=.125, memory=256, timeout=60)


@app.function(image=image, volumes={'/state': volume}, cpu=(2, 2), memory=(8192, 8192),
    max_containers=1, min_containers=0, scaledown_window=2, timeout=360,
    single_use_containers=True, retries=0, nonpreemptible=True)
def worker(debug: dict):
    return exclusive('worker', _run, debug)


def _run(request):
    from paperlab.fly_learning_scale_cloud import run_budgeted_request
    volume.reload()
    return run_budgeted_request(request, '/state', '/state/fly-data',
        call_id=modal.current_function_call_id(), input_id=modal.current_input_id(),
        commit=volume.commit)
