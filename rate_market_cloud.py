"""Dedicated study 12 worker; registration must exist before deployment."""
from pathlib import Path
import modal
from cloud import image,volume,discovery_volume,exclusive,FULL_FLY,UNIVERSE

if not (FULL_FLY and UNIVERSE):raise ValueError('Study 12 requires full fly and universe settings')
ROOT=Path(__file__).parent
REGISTRATION=ROOT/'reports/fly-rate-market-registration-12.json'
REFERENCE=ROOT/'runs/rate-market-reference-12'
if not REGISTRATION.is_file() or not REFERENCE.is_dir():
    raise ValueError('Prepare and verify the future registration and reference before deployment')

app=modal.App('fly-paper-rate-market-12')
image=(image.add_local_file(ROOT/'cloud.py','/opt/paperlab/cloud.py',copy=True)
       .add_local_file(ROOT/'rate_market_cloud.py','/opt/paperlab/rate_market_cloud.py',copy=True)
       .add_local_file(REGISTRATION,'/opt/paperlab/reports/'+REGISTRATION.name,copy=True)
       .add_local_dir(REFERENCE,'/opt/paperlab/runs/rate-market-reference-12',copy=True))


def _verify_import():
    import importlib,json
    importlib.import_module('rate_market_cloud')
    from paperlab.fly_rate_schedule import source_hashes,verify_inputs
    root=Path('/opt/paperlab')
    verify_inputs(json.loads((root/'reports/fly-rate-market-registration-12.json').read_text()),
                  root/'reports',root/'runs/rate-market-reference-12')
    source_hashes()
    print('Study 12 entrypoint, pinned checkpoint evidence and source manifest verified',flush=True)


image=image.run_function(_verify_import,cpu=.125,memory=512,timeout=60)


@app.function(image=image,volumes={'/state':volume,'/discovery':discovery_volume},
              cpu=(2,2),memory=(8192,8192),max_containers=1,min_containers=0,scaledown_window=2,
              timeout=600,single_use_containers=True,retries=0,nonpreemptible=True,
              schedule=modal.Cron('2-59/5 * * * *'))
def worker():
    return exclusive('worker',_run)


def _run():
    from paperlab.fly_rate_runtime import budgeted_execute
    volume.reload();discovery_volume.reload()
    return budgeted_execute('/state','/discovery','/opt/paperlab/reports',
        '/opt/paperlab/runs/rate-market-reference-12',call_id=modal.current_function_call_id(),
        input_id=modal.current_input_id(),commit=volume.commit)
