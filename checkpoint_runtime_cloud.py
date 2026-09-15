"""Resource/lifecycle repair around the unchanged sealed evaluation worker."""
from pathlib import Path
import modal
from cloud import exclusive,volume,writers
from checkpoint_eval_cloud import image as sealed_image, _evaluate
ROOT=Path(__file__).resolve().parent if modal.is_local() else Path('/opt/paperlab')
image=sealed_image.add_local_file(ROOT/'checkpoint_preparation.py','/opt/paperlab/checkpoint_preparation.py',copy=True)
app=modal.App('fly-paper-checkpoint-eval')

@app.function(image=image,volumes={'/state':volume},cpu=(2,2),memory=(8192,8192),timeout=1200,
              max_containers=1,min_containers=0,retries=0,single_use_containers=True,nonpreemptible=True)
def worker(batch,policy):
    return exclusive('worker',_evaluate,batch,policy)

@app.function(image=image,volumes={'/state':volume},cpu=(2,2),memory=(8192,8192),timeout=600,
              max_containers=1,min_containers=0,retries=0,single_use_containers=True,nonpreemptible=True,
              schedule=modal.Cron('7,22,37,52 * * * *'))
def coordinator():
    return exclusive('checkpoint-eval-coordinator',lambda:exclusive('worker',_prepare))

def _prepare():
    from checkpoint_preparation import prepare
    volume.reload()
    return prepare('/state','/opt/paperlab',commit=volume.commit)

@app.function(image=image,volumes={'/state':volume},cpu=(2,2),memory=(8192,8192),timeout=600,
              max_containers=1,min_containers=0,retries=0,single_use_containers=True,nonpreemptible=True)
def request_prospective(expected_last_batch):
    return exclusive('checkpoint-eval-coordinator',lambda:exclusive('worker',_request_prospective,expected_last_batch))

def _request_prospective(expected_last_batch):
    from checkpoint_preparation import request_prospective as request
    volume.reload()
    return request('/state','/opt/paperlab',expected_last_batch=expected_last_batch,commit=volume.commit)

@app.function(image=image,volumes={'/state':volume},cpu=(.125,.125),memory=(512,512),timeout=60,
              max_containers=1,min_containers=0,retries=0,single_use_containers=True)
def recover_coordinator(expected_call_id):
    return exclusive('checkpoint-eval-recovery',_recover,expected_call_id)

def _recover(expected_call_id):
    import time
    from modal.exception import FunctionTimeoutError
    from paperlab.core import atomic_json
    key='checkpoint-eval-coordinator';owner=writers.get(key)
    if not owner or owner.get('call_id')!=expected_call_id:raise ValueError('Coordinator owner changed')
    try:
        modal.FunctionCall.from_id(expected_call_id).get(timeout=0)
    except FunctionTimeoutError:
        pass
    else:
        raise ValueError('Only a confirmed timed-out coordinator may be reclaimed')
    volume.reload();p=Path('/state/checkpoint-eval/lease-recovery')/(expected_call_id+'.json')
    if p.exists():raise ValueError('Recovery already recorded; inspect current state')
    atomic_json(p,dict(at=time.time(),owner=owner,confirmed_outcome='FunctionTimeoutError',
        action='release_exact_dead_coordinator_lease',authorization='User requested diagnosis and repair'))
    volume.commit()
    if writers.get(key)!=owner:raise ValueError('Coordinator owner changed during verification')
    removed=writers.pop(key)
    if removed!=owner:raise ValueError('Unexpected reclaimed owner')
    return dict(status='reclaimed_confirmed_timeout',owner=owner)
