"""Wait for existing study 12 recordings; never submit or restart cloud work."""
import argparse
import fcntl
import json
import math
from pathlib import Path
import time

from .core import atomic_json
from .fly_rate_results import observe

PENDING={'waiting_for_cloud_arming','arming_call_pending','waiting_for_development',
         'development_collection','test_collection','evaluation','halt_call_pending',
         'sealing_call_pending','chunk_call_pending','chunk_worker_return_pending',
         'captured_waiting_for_cloud_summary'}
READY={'captured_pending_download','downloaded_pending_audit'}
ATTENTION={'registration_missed','halt_requires_inspection','sealing_receipt_unresolved',
           'chunk_receipt_unresolved'}


def watch(registration,output,*,interval=300,max_seconds=21600,read=observe,
          clock=time.monotonic,sleep=time.sleep,emit=print):
    if (not all(math.isfinite(v) for v in (interval,max_seconds))
            or not 30<=interval<=900 or not interval<=max_seconds<=86400):
        raise ValueError('Use a 30–900 second interval and a bounded duration up to one day')
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    with (root/'watch.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise ValueError('Another observer is already watching this output') from exc
        deadline=clock()+max_seconds;previous=None;timeouts=0;checks=0
        def save(status,**details):
            value={'status':status,'observer_checks':checks,'cloud_submissions':0,
                   'new_neural_observations':0,'observed_at':time.time(),**details}
            atomic_json(root/'watch.json',value)
            return value
        while clock()<deadline:
            checks+=1
            try:
                state=read(registration,root,download=False)
            except TimeoutError as exc:
                # Only the observer's exact built-in timeout is recoverable here.
                # Cloud function timeout subclasses must remain terminal attention.
                if type(exc) is not TimeoutError:
                    emit(json.dumps(save('observer_error',error_type=type(exc).__name__,study_failure_established=False)),flush=True)
                    raise
                timeouts+=1
                value=save('observer_read_timeout',consecutive_timeouts=timeouts,
                           study_failure_established=False)
                emit(json.dumps(value),flush=True)
                if timeouts>=3:return value
            except Exception as exc:
                emit(json.dumps(save('observer_error',error_type=type(exc).__name__,study_failure_established=False)),flush=True)
                raise
            else:
                timeouts=0;status=state['status']
                if status not in PENDING|READY|ATTENTION:
                    emit(json.dumps(save('observer_error',error_type='UnknownStudyState',study_failure_established=False)),flush=True)
                    raise ValueError('Unknown study state; inspect before continuing: '+str(status))
                summary={k:state[k] for k in ('status','completed_chunks','downloaded_chunks','chunk','call_id') if k in state}
                value=save('observer_ready' if status in READY else 'observer_attention' if status in ATTENTION else 'observer_waiting',
                           study=summary)
                if summary!=previous:
                    emit(json.dumps(value),flush=True);previous=summary
                if status in READY|ATTENTION:return value
            next_check=min(clock()+interval,deadline)
            while clock()<next_check:
                sleep(min(30,next_check-clock()))
        value=save('observer_deadline',study_failure_established=False)
        emit(json.dumps(value),flush=True);return value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--interval',type=float,default=300);p.add_argument('--max-seconds',type=float,default=21600)
    a=p.parse_args();result=watch(a.registration,a.out,interval=a.interval,max_seconds=a.max_seconds)
    if result['status']!='observer_ready':raise SystemExit(2)


if __name__=='__main__':main()
