"""Observer scheduling, failure boundaries and no duplicate model dispatch."""
import fcntl
import json

import pytest

from paperlab.fly_rate_watch import watch


def harness(tmp_path,states,**options):
    now=[0.];sleeps=[];calls=[];messages=[];states=iter(states)
    def read(registration,root,*,download):
        assert not download
        calls.append(now[0]);result=next(states)
        if isinstance(result,BaseException):raise result
        return result
    def sleep(seconds):
        assert 0<seconds<=30
        sleeps.append(seconds);now[0]+=seconds
    result=watch(tmp_path/'registration.json',tmp_path,interval=60,max_seconds=300,
                 read=read,clock=lambda:now[0],sleep=sleep,
                 emit=lambda text,**kw:messages.append(json.loads(text)),**options)
    return result,calls,sleeps,messages


def test_only_changed_milestones_are_emitted_and_ready_stops(tmp_path):
    initial={'status':'development_collection','completed_chunks':0}
    ready={'status':'captured_pending_download','completed_chunks':16}
    result,calls,sleeps,messages=harness(tmp_path,[initial,initial,ready])
    assert result['status']=='observer_ready'
    assert calls==[0,60,120] and len(messages)==2
    assert all(m['cloud_submissions']==0 for m in messages)
    assert result['study']==ready


def test_observation_timeout_does_not_mean_worker_failed(tmp_path):
    result,calls,_,messages=harness(tmp_path,[TimeoutError(),{'status':'chunk_call_pending','call_id':'fc-original'},
                                            {'status':'captured_pending_download','completed_chunks':16}])
    assert result['status']=='observer_ready' and len(calls)==3
    assert messages[0]['status']=='observer_read_timeout'
    assert messages[0]['study_failure_established'] is False
    assert messages[1]['study']['call_id']=='fc-original'


def test_three_read_timeouts_stop_only_the_observer(tmp_path):
    result,calls,_,_=harness(tmp_path,[TimeoutError(),TimeoutError(),TimeoutError()])
    assert len(calls)==3 and result['status']=='observer_read_timeout'
    assert result['consecutive_timeouts']==3 and not result['study_failure_established']


@pytest.mark.parametrize('error',[ValueError('source changed'),RuntimeError('function failed')])
def test_integrity_and_function_errors_are_not_retried(tmp_path,error):
    with pytest.raises(type(error),match=str(error)):harness(tmp_path,[error])
    assert json.loads((tmp_path/'watch.json').read_text())['status']=='observer_error'


def test_timeout_subclass_from_cloud_is_not_swallowed(tmp_path):
    class FunctionTimeout(TimeoutError):pass
    with pytest.raises(FunctionTimeout):harness(tmp_path,[FunctionTimeout('worker deadline')])
    assert json.loads((tmp_path/'watch.json').read_text())['error_type']=='FunctionTimeout'


@pytest.mark.parametrize('status',['registration_missed','halt_requires_inspection','sealing_receipt_unresolved','chunk_receipt_unresolved'])
def test_attention_stops_without_sleeping(tmp_path,status):
    result,calls,sleeps,_=harness(tmp_path,[{'status':status,'call_id':'fc-original'}])
    assert result['status']=='observer_attention' and len(calls)==1 and not sleeps


def test_unknown_status_stops(tmp_path):
    with pytest.raises(ValueError,match='Unknown study state'):harness(tmp_path,[{'status':'unknown'}])


def test_duplicate_observer_does_not_read_cloud(tmp_path):
    with (tmp_path/'watch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(ValueError,match='already watching'):
            harness(tmp_path,[])


def test_deadline_is_not_reported_as_study_failure(tmp_path):
    state={'status':'test_collection'}
    result,calls,_,_=harness(tmp_path,[state]*5)
    assert calls==[0,60,120,180,240]
    assert result['status']=='observer_deadline' and not result['study_failure_established']
