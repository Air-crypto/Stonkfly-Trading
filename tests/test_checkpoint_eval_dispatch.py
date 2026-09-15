import json
from pathlib import Path
from unittest.mock import Mock
import pytest
from checkpoint_eval_dispatch import dispatch, priority_order, digest, read, atomic_json


@pytest.fixture
def setup(tmp_path):
    base = tmp_path/'checkpoint-eval/evaluation-100'
    base.mkdir(parents=True)
    (tmp_path/'solana-online').mkdir()
    plan = dict(cutoff=100, policies=['untrained','cash','always_long','old','new'],
                checkpoints=[dict(id='old',ended=10),dict(id='new',ended=99)],
                tape=dict(id='future'), source_hashes={'sealed_worker':'original'})
    atomic_json(base/'plan.json', plan)
    atomic_json(base.parent/'control.json', dict(enabled=True,batch='evaluation-100',pending=None))
    atomic_json(tmp_path/'solana-online/control.json', dict(enabled=True,next_at=5000,pending=None))
    calls = dict(poll=Mock(return_value=None),spawn=Mock(return_value='fc-test'),
                 commit=Mock(),refresh_training=Mock(),worker_busy=Mock(return_value=False),clock=lambda:1000)
    def run():return dispatch(tmp_path,**calls)
    return tmp_path,base,calls,run


def test_prioritizes_newest_without_mutating_plan_or_excluding_controls(setup):
    root,base,calls,run=setup
    before=(base/'plan.json').read_bytes()
    c=run()
    assert c['pending']==dict(policy='new',call_id='fc-test')
    assert (base/'plan.json').read_bytes()==before
    assert read(base/'dispatch-policy-v1.json')['policies']==['new','untrained','cash','always_long','old']
    calls['spawn'].assert_called_once_with('evaluation-100','new')


def test_completed_call_dispatches_next_in_same_tick(setup):
    root,base,calls,run=setup
    run();(base/'new').mkdir()
    atomic_json(base/'new/completed.json',dict(status='completed',policy='new'))
    calls['poll'].return_value=dict(status='completed',policy='new')
    calls['spawn'].reset_mock()
    assert run()['pending']['policy']=='untrained'
    calls['spawn'].assert_called_once_with('evaluation-100','untrained')


def test_active_call_and_unknown_dispatch_are_not_retried(setup):
    root,base,calls,run=setup
    run();calls['spawn'].reset_mock()
    assert run()['status']=='evaluation_running'
    c=read(base.parent/'control.json');c['pending']['call_id']=None;atomic_json(base.parent/'control.json',c)
    assert run()['status']=='ambiguous_dispatch_requires_review'
    calls['spawn'].assert_not_called()


def test_uncertain_spawn_retains_durable_intent(setup):
    root,base,calls,run=setup
    calls['spawn'].side_effect=RuntimeError('network response lost')
    with pytest.raises(RuntimeError):run()
    assert read(base.parent/'control.json')['pending']==dict(policy='new',call_id=None)
    assert run()['status']=='ambiguous_dispatch_requires_review'
    assert calls['spawn'].call_count==1


@pytest.mark.parametrize('patch',[dict(enabled=False),dict(audit_pause={'reason':'review'}),
    dict(error='failure'),dict(budget_paused_until=6000),dict(status='storage_paused_requires_review')])
def test_preserves_training_pauses(setup,patch):
    root,base,calls,run=setup
    path=root/'solana-online/control.json';live=read(path);live.update(patch);atomic_json(path,live)
    before=path.read_bytes()
    assert run()['status']=='respecting_training_pause'
    assert path.read_bytes()==before
    calls['spawn'].assert_not_called();calls['refresh_training'].assert_not_called()


def test_evaluation_pause_is_not_cleared(setup):
    root,base,calls,run=setup
    path=base.parent/'control.json';c=read(path);c['enabled']=False;atomic_json(path,c);before=path.read_bytes()
    assert run()['status']=='respecting_evaluation_pause'
    assert path.read_bytes()==before
    calls['spawn'].assert_not_called()


def test_finished_training_is_reconciled_before_dispatch(setup):
    root,base,calls,run=setup
    atomic_json(root/'solana-online/control.json',dict(enabled=True,next_at=500,pending=dict(call_id='fc-training')))
    calls['poll'].return_value=dict(status='completed')
    calls['refresh_training'].return_value=dict(enabled=True,next_at=5000,pending=None)
    assert run()['status']=='evaluation_dispatched'
    calls['refresh_training'].assert_called_once()


def test_due_training_takes_priority(setup):
    root,base,calls,run=setup
    atomic_json(root/'solana-online/control.json',dict(enabled=True,next_at=999,pending=None))
    calls['refresh_training'].return_value=dict(enabled=True,next_at=999,pending=dict(call_id='fc-training'))
    assert run()['status']=='waiting_for_training_gap'
    calls['spawn'].assert_not_called()


@pytest.mark.parametrize('gap,expected',[(1249,'waiting_for_training_gap'),(1250,'evaluation_dispatched')])
def test_preserves_full_worker_timeout_headroom(setup,gap,expected):
    root,base,calls,run=setup
    atomic_json(root/'solana-online/control.json',dict(enabled=True,next_at=1000+gap,pending=None))
    assert run()['status']==expected


def test_busy_lease_is_never_cleared_or_dispatched_over(setup):
    root,base,calls,run=setup;calls['worker_busy'].return_value=True
    assert run()['status']=='worker_busy_wait';calls['spawn'].assert_not_called()


def test_existing_attempt_is_never_overwritten(setup):
    root,base,calls,run=setup;(base/'new').mkdir()
    assert run()['status']=='ambiguous_attempt_requires_review';calls['spawn'].assert_not_called()


@pytest.mark.parametrize('status,expected',[('budget_stopped','budget_paused_requires_review'),
    ('failed','evaluation_failed'),('writer_busy','worker_busy_wait')])
def test_outcomes_do_not_trigger_blind_retries(setup,status,expected):
    root,base,calls,run=setup;run();calls['spawn'].reset_mock()
    calls['poll'].return_value=dict(status=status)
    assert run()['status']==expected;calls['spawn'].assert_not_called()


def test_plan_change_and_future_checkpoint_fail_closed(setup):
    root,base,calls,run=setup
    p=read(base/'plan.json');p['checkpoints'][1]['ended']=100
    with pytest.raises(ValueError,match='cutoff'):priority_order(p)
    run();c=read(base.parent/'control.json');c['pending']=None;atomic_json(base.parent/'control.json',c)
    p=read(base/'plan.json');p['source_hashes']['sealed_worker']='different';atomic_json(base/'plan.json',p)
    assert run()['status']=='dispatch_manifest_requires_review'


def test_scheduler_files_do_not_change_worker_source_seal(tmp_path):
    from paperlab.checkpoint_eval import source_fingerprint
    for name in ('cloud.py','solana_cloud.py','checkpoint_eval_cloud.py'):(tmp_path/name).write_text('# sealed')
    before=source_fingerprint(tmp_path)
    (tmp_path/'checkpoint_eval_dispatch.py').write_text('# external scheduling')
    (tmp_path/'checkpoint_eval_dispatch_cloud.py').write_text('# external app')
    assert source_fingerprint(tmp_path)==before


def test_dispatch_budget_is_separate_and_fails_closed(setup):
    from checkpoint_eval_dispatch import reserve_dispatch,settle_dispatch
    root,base,calls,run=setup
    atomic_json(root/'budget.json',dict(untouched=True))
    before=(root/'budget.json').read_bytes()
    r=reserve_dispatch(root,1000)
    result=settle_dispatch(root,r,1002)
    assert 0<result['estimated_compute_usd']<r['reserved']
    assert (root/'budget.json').read_bytes()==before
    p=base.parent/'dispatch-budget.json';state=read(p);state['months'][r['month']]=3.75;atomic_json(p,state)
    assert reserve_dispatch(root,1003) is None
    assert (root/'budget.json').read_bytes()==before


def test_cloud_poll_refreshes_completed_result_before_dispatch_validation(setup,monkeypatch):
    from types import SimpleNamespace
    import checkpoint_eval_dispatch_cloud as cloud
    root,base,calls,run=setup
    run();events=[];result=dict(status='completed',policy='new')
    def get(**kwargs):events.append('completed');return result
    def reload():
        events.append('reload');(base/'new').mkdir()
        atomic_json(base/'new/completed.json',result)
    monkeypatch.setattr(cloud.modal.FunctionCall,'from_id',lambda _:SimpleNamespace(get=get))
    monkeypatch.setattr(cloud,'volume',SimpleNamespace(reload=reload))
    calls['poll']=cloud.poll_completed
    assert run()['pending']['policy']=='untrained'
    assert events==['completed','reload']


def test_cloud_poll_does_not_treat_pending_as_completed(monkeypatch):
    from types import SimpleNamespace
    import checkpoint_eval_dispatch_cloud as cloud
    get=Mock(side_effect=TimeoutError);reload=Mock()
    monkeypatch.setattr(cloud.modal.FunctionCall,'from_id',lambda _:SimpleNamespace(get=get))
    monkeypatch.setattr(cloud,'volume',SimpleNamespace(reload=reload))
    assert cloud.poll_completed('fc-live') is None
    reload.assert_not_called()


@pytest.fixture
def missing_result(setup):
    root,base,calls,run=setup;run()
    result=dict(status='completed',policy='new',tape='future',weights_unchanged=True,pnl_usd=-1.)
    calls['poll'].return_value=result
    assert run()['error']=='MissingDurableEvaluationResult'
    (base/'new').mkdir()
    atomic_json(base/'new/completed.json',result)
    atomic_json(base/'new/owner.json',dict(call_id='fc-test'))
    return root,base,calls,result


def test_missing_result_recovery_requires_exact_persisted_result_and_keeps_scores(missing_result):
    from checkpoint_eval_dispatch import recover_result
    root,base,calls,result=missing_result
    before=read(base.parent/'control.json');raw=(base/'new/completed.json').read_bytes()
    r=recover_result(root,'fc-test',poll=calls['poll'],commit=calls['commit'],clock=lambda:1001)
    assert r['status']=='verified_result_reconciled'
    c=read(base.parent/'control.json');assert c['enabled'] and c['pending'] is None and 'error' not in c
    assert (base/'new/completed.json').read_bytes()==raw
    assert read(base/'reconciliation-recovery/fc-test.json')['before']==before
    with pytest.raises(ValueError):recover_result(root,'fc-test',poll=calls['poll'],commit=calls['commit'])


@pytest.mark.parametrize('problem',['pending','mismatch','owner','missing','audit','wrong_call'])
def test_missing_result_recovery_rejects_uncertain_or_unrelated_state(missing_result,problem):
    from checkpoint_eval_dispatch import recover_result
    root,base,calls,result=missing_result
    expected='fc-test'
    if problem=='pending':calls['poll'].return_value=None
    if problem=='mismatch':atomic_json(base/'new/completed.json',{**result,'pnl_usd':999})
    if problem=='owner':atomic_json(base/'new/owner.json',dict(call_id='fc-other'))
    if problem=='missing':(base/'new/completed.json').unlink()
    if problem=='audit':
        p=base.parent/'control.json';c=read(p);c['audit_pause']=True;atomic_json(p,c)
    if problem=='wrong_call':expected='fc-other'
    before=(base.parent/'control.json').read_bytes()
    with pytest.raises((ValueError,FileNotFoundError)):
        recover_result(root,expected,poll=calls['poll'],commit=calls['commit'])
    assert (base.parent/'control.json').read_bytes()==before
    assert not (base/'reconciliation-recovery').exists()
