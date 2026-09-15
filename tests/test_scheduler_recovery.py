from pathlib import Path
import copy, json, importlib.util
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from paperlab.core import atomic_json
import training_service_v2 as training
import checkpoint_preparation as preparation


def test_storage_pause_reason_survives_later_ticks(tmp_path,monkeypatch):
    monkeypatch.setattr(training,'ARCHIVE_LIMIT_BYTES',1)
    p=tmp_path/'solana-live/evidence';p.parent.mkdir();p.write_bytes(b'keep')
    calls=dict(dispatch=Mock(),poll=Mock(),commit=lambda:None,now=1000)
    first=training.service(tmp_path,**calls)
    assert not first['enabled'] and first['status']=='storage_cap_requires_review'
    for now in (2000,3000):
        calls['now']=now;c=training.service(tmp_path,**calls)
        assert c['status']==c['pause_reason']=='storage_cap_requires_review' and c['paused_at']==1000
    assert p.read_bytes()==b'keep';calls['dispatch'].assert_not_called()


def paused_root(root):
    parent='solana-online-old'
    atomic_json(root/'solana-live'/parent/'completed.json',dict(status='completed',paper_only=True))
    c=dict(enabled=False,pending=None,parent=parent,status='disabled',archive_bytes=9*1024**3,
           next_at=1000,mode='hourly',completed_windows=29,episode_totals={'sum_episode_pnl_usd':-99})
    atomic_json(root/'solana-online/control.json',c)
    return parent,c


def test_storage_recovery_preserves_parent_losses_and_original_control(tmp_path):
    parent,before=paused_root(tmp_path)
    c=training.recover_storage(tmp_path,parent,commit=lambda:None,now=1000)
    assert c['enabled'] and c['parent']==parent and c['episode_totals']==before['episode_totals']
    assert c['next_at']==2800
    saved=json.loads((tmp_path/'solana-online'/('storage-recovery-'+parent+'.json')).read_text())
    assert saved['before']==before and saved['new_limit_bytes']>saved['previous_limit_bytes']
    with pytest.raises(ValueError):training.recover_storage(tmp_path,parent,commit=lambda:None,now=1001)


@pytest.mark.parametrize('patch',[dict(pending={'call_id':'active'}),dict(enabled=True),dict(error='failed'),
    dict(audit_pause=True),dict(budget_paused_until=5000),dict(status='failed_requires_review'),dict(archive_bytes=1)])
def test_storage_recovery_cannot_override_other_pauses(tmp_path,patch):
    parent,c=paused_root(tmp_path);c.update(patch);p=tmp_path/'solana-online/control.json';atomic_json(p,c)
    before=p.read_bytes()
    with pytest.raises(ValueError):training.recover_storage(tmp_path,parent,commit=lambda:None,now=1000)
    assert p.read_bytes()==before


@pytest.mark.parametrize('name',['test_dispatch_ambiguity_does_not_retry','test_budget_pause_preserves_parent',
    'test_episode_losses_survive_reset_and_next_window_keeps_cadence','test_episode_dispatch_requires_matching_completion_mode'])
def test_existing_scheduler_contracts_on_repaired_service(tmp_path,monkeypatch,name):
    spec=importlib.util.spec_from_file_location('old_scheduler_fixtures',Path(__file__).with_name('test_solana_online.py'))
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    monkeypatch.setattr(fixture,'service',training.service)
    getattr(fixture,name)(tmp_path)


@pytest.fixture
def prep(tmp_path,monkeypatch):
    base=tmp_path/'checkpoint-eval/evaluation-100'
    plan=dict(id='evaluation-100',cutoff=100,policies=['cash'],checkpoints=[],tape=None,source_hashes={'immutable':'yes'})
    atomic_json(base/'plan.json',plan)
    atomic_json(base.parent/'control.json',dict(enabled=True,batch='evaluation-100',pending=None,next_batch_at=9999))
    atomic_json(tmp_path/'solana-online/control.json',dict(enabled=True,next_at=9999))
    monkeypatch.setattr(preparation,'source_fingerprint',lambda _:plan['source_hashes'])
    monkeypatch.setattr(preparation,'register',Mock(return_value=[]))
    monkeypatch.setattr(preparation,'reserve',Mock(return_value=dict(started=1000,rate=1,startup_seconds=30)))
    monkeypatch.setattr(preparation,'settle',Mock(return_value={'estimated_compute_usd':1}))
    monkeypatch.setattr(preparation,'seal_tape',Mock(return_value={'id':'unseen','files':{}}))
    return tmp_path,base,plan


def test_preparation_seals_tape_without_changing_cohort_or_sources(prep):
    root,base,plan=prep
    c=preparation.prepare(root,'.',commit=lambda:None,clock=lambda:1001)
    actual=json.loads((base/'plan.json').read_text());tape=actual.pop('tape');before=copy.deepcopy(plan);before.pop('tape')
    assert actual==before and tape['id']=='unseen' and c['status']=='tape_sealed'
    assert json.loads((base.parent/'preparation-latest.json').read_text())['stage']=='completed'
    preparation.settle.assert_called_once()
    preparation.prepare(root,'.',commit=lambda:None,clock=lambda:1002)
    assert preparation.register.call_count==1 and preparation.reserve.call_count==1


def test_preparation_failure_is_durable_and_no_retry(prep):
    root,base,plan=prep;preparation.seal_tape.side_effect=ValueError('Invalid causal tape')
    with pytest.raises(ValueError):preparation.prepare(root,'.',commit=lambda:None,clock=lambda:1001)
    c=json.loads((base.parent/'control.json').read_text())
    assert not c['enabled'] and c['status']=='preparation_failed_requires_review'
    assert json.loads((base/'plan.json').read_text())==plan
    assert preparation.prepare(root,'.',commit=lambda:None)['status']=='respecting_evaluation_pause'
    assert preparation.seal_tape.call_count==1


def test_preparation_respects_disabled_training(prep):
    root,base,_=prep;atomic_json(root/'solana-online/control.json',dict(enabled=False))
    assert preparation.prepare(root,'.',commit=lambda:None)['status']=='respecting_training_pause'
    preparation.reserve.assert_not_called();preparation.register.assert_not_called()


def test_cloud_coordinator_entrypoint_uses_both_leases_and_reload(monkeypatch):
    import checkpoint_runtime_cloud as cloud
    calls=[]
    monkeypatch.setattr(cloud,'exclusive',lambda name,fn,*a:(calls.append(name),fn(*a))[1])
    monkeypatch.setattr(cloud,'volume',SimpleNamespace(reload=lambda:calls.append('reload'),commit=lambda:None))
    monkeypatch.setattr(preparation,'prepare',lambda *a,**kw:(calls.append('prepare'),{'status':'ok'})[1])
    assert cloud.coordinator.local()['status']=='ok'
    assert calls==['checkpoint-eval-coordinator','worker','reload','prepare']


def test_recovery_rejects_live_or_uncertain_coordinator(monkeypatch):
    import checkpoint_runtime_cloud as cloud
    owner={'call_id':'fc-old'};writers=Mock();writers.get.return_value=owner
    monkeypatch.setattr(cloud,'writers',writers)
    get=Mock(side_effect=TimeoutError('still running'))
    monkeypatch.setattr(cloud.modal.FunctionCall,'from_id',lambda _:SimpleNamespace(get=get))
    with pytest.raises(TimeoutError):cloud._recover('fc-old')
    writers.pop.assert_not_called()


def test_repaired_schedulers_leave_sealed_evaluator_sources_unchanged():
    from paperlab.checkpoint_eval import source_fingerprint
    plan=json.loads(Path('reports/all-pump-evaluation-plan-20260914.json').read_text())
    assert source_fingerprint('.')==plan['source_hashes']


def test_due_training_waits_for_shared_worker_without_dispatch_intent(tmp_path):
    dispatch=Mock(return_value='fc-next')
    c=training.service(tmp_path,dispatch=dispatch,poll=Mock(),commit=lambda:None,now=1000,
        mode='hourly',worker_busy=lambda:True)
    assert c['enabled'] and c['pending'] is None and c['status']=='waiting_for_shared_worker'
    dispatch.assert_not_called()
    c=training.service(tmp_path,dispatch=dispatch,poll=Mock(),commit=lambda:None,now=1060,
        mode='hourly',worker_busy=lambda:False)
    assert c['pending']['call_id']=='fc-next' and c['status']=='dispatched'
    dispatch.assert_called_once()


def test_prospective_request_freezes_latest_then_waits_for_future_data(prep,monkeypatch):
    from paperlab.checkpoint_eval import register,seal_tape
    from paperlab.solana_universe import QUOTE_PROTOCOL
    root,old_base,_=prep
    old_plan=(old_base/'plan.json').read_bytes()
    control=old_base.parent/'control.json'
    atomic_json(control,dict(enabled=True,batch=None,pending=None,next_batch_at=9999,
                            last_completed_batch=old_base.name))
    monkeypatch.setattr(preparation,'register',register)
    monkeypatch.setattr(preparation,'seal_tape',seal_tape)
    def checkpoint(name,start,end):
        p=root/'solana-live'/name;p.mkdir(parents=True)
        atomic_json(p/'completed.json',dict(status='completed',account_mode='fresh_training_episode',
            started=start,ended=end,readout_updates=10))
        for n in ('fly-final.npz','head-final.pt','events.db','fx.jsonl'):(p/n).write_bytes(b'fixture')
        row=dict(at=end,observation_at=start,event_cursor=0,fx_state=dict(sol_usd=0,seen_at=0),
                 quote_protocol=QUOTE_PROTOCOL,decision=None,neural=None)
        (p/'decisions.jsonl').write_text(json.dumps(row)+'\n')
        return p.name
    earlier=checkpoint('solana-online-earlier',100,200)
    latest=checkpoint('solana-online-latest',300,400)
    c=preparation.request_prospective(root,'.',expected_last_batch=old_base.name,commit=lambda:None,clock=lambda:1000)
    plan_path=old_base.parent/c['batch']/'plan.json';plan=json.loads(plan_path.read_text())
    assert c['status']=='waiting_for_unseen_market_window'
    assert plan['cutoff']==1000 and plan['primary_policy']==latest and plan['tape'] is None
    assert plan['policies']==['untrained','cash','always_long',earlier,latest]
    assert not plan['learning'] and plan['initial_cash']==1000
    assert c['next_batch_at']==86400 and c['prospective_request']['previous_next_batch_at']==9999
    assert (old_base/'plan.json').read_bytes()==old_plan
    # A duplicate request must not move the sealed cutoff or add policies.
    assert preparation.request_prospective(root,'.',expected_last_batch=old_base.name,
        commit=lambda:None,clock=lambda:1001)['status']=='existing_evaluation_preserved'
    assert json.loads(plan_path.read_text())==plan
    checkpoint('solana-online-overlap',999,1100)
    preparation.prepare(root,'.',commit=lambda:None,clock=lambda:1101)
    assert json.loads(plan_path.read_text())['tape'] is None
    future=checkpoint('solana-online-future',1102,1200)
    preparation.prepare(root,'.',commit=lambda:None,clock=lambda:1201)
    sealed=json.loads(plan_path.read_text())
    assert sealed['tape']['id']==future and sealed['policies']==plan['policies']
    assert sealed['tape']['started']>sealed['cutoff']>max(cp['ended'] for cp in sealed['checkpoints'])


@pytest.mark.parametrize('patch',[dict(enabled=False),dict(error='failed'),dict(audit_pause=True),
    dict(budget_paused_until=5000),dict(status='preparation_budget_stopped')])
@pytest.mark.parametrize('target',['checkpoint-eval','solana-online'])
def test_prospective_request_preserves_pauses(prep,patch,target):
    root,base,_=prep;p=root/target/'control.json';c=json.loads(p.read_text());c.update(patch);atomic_json(p,c)
    before=p.read_bytes()
    with pytest.raises(ValueError,match='pause'):
        preparation.request_prospective(root,'.',expected_last_batch='old',commit=lambda:None,clock=lambda:1000)
    assert p.read_bytes()==before
    preparation.reserve.assert_not_called()


def test_prospective_request_rejects_stale_completed_batch(prep):
    root,base,_=prep;before=(base.parent/'control.json').read_bytes()
    with pytest.raises(ValueError,match='Completed evaluation changed'):
        preparation.request_prospective(root,'.',expected_last_batch='wrong',commit=lambda:None,clock=lambda:1000)
    assert (base.parent/'control.json').read_bytes()==before


def test_cloud_prospective_entrypoint_uses_both_leases(monkeypatch):
    import checkpoint_runtime_cloud as cloud
    calls=[]
    monkeypatch.setattr(cloud,'exclusive',lambda name,fn,*a:(calls.append(name),fn(*a))[1])
    monkeypatch.setattr(cloud,'volume',SimpleNamespace(reload=lambda:calls.append('reload'),commit=lambda:None))
    def request(*a,**kw):
        calls.append(kw['expected_last_batch']);return {'status':'waiting_for_unseen_market_window'}
    monkeypatch.setattr(preparation,'request_prospective',request)
    assert cloud.request_prospective.local('old')['status']=='waiting_for_unseen_market_window'
    assert calls==['checkpoint-eval-coordinator','worker','reload','old']
