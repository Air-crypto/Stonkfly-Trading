"""The cloud observer reattaches exact calls and cannot dispatch compute."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from paperlab.core import atomic_json, digest
from paperlab.fly_online_cloud import observe, validate_selection_receipt
from paperlab.fly_online_schedule import execute_due
from test_fly_online_schedule import setup, arm


@pytest.fixture
def cloud(setup,monkeypatch):
    import modal
    s=setup;calls=[];replies={}
    class Volume:
        def read_file(self,path):
            # Model the deployed /state mount while retaining isolated local files.
            return [(s.args[0]/path.lstrip('/')).read_bytes().replace(str(s.args[0]).encode(),b'/state')]
    class Call:
        def __init__(self,identity):self.identity=identity
        def get(self,timeout):
            assert timeout==0;calls.append(self.identity)
            value=replies.get(self.identity)
            if value is None:raise TimeoutError('still running')
            if isinstance(value,Exception):raise value
            return json.loads(json.dumps(value).replace(str(s.args[0]),'/state'))
    monkeypatch.setattr(modal.Volume,'from_name',lambda *a,**k:Volume())
    monkeypatch.setattr(modal.FunctionCall,'from_id',lambda identity:Call(identity))
    monkeypatch.setattr(modal.Function,'from_name',lambda *a,**k:pytest.fail('Observer must not submit compute'))
    monkeypatch.setattr('paperlab.fly_online_cloud.expected_sources',lambda:s.sources.copy())
    monkeypatch.setattr('paperlab.fly_online_cloud.validate',lambda env:env['plan'])
    s.kw['now']=s.r['development_start']-.5
    monkeypatch.setattr('paperlab.fly_online_schedule.time.time',lambda:s.r['end']+1)
    return SimpleNamespace(s=s,calls=calls,replies=replies,out=s.args[0].parent/'observer')


def arm_cloud(c):
    arm(c.s);c.replies['fc-armed']={'status':'paper_research','study_11':{'status':'paper_online_collecting'}}


def capture(c):
    result=execute_due(*c.s.args,**c.s.kw)
    c.replies[c.s.kw['call_id']]={'status':'paper_research','study_11':result}
    return result


def test_absent_registration_does_not_invent_a_call_or_submit(cloud):
    c=cloud
    assert observe(c.s.r,c.out)['status']=='waiting_for_cloud_arming'
    assert not c.calls and not c.out.exists()


def test_arming_receipt_reuses_its_exact_call_until_completion(cloud):
    c=cloud;arm(c.s)
    for _ in range(2):assert observe(c.s.r,c.out)['status']=='arming_call_pending'
    assert c.calls==['fc-armed','fc-armed']
    c.replies['fc-armed']={'status':'paper_research','study_11':{'status':'paper_online_collecting'}}
    assert observe(c.s.r,c.out)['completed_chunks']==0
    assert json.loads((c.out/'progress.json').read_text())['completed_chunks']==0


def test_uncertain_sealing_reattaches_owner_before_reporting_state(cloud,monkeypatch):
    c=cloud;arm_cloud(c)
    def interrupted(*a):raise KeyboardInterrupt()
    monkeypatch.setattr('paperlab.fly_online_schedule.seal',interrupted)
    with pytest.raises(KeyboardInterrupt):execute_due(*c.s.args,**c.s.kw)
    for _ in range(2):assert observe(c.s.r,c.out)['status']=='sealing_call_pending'
    assert c.calls==['fc-armed','fc-next','fc-armed','fc-next']
    c.replies['fc-next']={'status':'paper_online_seal_unresolved'}
    result=observe(c.s.r,c.out)
    assert result['status']=='sealing_receipt_unresolved'
    assert json.loads((c.out/'progress.json').read_text())==result


def test_saved_completed_chunk_waits_for_outer_worker_return_without_replacement(cloud):
    c=cloud;arm_cloud(c);capture(c);c.replies.pop('fc-next')
    for _ in range(2):assert observe(c.s.r,c.out)['status']=='chunk_worker_return_pending'
    assert c.calls==['fc-armed','fc-next','fc-armed','fc-next']


def test_live_claim_reattaches_and_failed_call_does_not_become_a_new_submission(cloud):
    c=cloud;arm_cloud(c)
    def fail(*a,**kw):raise KeyboardInterrupt()
    c.s.kw['run']=fail
    with pytest.raises(KeyboardInterrupt):execute_due(*c.s.args,**c.s.kw)
    assert observe(c.s.r,c.out)['status']=='chunk_call_pending'
    c.replies['fc-next']=RuntimeError('remote worker terminated')
    with pytest.raises(RuntimeError,match='terminated'):observe(c.s.r,c.out)
    assert c.calls[-1]=='fc-next'


def test_completed_chunk_identity_is_pinned_between_reads(cloud):
    c=cloud;arm_cloud(c);capture(c)
    assert observe(c.s.r,c.out)['completed_chunks']==1
    path=c.s.root/'chunks'/c.s.calls[0]/'receipt.json'
    receipt=json.loads(path.read_text());receipt['call_id']='fc-replacement';atomic_json(path,receipt)
    with pytest.raises(ValueError,match='replaced'):observe(c.s.r,c.out)
    assert 'fc-replacement' not in c.calls


def test_observer_rejects_changed_local_sources_before_call_lookup(cloud):
    c=cloud;arm_cloud(c);c.s.sources['cloud.py']='e'*64
    with pytest.raises(ValueError,match='source differs'):observe(c.s.r,c.out)
    assert not c.calls


def test_status_reconstructs_every_chunk_and_selection_without_running_a_model(cloud):
    c=cloud;arm_cloud(c)
    for i in range(16):
        c.s.kw.update(call_id=f'fc-chunk-{i}',input_id=f'in-chunk-{i}');capture(c)
    execute_due(*c.s.args,**c.s.kw)
    status=observe(c.s.r,c.out)
    assert status['status']=='captured_pending_audit' and status['completed_chunks']==16 and status['audited_chunks']==0
    assert c.calls==['fc-armed']+[f'fc-chunk-{i}' for i in range(16)]
    assert not (c.out/'report.json').exists() and not (c.out/'views').exists()


def test_price_audit_precedes_any_neural_download_or_view_publication(cloud,monkeypatch):
    c=cloud;arm_cloud(c);capture(c)
    import hashlib
    env=copy.deepcopy(c.s.env)
    env['plan']['snapshot_sha256']=hashlib.sha256((c.s.root/'universe.db').read_bytes()).hexdigest()
    # Only the stub envelope is expanded; this test stops at the raw-price audit.
    atomic_json(c.s.root/'plan.json',env)
    seen=[]
    def reject(*a):seen.append(a);raise ValueError('raw prices rejected')
    monkeypatch.setattr('paperlab.fly_online_cloud.audit_prices',reject)
    monkeypatch.setattr('paperlab.fly_online_cloud.audit_chunk',lambda *a:pytest.fail('Prices first'))
    with pytest.raises(ValueError,match='raw prices'):observe(c.s.r,c.out,audit=True,data='graph')
    assert len(seen)==1 and not (c.out/'views').exists()


@pytest.mark.parametrize('damage',[None,'raw_trace','starting_state'])
def test_full_observer_audit_downloads_all_chunks_and_reconstructs_report(cloud,monkeypatch,damage):
    """Exercise observer wiring; native arithmetic is tested in test_fly_online."""
    c=cloud;arm_cloud(c);s=c.s;original=s.kw['run'];seen=[]
    s.env['plan']['snapshot_sha256']=digest(s.args[1]/'universe-snapshot.db')
    def record(*args,**kw):
        result=original(*args,**kw);out=args[4]
        artifacts={'initial-dynamics.npz':b'fresh activity fixture','pristine-memory.npz':b'pristine memory fixture',
            'trace/step-000.npz':result['chunk'].encode()}
        if damage=='starting_state' and result['chunk']=='test-pool1-trained_online_reset_rates':
            artifacts['initial-dynamics.npz']=b'different starting activity'
        for name,value in artifacts.items():
            path=out/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(value)
        result['native_build']={'binary_sha256':'d'*64}
        result['artifact_sha256']={name:digest(out/name) for name in artifacts}
        atomic_json(out/'summary.json',result)
        return result
    s.kw['run']=record
    for i in range(16):
        s.kw.update(call_id=f'fc-chunk-{i}',input_id=f'in-chunk-{i}');capture(c)
    if damage=='raw_trace':
        (s.root/'chunks'/s.calls[0]/'artifacts/trace/step-000.npz').write_bytes(b'corrupted after capture')
    def prices(envelope,path):
        assert digest(path)==envelope['plan']['snapshot_sha256'];seen.append('prices')
        return {'status':'raw_price_audit_fixture'}
    def audit(envelope,summary,path,data):
        assert seen[0]=='prices' and data=='graph fixture'
        assert all(digest(path/name)==sha for name,sha in summary['artifact_sha256'].items())
        seen.append(summary['chunk'])
        return ({'status':'paper_online_chunk_audited','verification':{'observations':24}},
            {'report':{'run_id':summary['chunk']},'validation_only':True})
    monkeypatch.setattr('paperlab.fly_online_cloud.audit_prices',prices)
    monkeypatch.setattr('paperlab.fly_online_cloud.audit_chunk',audit)
    if damage:
        message='artifact hash differs' if damage=='raw_trace' else 'starting state'
        with pytest.raises(ValueError,match=message):observe(s.r,c.out,audit=True,data='graph fixture')
        assert not (c.out/'report.json').exists()
        if damage=='raw_trace':assert seen==['prices'] and not (c.out/'views').exists()
        return
    progress=observe(s.r,c.out,audit=True,data='graph fixture')
    assert progress['status']=='paper_online_study_audited' and progress['audited_chunks']==16
    assert seen==['prices']+s.calls
    report=json.loads((c.out/'report.json').read_text())
    assert report['audited'] and report['verification']['distinct_completed_worker_calls']==16
    assert report['selection']['selected']=='trained_online_reset_rates'
    assert report['total_equity']['trained_online_reset_rates']=={'development':1002,'test':1002}
    assert len(report['audit_sha256'])==16 and set(report['phase_diagnostics'])==set(s.r['cohort'])
    for name in s.calls:
        folder=c.out/'views'/name
        assert (folder/'step-000.npz').samefile(c.out/'chunks'/name/'trace/step-000.npz')
        assert json.loads((folder/'view.json').read_text())['validation_only']
        assert json.loads((folder/'remote.json').read_text())['call_id'].startswith('fc-chunk-')
    # Repeat observation reuses the existing files and does not run another model.
    assert observe(s.r,c.out,audit=True,data='graph fixture')['audited_chunks']==16
    assert len(s.calls)==16


def test_selection_timestamp_must_follow_development_and_precede_test():
    hashes={str(i):'a'*64 for i in range(8)}
    selection={'development_chunk_sha256':hashes}
    receipt={'call_id':'fc-selection','input_id':'in-selection','selected_at':20,'development_chunk_sha256':hashes}
    chunks={f'development-{i}':{'completed_at':10+i} for i in range(8)}
    chunks['test-0']={'claimed_at':21}
    validate_selection_receipt(selection,receipt,chunks)
    early=copy.deepcopy(receipt);early['selected_at']=16
    with pytest.raises(ValueError,match='preceded'):validate_selection_receipt(selection,early,chunks)
    late=copy.deepcopy(receipt);late['selected_at']=22
    with pytest.raises(ValueError,match='before'):validate_selection_receipt(selection,late,chunks)
