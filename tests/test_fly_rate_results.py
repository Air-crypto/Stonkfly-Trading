"""Cloud-result and offline-audit wiring with synthetic ledgers; no native brain."""
import asyncio
import copy
import fcntl
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from paperlab.core import atomic_json, digest
from paperlab.fly_rate_results import audit_saved, existing_call, observe
from paperlab.fly_rate_schedule import execute_due
from test_fly_rate_pipeline import sealed
from test_fly_rate_schedule import setup, arm


@pytest.fixture
def cloud(setup, monkeypatch):
    import modal
    s=setup; calls=[]; reads=[]; closed=[]; replies={}
    monkeypatch.setattr('paperlab.fly_rate_results.source_hashes',lambda:copy.deepcopy(s.sources))
    monkeypatch.setattr('time.time',lambda:s.r['end']+1)
    async def read(path):
        reads.append(path)
        try:
            content=(s.args[0]/path.lstrip('/')).read_bytes()
            if path.endswith('.json'):content=content.replace(str(s.args[0]).encode(),b'/state')
            yield content
        finally:closed.append(path)
    volume=SimpleNamespace(read_file=SimpleNamespace(aio=read))
    class Call:
        def __init__(self,identity):
            self.identity=identity;self.get=SimpleNamespace(aio=self._get)
        async def _get(self,timeout):
            assert timeout==0; calls.append(self.identity)
            value=replies.get(self.identity)
            if value is None:raise TimeoutError('observation pending')
            if isinstance(value,BaseException):raise value
            return json.loads(json.dumps(value).replace(str(s.args[0]),'/state'))
    monkeypatch.setattr(modal.Volume,'from_name',lambda *a,**kw:volume)
    monkeypatch.setattr(modal.FunctionCall,'from_id',lambda identity:Call(identity))
    monkeypatch.setattr(modal.Function,'from_name',lambda *a,**kw:pytest.fail('No model submission API is allowed'))
    monkeypatch.setattr(modal.Dict,'from_name',lambda *a,**kw:pytest.fail('No cloud writer mutation is allowed'))
    original=s.kw['run']
    def record(*args,**kw):
        result=original(*args,**kw);out=args[4]
        artifacts={'initial-dynamics.npz':b'synthetic fresh state','pristine-memory.npz':b'synthetic pristine memory',
                   'trace/step-01.npz':b'synthetic recording for '+result['chunk'].encode()}
        for name,value in artifacts.items():
            target=out/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(value)
        result['native_build']={'binary_sha256':'c'*64,'model':'synthetic wiring fixture'}
        result['artifact_sha256']={name:digest(out/name) for name in artifacts}
        atomic_json(out/'summary.json',result);return result
    s.kw['run']=record
    return SimpleNamespace(s=s,calls=calls,replies=replies,reads=reads,closed=closed,volume=volume,
                           out=s.args[0].parent/'download',audit_out=s.args[0].parent/'audit',
                           registration=s.args[2]/'fly-rate-market-registration-12.json')


def arm_cloud(c):
    arm(c.s)
    c.replies['fc-arm']={'status':'paper_rate_collecting','call_id':'fc-arm','input_id':'in-arm',
                        **{k:c.s.r[k] for k in ('development_start','test_start','end')}}


def capture(c,index):
    c.s.kw.update(call_id=f'fc-{index}',input_id=f'in-{index}')
    result=execute_due(*c.s.args,**c.s.kw)
    c.replies[f'fc-{index}']={**result,'call_id':f'fc-{index}','input_id':f'in-{index}'}
    return result


def complete(c):
    arm_cloud(c)
    for i in range(16):capture(c,i)
    execute_due(*c.s.args,**c.s.kw)


def read(c,download=False):
    return observe(c.registration,c.out,download=download)


def test_no_witness_does_not_lookup_or_create_a_call(cloud):
    assert read(cloud)['status']=='waiting_for_cloud_arming'
    assert not cloud.calls and not cloud.s.calls


@pytest.mark.parametrize('kind',['builtin','modal'])
def test_poll_timeouts_reuse_the_exact_arming_handle(cloud,kind):
    import modal
    c=cloud;arm_cloud(c)
    c.replies['fc-arm']=TimeoutError() if kind=='builtin' else modal.exception.TimeoutError()
    assert read(c)['status']==read(c)['status']=='arming_call_pending'
    assert c.calls==['fc-arm','fc-arm'] and not c.s.calls


@pytest.mark.parametrize('kind',['execution','expired'])
def test_execution_timeout_or_expired_output_is_not_poll_pending(cloud,kind):
    import modal
    c=cloud
    error=modal.exception.FunctionTimeoutError if kind=='execution' else modal.exception.OutputExpiredError
    c.replies['fc-test']=error('specific terminal or expired evidence')
    with pytest.raises(error):asyncio.run(existing_call('fc-test'))
    assert c.calls==['fc-test'] and not c.s.calls


def test_changed_sources_are_rejected_before_call_lookup(cloud):
    c=cloud;arm_cloud(c);c.s.sources['rate_market_cloud.py']='f'*64
    with pytest.raises(ValueError,match='sources differ'):read(c)
    assert not c.calls


def test_completed_receipt_waits_for_return_without_downloading_arrays(cloud):
    c=cloud;arm_cloud(c);capture(c,0);c.replies.pop('fc-0')
    for _ in range(2):assert read(c,True)['status']=='chunk_worker_return_pending'
    assert c.calls==['fc-arm','fc-0','fc-arm','fc-0']
    assert not any('/artifacts/' in p for p in c.reads)


def test_changed_completed_owner_cannot_replace_saved_call(cloud):
    c=cloud;arm_cloud(c);capture(c,0);assert read(c)['completed_chunks']==1
    path=c.s.root/'chunks'/c.s.calls[0]/'receipt.json';r=json.loads(path.read_text())
    r['call_id']='fc-replacement';atomic_json(path,r)
    with pytest.raises(ValueError,match='replaced'):read(c)
    assert 'fc-replacement' not in c.calls


def test_interrupted_capture_is_observed_without_restart(cloud):
    c=cloud;arm_cloud(c)
    def fail(*a,**kw):raise KeyboardInterrupt()
    c.s.kw['run']=fail
    with pytest.raises(KeyboardInterrupt):capture(c,0)
    assert read(c)['status']=='chunk_call_pending'
    c.replies['fc-0']={'status':'paper_rate_halted'}
    assert read(c)['status']=='chunk_receipt_unresolved'
    assert not c.s.calls


def test_wrong_completed_call_result_blocks_artifact_reads(cloud):
    c=cloud;arm_cloud(c);capture(c,0);c.replies['fc-0']['input_id']='in-other'
    with pytest.raises(ValueError,match='owning call differs'):read(c,True)
    assert not any('/artifacts/' in p for p in c.reads)


@pytest.mark.parametrize('damage',['path','time','nan','missing','out_of_order','reused'])
def test_receipt_path_clock_sequence_and_uniqueness_guards(cloud,damage):
    c=cloud;arm_cloud(c);capture(c,0);capture(c,1)
    name=c.s.calls[0];path=c.s.root/'chunks'/name/'receipt.json';r=json.loads(path.read_text())
    if damage=='path':r['remote_path']='/state/unrelated/private'
    if damage=='time':r['claimed_at']=c.s.r['end']-1
    if damage=='nan':r['completed_at']=float('nan')
    if damage=='out_of_order':r['status']='claimed'
    if damage=='missing':path.unlink()
    else:path.write_text(json.dumps(r))
    if damage=='reused':
        path=c.s.root/'chunks'/c.s.calls[1]/'receipt.json';r=json.loads(path.read_text());r['call_id']='fc-0';atomic_json(path,r)
    with pytest.raises(ValueError):read(c)
    assert len(c.s.calls)==2


def test_all_sixteen_returns_and_arrays_are_preserved_and_reused(cloud):
    c=cloud;complete(c)
    status=read(c,True)
    assert status['status']=='downloaded_pending_audit' and status['downloaded_chunks']==status['completed_chunks']==16
    assert status['cloud_submissions']==status['new_neural_observations']==0
    artifact_reads=[p for p in c.reads if p.endswith('.npz')]
    assert len(artifact_reads)==48 and set(c.reads)==set(c.closed)
    again=read(c,True)
    assert again['completed_chunks']==16 and len([p for p in c.reads if p.endswith('.npz')])==48
    assert len(c.s.calls)==16 and not c.audit_out.exists()


def test_capture_can_finish_before_the_aggregate_call(cloud):
    c=cloud;arm_cloud(c)
    for i in range(16):capture(c,i)
    assert read(c)['status']=='captured_waiting_for_cloud_summary'
    assert not (c.out/'cloud-summary.json').exists()


def test_corrupt_download_does_not_become_a_verified_recording(cloud):
    c=cloud;arm_cloud(c);capture(c,0)
    name=c.s.calls[0];(c.s.root/'chunks'/name/'artifacts/trace/step-01.npz').write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='artifact hash differs'):read(c,True)
    assert not (c.out/'chunks'/name/'trace/step-01.npz').exists()
    assert not (c.out/'downloads'/(name+'.json')).exists()


def test_raw_price_reconstruction_precedes_large_neural_downloads(cloud,monkeypatch):
    c=cloud;arm_cloud(c);capture(c,0);seen=[]
    def reject(*args):seen.append(args);raise ValueError('raw price audit rejected')
    monkeypatch.setattr('paperlab.fly_rate_price_audit.audit_prices',reject)
    with pytest.raises(ValueError,match='raw price'):read(c,True)
    assert len(seen)==1 and not any(p.endswith('.npz') for p in c.reads)


def test_metadata_timeout_closes_stream_without_retrying_compute(cloud):
    c=cloud;closed=[];original_timeout=asyncio.timeout
    async def stall(path):
        try:
            yield b'partial';await asyncio.Event().wait()
        finally:closed.append(path)
    c.volume.read_file.aio=stall
    from unittest.mock import patch
    with patch('paperlab.fly_rate_results.asyncio.timeout',lambda seconds:original_timeout(.01)):
        with pytest.raises(TimeoutError):read(c)
    assert len(closed)==1 and not c.calls and not c.s.calls
    error=json.loads((c.out/'observation-error.json').read_text())
    assert error['type']=='TimeoutError' and error['cloud_submissions']==0


def test_second_local_reader_cannot_race_on_partial_downloads(cloud):
    c=cloud;c.out.mkdir()
    with (c.out/'observe.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(ValueError,match='Another reader'):read(c)
    assert not c.reads


@pytest.fixture
def audit_fixture(cloud,monkeypatch):
    """Only orchestration is simulated; no synthetic array is passed as a real audit."""
    c=cloud;complete(c);read(c,True);seen=[]
    from paperlab.fly_rate_price_audit import audit_prices as real_prices
    def prices(env,path):
        result=real_prices(env,path);seen.append('prices');return result
    def audit(env,summary,recording,data):
        assert seen[0]=='prices';seen.append(('audit',summary['chunk']))
        count=sum(row['event'] is not None for row in summary['outcome']['rows'])
        checks={'decision_slots':25,'observations':count,'native_bins':count*50,'ledger_replayed':True,
                'feedback_reconstructed':True,'all_boundaries_verified':True,'all_weights_exact':True}
        return ({'status':'paper_rate_chunk_audited','chunk':summary['chunk'],'verification':checks,'validation_only':True,
                 'plan_sha256':env['sha256'],'artifact_sha256':summary['artifact_sha256'],
                 'executed_source_sha256':summary['code_sha256']},
                {'report':{'run_id':summary['chunk']},'validation_only':True})
    def projection(view,path,graph,observations):
        assert view['validation_only'];seen.append(('projection',view['report']['run_id']))
        return {'observations':observations,'bins':observations*50,'validation_only':True}
    monkeypatch.setattr('paperlab.fly_rate_price_audit.audit_prices',prices)
    monkeypatch.setattr('paperlab.fly_rate_audit.audit_chunk',audit)
    monkeypatch.setattr('paperlab.fly_view_projection.load_graph',lambda data:SimpleNamespace(validation_only=True))
    monkeypatch.setattr('paperlab.fly_view_projection.audit_view',projection)
    monkeypatch.setattr('paperlab.fly_trace.TraceLab',lambda *a,**kw:pytest.fail('Native brain construction forbidden'))
    c.seen=seen;return c


def audit(c):
    return audit_saved(c.out,'synthetic graph fixture',c.audit_out,registration=c.registration)


def test_offline_audit_reconciles_every_recording_and_checks_views(audit_fixture):
    c=audit_fixture;calls=list(c.calls);result=audit(c)
    assert result['status']=='paper_rate_study_audited' and result['audited'] and result['validation_only']
    assert result['verification']['all_view_projections_verified']
    assert len(result['audit_sha256'])==len(result['projection_sha256'])==16
    assert not result['test_coverage_sufficient'] and not result['policy_promoted']
    assert result['new_neural_observations']==result['cloud_submissions']==0 and c.calls==calls
    assert c.seen==['prices']+[item for name in c.s.calls for item in (('audit',name),('projection',name))]
    for name in c.s.calls:
        assert (c.audit_out/'views'/name/'step-01.npz').samefile(c.out/'chunks'/name/'trace/step-01.npz')
    with pytest.raises(ValueError,match='earlier audit'):audit(c)


@pytest.mark.parametrize('damage',['recording','selection','test_receipt','aggregate','native','registration','price','projection','audit_count','audit_identity'])
def test_offline_corruption_never_publishes_a_completed_report(audit_fixture,monkeypatch,damage):
    c=audit_fixture
    if damage=='recording':(c.out/'chunks'/c.s.calls[0]/'trace/step-01.npz').write_bytes(b'changed')
    if damage=='selection':
        path=c.out/'selection.json';v=json.loads(path.read_text());v['selected']=None;atomic_json(path,v)
    if damage=='test_receipt':
        path=c.out/'selection-receipt.json';v=json.loads(path.read_text());v['selected_at']+=1;atomic_json(path,v)
    if damage=='aggregate':
        path=c.out/'cloud-summary.json';v=json.loads(path.read_text());v['test_coverage_sufficient']=True;atomic_json(path,v)
    if damage=='native':
        name=c.s.calls[-1];path=c.out/'chunks'/name/'summary.json';v=json.loads(path.read_text());v['native_build']['binary_sha256']='e'*64;atomic_json(path,v)
    if damage=='registration':
        path=c.registration;v=json.loads(path.read_text());v['recorded_at']-=1;atomic_json(path,v)
    if damage=='price':(c.out/'universe.db').write_bytes(b'changed')
    if damage=='projection':
        monkeypatch.setattr('paperlab.fly_view_projection.audit_view',lambda *a,**kw:(_ for _ in ()).throw(ValueError('projection rejected')))
    if damage in ('audit_count','audit_identity'):
        import paperlab.fly_rate_audit as module
        original=module.audit_chunk
        def wrong(*args):
            checked,view=original(*args)
            if damage=='audit_count':checked['verification']['native_bins']+=1
            else:checked['plan_sha256']='f'*64
            return checked,view
        monkeypatch.setattr(module,'audit_chunk',wrong)
    with pytest.raises(ValueError):audit(c)
    assert not (c.audit_out/'report.json').exists() and len(c.s.calls)==16
