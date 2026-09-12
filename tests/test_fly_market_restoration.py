import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from paperlab.cloud_debug import validate_request
from paperlab.fly_market_restoration import ARMS, cloud_run, restore_posts, validate


def payload():
    return {'protocol':json.loads(Path('reports/fly-market-restoration-protocol-01.json').read_text()),
            'reference_json':Path('reports/fly-market-study-05.json').read_text(),
            'market_plan':json.loads(Path('reports/fly-market-study-05-plan.json').read_text())}


def test_registered_restoration_controls_are_bounded_and_routed(monkeypatch,tmp_path):
    from paperlab.cloud_debug import run
    p=payload();protocol,reference,plan=validate(p)
    assert len(ARMS)==5 and protocol['inference']['steps']==3
    request={'run_id':'assay-restoration-test','restoration_plan':p}
    assert validate_request(request)==p
    seen=[]
    def fake(payload,data,output):seen.append((payload,data,output));return {'ok':True}
    monkeypatch.setattr('paperlab.fly_market_restoration.run',fake)
    result=run(request,tmp_path,'data')
    assert result['status']=='market_restoration_study_completed' and seen[0][2]==tmp_path/'assay-restoration-test'
    with pytest.raises(ValueError):validate_request({**request,'config':{}})


@pytest.mark.parametrize('mutation',['arms','learning','reference','plan','source','missing','input','frozen'])
def test_restoration_plan_rejects_changed_controls(mutation):
    p=payload()
    if mutation=='arms':p['protocol']['arms'].pop('restore_both')
    if mutation=='learning':p['protocol']['inference']['learning']=True
    if mutation=='reference':p['reference_json']+=' '
    if mutation=='plan':p['market_plan']['plan']['start']+=300
    if mutation=='source':p['protocol']['training_source']='trained_frozen'
    if mutation in ('missing','input','frozen'):
        r=json.loads(p['reference_json']);row=r['phase_diagnostics'][p['protocol']['pool']]['trained_frozen']['test']['decisions'][1]
        if mutation=='missing':row['neural']=None
        if mutation=='input':row['neural']['input_sha256']='altered'
        if mutation=='frozen':row['neural']['weight_delta_l2']=1
        p['reference_json']=json.dumps(r);p['protocol']['reference_report_sha256']=hashlib.sha256(p['reference_json'].encode()).hexdigest()
    with pytest.raises(ValueError):validate(p)


def test_target_restoration_changes_only_incoming_memory_and_preserves_inputs():
    pristine={k:np.array([1.,2.,3.]) for k in ('weights','u','w')}
    trained={k:v+10 for k,v in pristine.items()};posts=np.array(['10704','11402','10704'])
    result,mask=restore_posts(pristine,trained,posts,['10704'])
    assert mask.tolist()==[True,False,True]
    for k in result:
        np.testing.assert_array_equal(result[k],[1.,12.,3.])
        np.testing.assert_array_equal(trained[k],[11.,12.,13.])
        assert not np.shares_memory(result[k],trained[k])
    both,_=restore_posts(pristine,trained,posts,['10704','11402'])
    assert all(np.array_equal(v,pristine[k]) for k,v in both.items())
    for targets in (['absent'],['10704','10704']):
        with pytest.raises(ValueError):restore_posts(pristine,trained,posts,targets)


def test_restoration_observer_resumes_without_duplicate_compute(tmp_path,monkeypatch):
    import modal
    calls=[]
    class Remote:
        object_id='fc-restoration-test'
        def get(self,timeout=0):raise TimeoutError('still running')
    remote=Remote()
    monkeypatch.setattr(modal.Function,'from_name',lambda *a,**kw:SimpleNamespace(spawn=lambda **kw:(calls.append(kw) or remote)))
    monkeypatch.setattr(modal.FunctionCall,'from_id',lambda *a:remote)
    p=payload();cloud_run(p,tmp_path);cloud_run(p,tmp_path)
    assert len(calls)==1 and 'restoration_plan' in calls[0]['debug']
    p['protocol']['recorded_at']+=1
    with pytest.raises(ValueError,match='another payload'):cloud_run(p,tmp_path)
    assert len(calls)==1


def test_uncertain_restoration_submission_is_not_retried(tmp_path,monkeypatch):
    import modal
    calls=[]
    def spawn(**kw):calls.append(kw);raise ConnectionError('lost response')
    monkeypatch.setattr(modal.Function,'from_name',lambda *a,**kw:SimpleNamespace(spawn=spawn))
    with pytest.raises(ConnectionError):cloud_run(payload(),tmp_path)
    with pytest.raises(RuntimeError,match='Uncertain'):cloud_run(payload(),tmp_path)
    assert len(calls)==1


def factorial_payload():
    return {'protocol':json.loads(Path('reports/fly-market-restoration-protocol-02.json').read_text()),
            'reference_json':Path('reports/fly-market-study-06.json').read_text(),
            'market_plan':json.loads(Path('reports/fly-market-study-06-plan.json').read_text())}


def test_factorial_registration_covers_each_combination_and_reconstructs_images(monkeypatch):
    monkeypatch.syspath_prepend(str(Path("vendor/stonkfly").resolve()))
    from paperlab.fly_market_restoration import FACTORIAL_ARMS,MBON07
    from paperlab.fly_market_pulse import input_sequences
    p=factorial_payload();protocol,reference,plan=validate(p)
    assert len(FACTORIAL_ARMS)==9
    targets=[set(a['restore_post_ids']) for n,a in FACTORIAL_ARMS.items() if n!='pristine_frozen']
    assert len({tuple(sorted(s)) for s in targets})==8
    for group in [set(MBON07),{'10704'},{'11402'}]:
        assert sum(group<=s for s in targets)==4
    sequences=input_sequences(plan,reference,protocol['pool'],protocol['training_source'])
    assert [len(sequences[p]) for p in ('training','test')]==[3,3]
    assert validate_request({'run_id':'assay-factorial-test','restoration_plan':p})==p


@pytest.mark.parametrize('mutation',['omit','target','source','control'])
def test_factorial_registration_rejects_incomplete_factors_and_bad_references(mutation):
    p=factorial_payload()
    if mutation=='omit':p['protocol']['arms'].pop('restore_MBON07_11402')
    if mutation=='target':p['protocol']['arms']['restore_MBON07']['restore_post_ids'].pop()
    if mutation=='source':p['protocol']['training_source']='online_original'
    if mutation=='control':
        r=json.loads(p['reference_json']);row=r['phase_diagnostics'][p['protocol']['pool']]['restore_11402']['test']['decisions'][1]
        row['neural']['plasticity_enabled']=True
        p['reference_json']=json.dumps(r);p['protocol']['reference_report_sha256']=hashlib.sha256(p['reference_json'].encode()).hexdigest()
    with pytest.raises(ValueError):validate(p)
