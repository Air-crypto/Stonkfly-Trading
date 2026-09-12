from dataclasses import replace
import json
import os
from pathlib import Path
import types

import numpy as np
import pytest

from paperlab.core import digest
from paperlab.fly_recipient_isolation import SelectiveCurrent, protocol, validate


def payload():
    from test_fly_paper_stimulation import payload as parent
    raw = Path('reports/fly-paper-stimulation-study-01.json').read_text()
    audit = Path('reports/fly-paper-stimulation-audit-01.json').read_text()
    p = parent()
    return {'protocol':protocol(p,raw,audit,'assay-recipient-'+'a'*32),
            'paper_payload':p,'reference_json':raw,'audit_json':audit}


def test_isolation_retains_inputs_memory_and_matched_controls():
    from paperlab.cloud_debug import validate_request
    p = payload();validate_request({'run_id':'assay-isolation-test','isolation_plan':p})
    arms = list(p['protocol']['arms'].values())
    assert len(arms)==12 and [a['target_set'] for a in arms[:4]]==['both']*4
    assert {tuple(a['target_ids']) for a in arms[4:]}=={('10704',),('11402',)}
    assert all(a['current']==10 for a in arms)


@pytest.mark.parametrize('field',['target','current','learning','control','path'])
def test_reject_undeclared_isolation_conditions(field):
    p = payload();arm=next(iter(p['protocol']['arms'].values()))
    if field=='target':arm['target_ids']=['10527']
    if field=='current':arm['current']=5
    if field=='learning':p['protocol']['inference']['learning']=True
    if field=='control':p['protocol']['arms'].pop(next(iter(p['protocol']['arms'])))
    if field=='path':p['protocol']['reference_run_id']='../../paper-account'
    with pytest.raises(ValueError):validate(p)


def test_rehashed_extra_target_is_rejected_by_current_audit(tmp_path):
    from paperlab.fly_recipient_isolation_audit import audit_current
    class Brain:
        def rgb_step(self,*args,**kwargs):return 0,0
    b=Brain();path=tmp_path/'current.npz'
    with SelectiveCurrent(b,[2],10) as recorder:
        for _ in range(50):b.rgb_step(None,10,stimulation=[(np.array([2]),10)])
        recorder.save(path)
    assert 'rgb_step' not in vars(b)
    with np.load(path,allow_pickle=False) as f:arrays={k:f[k] for k in f.files}
    arm={'target_ids':['11402'],'current':10};lookup={'11402':2}
    event={'stimulation':{'target_ids':['11402'],'current':10,'duration_ms':500,'artifact_sha256':digest(path)}}
    audit_current(event,arrays,path,arm,lookup)
    arrays['target_indices']=np.array([2,3]);arrays['currents']=np.full((50,2),10)
    np.savez_compressed(path,**arrays);event['stimulation']['artifact_sha256']=digest(path)
    with pytest.raises(ValueError,match='different cells'):audit_current(event,arrays,path,arm,lookup)


def test_no_dispatch_until_exact_prior_call_completes(tmp_path,monkeypatch):
    import modal
    from paperlab.fly_paper_stimulation_cloud import cloud_run
    receipt={'status':'completed','call_id':'fc-prior','run_id':'assay-paper-10-fixture'}
    monkeypatch.setattr(modal.Volume,'from_name',lambda *a,**k:types.SimpleNamespace(read_file=lambda *a:[json.dumps(receipt).encode()]))
    def pending(timeout):raise TimeoutError()
    monkeypatch.setattr(modal.FunctionCall,'from_id',lambda *a:types.SimpleNamespace(get=pending))
    monkeypatch.setattr(modal.Function,'from_name',lambda *a,**k:pytest.fail('Must not submit while prior call is pending'))
    with pytest.raises(RuntimeError,match='still pending'):cloud_run(payload(),tmp_path,isolation=True,reference_artifacts=tmp_path)
    assert not (tmp_path/'cloud-call.json').exists()


def test_cloud_entrypoint_blocks_isolation_before_study_ten(tmp_path,monkeypatch):
    from paperlab.cloud_debug import run
    monkeypatch.setattr('paperlab.fly_recipient_isolation.run',lambda *a:pytest.fail('Native run must remain blocked'))
    with pytest.raises(ValueError,match='Study 10 must finish'):
        run({'run_id':'assay-isolation-fixture','isolation_plan':payload()},tmp_path/'fly-debugger',tmp_path/'data')


def test_isolation_timeout_reattaches_once(tmp_path,monkeypatch):
    import modal
    from paperlab.fly_paper_stimulation_cloud import cloud_run
    submitted=[];seen=[];prerequisites=[]
    monkeypatch.setattr('paperlab.fly_recipient_isolation_cloud.prior_comparison_complete',lambda:prerequisites.append(True))
    monkeypatch.setattr('paperlab.fly_recipient_isolation.verify_reference_artifacts',lambda *a:None)
    monkeypatch.setattr(modal.Function,'from_name',lambda *a,**k:types.SimpleNamespace(spawn=lambda **kw:(submitted.append(kw) or types.SimpleNamespace(object_id='fc-isolation'))))
    def pending(timeout):raise TimeoutError()
    monkeypatch.setattr(modal.FunctionCall,'from_id',lambda identity:(seen.append(identity) or types.SimpleNamespace(get=pending)))
    for _ in range(2):cloud_run(payload(),tmp_path,isolation=True,reference_artifacts=tmp_path)
    assert len(submitted)==len(prerequisites)==1 and seen==['fc-isolation','fc-isolation']
    assert 'isolation_plan' in submitted[0]['debug']


def test_control_files_are_verified_before_compute(tmp_path):
    from paperlab.fly_recipient_isolation import verify_reference_artifacts
    hashes={}
    for pool in (0,1):
        for memory in ('pristine','trained'):
            for observation in (1,2,3):
                name=f'pool{pool}-{memory}-current10/step-{observation:02}.npz'
                file=tmp_path/name;file.parent.mkdir(exist_ok=True);file.write_bytes(b'controlled file integrity fixture')
                hashes[name]=digest(file)
    assert len(verify_reference_artifacts({'artifact_sha256':hashes},tmp_path))==12
    file.write_bytes(b'changed')
    with pytest.raises(ValueError,match='control artifact differs'):verify_reference_artifacts({'artifact_sha256':hashes},tmp_path)


@pytest.mark.skipif(not os.environ.get('FLY_TRACE_DATA'),reason='requires prepared full native graph')
def test_native_single_target_capture_preserves_memory(tmp_path):
    from paperlab.fly_trace import TraceLab
    from paperlab.fly_market_study import learned_state,memory_signature,restore_learned
    from paperlab.fly_paper_stimulation import sequences
    lab=TraceLab(Path(os.environ['FLY_TRACE_DATA']));b=lab.brain;state=learned_state(b)
    rgb=next(iter(sequences(payload()['paper_payload']).values()))[0][0];counts=[]
    for name,ids in (('both',['10704','11402']),('only_10704',['10704']),('only_11402',['11402'])):
        restore_learned(b,state);b.weights_frozen=True;lab.fly.controller.s=replace(lab.fly.controller.s,learning=False)
        folder=tmp_path/name;folder.mkdir();indices=[lab.id_index[x] for x in ids]
        with SelectiveCurrent(b,indices,10) as current:lab.capture(rgb,'none',folder,1,indices,10)
        current.save(folder/'current.npz')
        with np.load(folder/'current.npz',allow_pickle=False) as f:
            assert np.array_equal(f['target_indices'],indices)
            assert f['currents'].shape==(50,len(ids)) and np.all(f['currents']==10)
        assert memory_signature(learned_state(b))==memory_signature(state)
        counts.append(b.counts.copy())
    assert not np.array_equal(counts[0],counts[1]) and not np.array_equal(counts[0],counts[2])
