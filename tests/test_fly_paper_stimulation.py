import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import types

import numpy as np
import pytest

from paperlab.fly_paper_stimulation import AppliedCurrent, protocol, sequences, validate


def payload():
    raw=Path('reports/fly-market-study-09.json').read_text()
    env=json.loads(Path('reports/fly-market-study-09-plan.json').read_text())
    return {'protocol':protocol(raw,env),'reference_json':raw,'paper_plan':env}


def test_pinned_inputs_reconstruct_all_original_images_and_complete_controls():
    from paperlab.cloud_debug import validate_request
    p=payload();validate_request({'run_id':'assay-recipient-test','stimulation_plan':p})
    seq=sequences(p)
    assert [len(rows) for rows in seq.values()]==[3,3]
    arms=p['protocol']['arms'];assert len(arms)==12
    assert [a['current'] for a in list(arms.values())[:4]]==[0]*4
    assert sum(3 for _ in arms)==36


@pytest.mark.parametrize('change',['targets','current','learning','missing_arm','new_quote'])
def test_protocol_rejects_undeclared_changes(change):
    p=payload()
    if change=='targets':p['protocol']['target_ids']=['10527','555871']
    if change=='current':p['protocol']['currents']=[0,5,20]
    if change=='learning':p['protocol']['inference']['learning']=True
    if change=='missing_arm':p['protocol']['arms'].pop(next(iter(p['protocol']['arms'])))
    if change=='new_quote':p['paper_plan']['plan']['series'][next(iter(p['paper_plan']['plan']['series']))][-1]['bid']*=2
    with pytest.raises(ValueError):validate(p)


def test_current_recorder_rejects_extra_targets_and_duration(tmp_path):
    class Brain:
        def rgb_step(self,*args,**kwargs):return 0,0
    b=Brain()
    with AppliedCurrent(b,[2,3],5) as trace:
        with pytest.raises(ValueError,match='targets'):b.rgb_step(None,10,stimulation=[(np.array([2,4]),5)])
        b.rgb_step(None,10,stimulation=[(np.array([2,3]),5)])
        with pytest.raises(ValueError,match='500 ms'):trace.save(tmp_path/'current.npz')
    assert 'rgb_step' not in vars(b)
    with AppliedCurrent(b,[2,3],0):
        with pytest.raises(ValueError,match='Zero-current'):b.rgb_step(None,10,stimulation=[(np.array([2,3]),1)])
    assert 'rgb_step' not in vars(b)


def test_observation_timeout_reattaches_without_resubmission(tmp_path,monkeypatch):
    import modal
    from paperlab.fly_paper_stimulation_cloud import cloud_run
    calls=[]; observed=[]
    def spawn(**kwargs):calls.append(kwargs);return types.SimpleNamespace(object_id='fc-same-call')
    def get(timeout):raise TimeoutError('still running')
    def existing(identity):observed.append(identity);return types.SimpleNamespace(get=get)
    monkeypatch.setattr(modal.Function,'from_name',lambda *a,**k:types.SimpleNamespace(spawn=spawn))
    monkeypatch.setattr(modal.FunctionCall,'from_id',existing)
    cloud_run(payload(),tmp_path);cloud_run(payload(),tmp_path)
    assert len(calls)==1 and observed==['fc-same-call','fc-same-call']
    assert json.loads((tmp_path/'cloud-call.json').read_text())['status']=='pending'


@pytest.mark.skipif(not os.environ.get('FLY_TRACE_DATA'),reason='requires prepared full native graph')
def test_native_zero_current_is_transparent_and_current_reaches_targets(tmp_path):
    from paperlab.fly_trace import TraceLab
    from paperlab.fly_market_study import learned_state,memory_signature,restore_learned
    from paperlab.fly_market_restoration import matches
    lab=TraceLab(Path(os.environ['FLY_TRACE_DATA']));b=lab.brain
    b.weights_frozen=True;lab.fly.controller.s=replace(lab.fly.controller.s,learning=False)
    state=learned_state(b);rgb=next(iter(sequences(payload()).values()))[0][0]
    indices=[lab.id_index[x] for x in ('10704','11402')];events=[];counts=[]
    for name,current in (('plain',None),('zero',0),('stimulated',10)):
        restore_learned(b,state);b.weights_frozen=True;folder=tmp_path/name;folder.mkdir()
        if current is None:e=lab.capture(rgb,'none',folder,1)
        else:
            with AppliedCurrent(b,indices,current) as trace:e=lab.capture(rgb,'none',folder,1,indices,current)
            trace.save(folder/'current.npz')
        assert memory_signature(learned_state(b))==memory_signature(state)
        events.append(e);counts.append(b.counts.copy())
    assert matches(events[0],events[1]) and np.array_equal(counts[0],counts[1])
    assert not np.array_equal(counts[1],counts[2])
    assert np.any(counts[2][indices]>counts[1][indices])


@pytest.mark.skipif(not os.environ.get('FLY_STIMULATION_EVIDENCE'),reason='requires captured stimulation evidence')
@pytest.mark.parametrize('corruption',['current_target','frozen_weight'])
def test_auditor_rejects_corrupted_native_evidence(monkeypatch,corruption):
    import paperlab.fly_paper_stimulation_audit as module
    root=Path(os.environ['FLY_STIMULATION_EVIDENCE'])
    study=json.loads((root/'cloud-result.json').read_text())['report']
    p=json.loads((root/'payload.json').read_text());original=module.read_arrays
    def corrupted(path):
        a=original(path)
        if str(path).endswith('pool0-pristine-current0/current-01.npz') and corruption=='current_target':
            a['target_indices'][0]+=1
        if str(path).endswith('pool0-pristine-current0/step-01.npz') and corruption=='frozen_weight':
            a['weights'][17,0]+=1
        return a
    monkeypatch.setattr(module,'read_arrays',corrupted)
    with pytest.raises(ValueError,match='targeted different|Plastic memory changed'):
        module.audit(study,p,root/'artifacts')
