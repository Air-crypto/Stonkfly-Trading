import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from paperlab.cloud_debug import validate_request
from paperlab.core import digest
from paperlab.fly_market_activity import ARMS, apply_boundary, array_hash, cloud_run, validate
from paperlab.fly_market_activity_audit import audit_boundary
from paperlab.fly_market_study import learned_state


def payload():
    return {'protocol':json.loads(Path('reports/fly-market-activity-protocol-01.json').read_text()),
        'reference_json':Path('reports/fly-market-study-06.json').read_text(),
        'market_plan':json.loads(Path('reports/fly-market-study-06-plan.json').read_text())}


def test_activity_request_routes_all_fixed_conditions(monkeypatch,tmp_path):
    from paperlab.cloud_debug import run
    p=payload();validate(p);request={'run_id':'assay-activity-test','activity_plan':p};assert validate_request(request)==p
    seen=[];monkeypatch.setattr('paperlab.fly_market_activity.run',lambda *a:(seen.append(a) or {'ok':True}))
    result=run(request,tmp_path,'graph');assert result['status']=='market_activity_study_completed' and seen[0][2]==tmp_path/'assay-activity-test'
    with pytest.raises(ValueError):validate_request({**request,'config':{}})


@pytest.mark.parametrize('bad',['arms','fields','timing','source','learning','hash','missing','frozen','input'])
def test_invalid_experiments_are_rejected(bad):
    p=payload()
    if bad=='arms':p['protocol']['arms'].pop('trained_full')
    if bad=='fields':p['protocol']['reset_fields']['visual_filters'].append('v')
    if bad=='timing':p['protocol']['reset_timing']='reset after observation'
    if bad=='source':p['protocol']['training_source']='pristine_frozen'
    if bad=='learning':p['protocol']['inference']['learning']=True
    if bad=='hash':p['reference_json']+=' '
    if bad in ('missing','frozen','input'):
        import hashlib
        r=json.loads(p['reference_json']);e=r['phase_diagnostics'][p['protocol']['pool']]['pristine_frozen']['test']['decisions'][1]
        if bad=='missing':e['neural']=None
        if bad=='frozen':e['neural']['plasticity_enabled']=True
        if bad=='input':e['neural']['input_sha256']='different'
        p['reference_json']=json.dumps(r);p['protocol']['reference_report_sha256']=hashlib.sha256(p['reference_json'].encode()).hexdigest()
    with pytest.raises(ValueError):validate(p)


class Brain:
    def __init__(self):
        self.initial={k:np.zeros(4,dtype=np.float32) for k in ('v','luminance','r8_light','adaptation','counts')}
        self.initial.update(memory_u=np.zeros(2),memory_w=np.zeros(2))
        for k,v in self.initial.items():setattr(self,k,v+2)
        self.weight=np.array([1.,2.,3.]);self.circuit={'edges':np.array([0,2])};self.cursor=5000;self.sim_ms=500.;self.total_spikes=12
    def reset(self,keep_memory=False):
        saved=(self.memory_u.copy(),self.memory_w.copy())
        for k,v in self.initial.items():getattr(self,k)[:]=v
        if keep_memory:self.memory_u[:],self.memory_w[:]=saved
        self.cursor=self.total_spikes=0;self.sim_ms=0.


@pytest.mark.parametrize('mode,fields',[('carry',set()),('full',{'v','luminance','r8_light','adaptation','counts'}),('visual_filters',{'luminance','r8_light'}),('adaptation',{'adaptation'})])
def test_only_declared_dynamic_fields_reset_and_memory_is_retained(tmp_path,mode,fields):
    b=Brain();memory=learned_state(b);initial={k:v for k,v in b.initial.items() if k not in ('memory_u','memory_w')};original=b.weight.copy()
    path=tmp_path/'boundary.npz';m=apply_boundary(b,mode,2,path);r=audit_boundary(path,m,initial,memory,mode,2)
    assert set(r['changed_fields'])==fields
    for k in initial:np.testing.assert_array_equal(getattr(b,k),initial[k] if k in fields else initial[k]+2)
    np.testing.assert_array_equal(b.weight,original)
    assert b.sim_ms==(0 if mode=='full' else 500)


def test_no_intervention_before_first_observation(tmp_path):
    b=Brain();b.reset(keep_memory=True);initial={k:v for k,v in b.initial.items() if k not in ('memory_u','memory_w')};memory=learned_state(b)
    p=tmp_path/'first.npz';m=apply_boundary(b,'full',1,p)
    assert m['mode']=='initial' and not m['changed_fields'];audit_boundary(p,m,initial,memory,'full',1)


@pytest.mark.parametrize('key',['after__v','after__luminance','memory_after__u'])
def test_array_audit_detects_wrong_targets_untouched_state_and_memory_even_with_rehashed_file(tmp_path,key):
    b=Brain();memory=learned_state(b);initial={k:v for k,v in b.initial.items() if k not in ('memory_u','memory_w')};p=tmp_path/'boundary.npz';m=apply_boundary(b,'visual_filters',2,p)
    with np.load(p) as a:values={k:a[k].copy() for k in a.files}
    values[key][0]+=7;np.savez_compressed(p,**values);m['artifact_sha256']=digest(p)
    if key.startswith('after__'):m['after_sha256'][key.removeprefix('after__')]=array_hash(values[key])
    with pytest.raises(ValueError):audit_boundary(p,m,initial,memory,'visual_filters',2)


def test_saved_call_is_resumed_without_duplicate_submission(tmp_path,monkeypatch):
    import modal
    calls=[]
    def timeout(**kw):raise TimeoutError()
    call=SimpleNamespace(object_id='fc-activity-test',get=timeout)
    monkeypatch.setattr(modal.Function,'from_name',lambda *a,**kw:SimpleNamespace(spawn=lambda **kw:(calls.append(kw) or call)))
    monkeypatch.setattr(modal.FunctionCall,'from_id',lambda *a:call)
    p=payload();cloud_run(p,tmp_path);cloud_run(p,tmp_path);assert len(calls)==1
    p['protocol']['recorded_at']+=1
    with pytest.raises(ValueError,match='another payload'):cloud_run(p,tmp_path)
    assert len(calls)==1


def test_uncertain_submit_does_not_retry(tmp_path,monkeypatch):
    import modal
    calls=[]
    def lost(**kw):calls.append(kw);raise ConnectionError()
    monkeypatch.setattr(modal.Function,'from_name',lambda *a,**kw:SimpleNamespace(spawn=lost))
    with pytest.raises(ConnectionError):cloud_run(payload(),tmp_path)
    with pytest.raises(RuntimeError,match='Uncertain'):cloud_run(payload(),tmp_path)
    assert len(calls)==1


@pytest.mark.skipif(not __import__('os').environ.get('FLY_TRACE_DATA'),reason='requires prepared full graph')
def test_native_boundary_preserves_graph_and_reproduces_frozen_input(tmp_path):
    from dataclasses import replace
    import os
    from paperlab.fly_trace import TraceLab, Assay, input_frames
    from paperlab.fly_market_activity_audit import DYNAMIC_FIELDS
    lab=TraceLab(Path(os.environ['FLY_TRACE_DATA']));b=lab.brain
    b.weights_frozen=True;lab.fly.controller.s=replace(lab.fly.controller.s,learning=False)
    initial={k:v for k,v in b.initial.items() if k not in ('memory_u','memory_w')};assert set(initial)==DYNAMIC_FIELDS
    rgb=input_frames(Assay(steps=1))[0];expected=lab.fly.controller.observe(rgb,'none');memory=learned_state(b)
    path=tmp_path/'boundary.npz';m=apply_boundary(b,'full',2,path);audit_boundary(path,m,initial,memory,'full',2)
    actual=lab.fly.controller.observe(rgb,'none')
    assert actual['spike_sha256']==expected['spike_sha256'] and actual['memory']==expected['memory']
