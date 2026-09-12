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


def refinement_payload():
    p=payload();p['protocol']=json.loads(Path('reports/fly-market-activity-protocol-02.json').read_text())
    p['activity_reference_json']=Path('reports/fly-market-activity-study-01.json').read_text();return p


def refinement_brain():
    b=Brain();b.ids=np.array([10527,99,10059,555871]);b.initial['g']=np.zeros(4,dtype=np.float32);b.g=np.full(4,3,dtype=np.float32)
    b.initial['last']=np.full(4,-1,dtype=np.int64);b.last=np.full(4,b.cursor-1,dtype=np.int64);return b


def test_refinement_has_four_controls_before_eight_partial_resets():
    from paperlab.fly_market_activity import catalog
    p=refinement_payload();validate(p)
    assert list(catalog(p['protocol']))[:4]==['pristine_carry','trained_carry','pristine_full','trained_full']
    assert len(catalog(p['protocol']))==12
    assert validate_request({'run_id':'assay-refinement-test','activity_plan':p})==p


@pytest.mark.parametrize('bad',['missing_parent','parent_hash','parent_pool','parent_input','parent_controls','gate_ids','fields'])
def test_refinement_rejects_changed_reference_or_scope(bad):
    import hashlib
    p=refinement_payload()
    if bad=='missing_parent':p.pop('activity_reference_json')
    if bad=='parent_hash':p['activity_reference_json']+=' '
    if bad=='gate_ids':p['protocol']['gate_ids']=['10059','10162']
    if bad=='fields':p['protocol']['reset_fields']['voltage'].append('g')
    if bad in ('parent_pool','parent_input','parent_controls'):
        r=json.loads(p['activity_reference_json'])
        if bad=='parent_pool':r['protocol']['pool']='other'
        if bad=='parent_input':r['reports']['trained_full']['events'][1]['input_sha256']='changed'
        if bad=='parent_controls':r['reports'].pop('trained_full')
        p['activity_reference_json']=json.dumps(r);p['protocol']['activity_reference_sha256']=hashlib.sha256(p['activity_reference_json'].encode()).hexdigest()
    with pytest.raises(ValueError):validate(p)


@pytest.mark.parametrize('mode,fields,indices',[('voltage',{'v'},None),('conductance',{'g'},None),('voltage_conductance',{'v','g'},None),('gate_voltage_conductance',{'v','g'},[0,3])])
def test_refinement_resets_only_declared_elements_at_a_materialized_boundary(tmp_path,mode,fields,indices):
    b=refinement_brain();memory=learned_state(b);initial={k:v for k,v in b.initial.items() if k not in ('memory_u','memory_w')};before={k:getattr(b,k).copy() for k in initial}
    path=tmp_path/'boundary.npz';m=apply_boundary(b,mode,2,path);audit_boundary(path,m,initial,memory,mode,2,b.ids)
    for k in initial:
        expected=before[k].copy()
        if k in fields:
            if indices is None:expected[:]=initial[k]
            else:expected[indices]=initial[k][indices]
        np.testing.assert_array_equal(getattr(b,k),expected)
    assert b.sim_ms==500 and b.cursor==5000 and b.total_spikes==12


def test_gate_reset_rejects_unmaterialized_state_and_unknown_identity(tmp_path):
    b=refinement_brain();b.last[1]-=1
    with pytest.raises(ValueError,match='materialized'):apply_boundary(b,'gate_voltage_conductance',2,tmp_path/'bad.npz')
    assert not (tmp_path/'bad.npz').exists()
    b=refinement_brain();b.ids[0]=7
    with pytest.raises(ValueError,match='gate neuron'):apply_boundary(b,'gate_voltage_conductance',2,tmp_path/'bad.npz')


@pytest.mark.parametrize('bad',['non_target_value','target_indices','identity_map'])
def test_gate_array_audit_catches_wrong_neuron_even_with_updated_hashes(tmp_path,bad):
    b=refinement_brain();memory=learned_state(b);initial={k:v for k,v in b.initial.items() if k not in ('memory_u','memory_w')};p=tmp_path/'boundary.npz';m=apply_boundary(b,'gate_voltage_conductance',2,p)
    ids=b.ids.copy()
    if bad=='non_target_value':
        with np.load(p) as a:values={k:a[k].copy() for k in a.files}
        values['after__v'][1]+=7;np.savez_compressed(p,**values);m['artifact_sha256']=digest(p);m['after_sha256']['v']=array_hash(values['after__v'])
    if bad=='target_indices':m['target_indices']=[1,2]
    if bad=='identity_map':ids=ids[::-1]
    with pytest.raises(ValueError):audit_boundary(p,m,initial,memory,'gate_voltage_conductance',2,ids)


@pytest.mark.skipif(not __import__('os').environ.get('FLY_TRACE_DATA'),reason='requires prepared full graph')
def test_native_gate_reset_preserves_every_other_neuron_and_clock(tmp_path):
    from dataclasses import replace
    import os
    from paperlab.fly_trace import TraceLab,Assay,input_frames
    lab=TraceLab(Path(os.environ['FLY_TRACE_DATA']));b=lab.brain;b.weights_frozen=True;lab.fly.controller.s=replace(lab.fly.controller.s,learning=False)
    rgb=input_frames(Assay(steps=1))[0];lab.fly.controller.observe(rgb,'none');memory=learned_state(b)
    initial={k:v for k,v in b.initial.items() if k not in ('memory_u','memory_w')};path=tmp_path/'boundary.npz'
    m=apply_boundary(b,'gate_voltage_conductance',2,path);r=audit_boundary(path,m,initial,memory,'gate_voltage_conductance',2,b.ids)
    assert r['target_ids']==['10527','555871'] and b.sim_ms==500 and b.cursor==5000
    assert set(r['changed_fields'])<= {'v','g'}
    event=lab.fly.controller.observe(rgb,'none');assert event['brain_ms']==1000
    for k,v in memory.items():np.testing.assert_array_equal(learned_state(b)[k],v)


def test_boundary_view_uses_actual_neuron_indices_and_rejects_changed_values(tmp_path):
    from paperlab.fly_market_activity import attach_boundary_state
    from paperlab.fly_market_activity_audit import audit_view_boundaries
    b=refinement_brain();b.reset(keep_memory=True);apply_boundary(b,'gate_voltage_conductance',1,tmp_path/'boundary-01.npz')
    b=refinement_brain();apply_boundary(b,'gate_voltage_conductance',2,tmp_path/'boundary-02.npz')
    v={'nodes':[{'id':str(b.ids[i]),'index':i} for i in (1,3,0)],'frames':[{},{}]}
    attach_boundary_state(v,tmp_path);audit_view_boundaries(v,tmp_path,b.ids)
    assert v['frames'][1]['activity_state']['before_v']==[2,2,2]
    assert v['frames'][1]['activity_state']['after_v']==[2,0,0]
    assert v['frames'][1]['activity_state']['after_g']==[3,0,0]
    changed=copy.deepcopy(v);changed['frames'][1]['activity_state']['after_v'][0]=0
    with pytest.raises(ValueError,match='values differ'):audit_view_boundaries(changed,tmp_path,b.ids)
    changed=copy.deepcopy(v);changed['nodes'][0]['id']='555871'
    with pytest.raises(ValueError,match='identities'):audit_view_boundaries(changed,tmp_path,b.ids)
