import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from paperlab.cloud_debug import validate_request
from paperlab.core import digest
from paperlab.fly_credit_reset import compare_trace, validate
from paperlab.fly_credit_reset_cloud import cloud_run
from paperlab.fly_market_activity import apply_boundary, array_hash, dynamic_state
from paperlab.fly_market_activity_audit import audit_boundary
from paperlab.fly_market_study import learned_state


def payload():
    root=Path('reports')
    pulse={'protocol':json.loads((root/'fly-market-pulse-protocol-01.json').read_text()),
        'reference_json':(root/'fly-market-study-04.json').read_text(),
        'market_plan':json.loads((root/'fly-market-study-04-plan.json').read_text())}
    return {'protocol_json':(root/'fly-credit-reset-protocol-01.json').read_text(),
        'pulse_payload_json':json.dumps(pulse,indent=2)+'\n',
        'reference_json':(root/'fly-market-pulse-study-01.json').read_text(),
        'audit_json':(root/'fly-credit-origin-audit-01.json').read_text()}


def test_registered_conditions_route_without_extra_arguments(monkeypatch,tmp_path):
    p=payload();protocol,*_=validate(p)
    assert len(protocol['arms'])==6
    assert [a['boundary'] for a in protocol['arms'].values()]==['carry']*3+['reset_rates']*3
    request={'run_id':'assay-credit-test','credit_plan':p};assert validate_request(request)==p
    from paperlab.cloud_debug import run
    called=[];monkeypatch.setattr('paperlab.fly_credit_reset.run',lambda *args:called.append(args) or {'ok':True})
    result=run(request,tmp_path,'data')
    assert result['status']=='credit_reset_completed'
    assert called[0][1]==tmp_path/protocol['reference_run_id']
    with pytest.raises(ValueError):validate_request({**request,'config':{}})


@pytest.mark.parametrize('key',['protocol_json','pulse_payload_json','reference_json','audit_json'])
def test_every_registered_document_is_pinned(key):
    p=payload();p[key]+=' '
    with pytest.raises(ValueError):validate(p)


def brain():
    b=SimpleNamespace(initial={k:np.zeros(4) for k in ('v','g','counts','rate_kc','rate_dan')},
        memory_u=np.array([.3,.4]),memory_w=np.array([.2,.3]),weight=np.array([1.,2.,3.]),
        circuit={'edges':np.array([0,2])},cursor=5000,sim_ms=500.,total_spikes=80)
    b.initial.update(memory_u=np.zeros(2),memory_w=np.zeros(2))
    for k in ('v','g','counts','rate_kc','rate_dan'):setattr(b,k,np.full(4,2.))
    return b


@pytest.mark.parametrize('mode',['carry','reset_rates'])
def test_rate_boundary_preserves_activity_memory_and_clock(tmp_path,mode):
    b=brain();memory=learned_state(b);initial={k:v for k,v in b.initial.items() if k not in ('memory_u','memory_w')}
    m=apply_boundary(b,mode,2,tmp_path/'boundary.npz')
    audit_boundary(tmp_path/'boundary.npz',m,initial,memory,mode,2)
    assert set(m['changed_fields'])==({'rate_kc','rate_dan'} if mode=='reset_rates' else set())
    np.testing.assert_array_equal(b.v,np.full(4,2.));np.testing.assert_array_equal(b.g,np.full(4,2.))
    assert b.cursor==5000 and b.sim_ms==500 and b.total_spikes==80


@pytest.mark.parametrize('key',['after__rate_kc','after__v','memory_after__w'])
def test_boundary_tampering_is_detected_even_with_updated_hashes(tmp_path,key):
    b=brain();initial={k:v for k,v in b.initial.items() if k not in ('memory_u','memory_w')};memory=learned_state(b)
    path=tmp_path/'b.npz';m=apply_boundary(b,'reset_rates',2,path)
    with np.load(path) as a:values={k:a[k].copy() for k in a.files}
    values[key][0]+=1;np.savez_compressed(path,**values);m['artifact_sha256']=digest(path)
    if key.startswith('after__'):m['after_sha256'][key.removeprefix('after__')]=array_hash(values[key])
    with pytest.raises(ValueError):audit_boundary(path,m,initial,memory,'reset_rates',2)


def cloud_fake(monkeypatch,*,owner=None,busy=False,lost=False):
    import modal
    calls=[]
    def submit(**kwargs):
        calls.append(kwargs)
        if lost:raise ConnectionError('Uncertain transport outcome')
        return SimpleNamespace(object_id='fc-credit-once')
    def timeout(**kwargs):raise TimeoutError()
    fn=SimpleNamespace(spawn=submit,get_current_stats=lambda:SimpleNamespace(num_total_runners=0,backlog=0,num_running_inputs=int(busy)))
    monkeypatch.setattr(modal.Function,'from_name',lambda *a,**kw:fn)
    monkeypatch.setattr(modal.FunctionCall,'from_id',lambda *a,**kw:SimpleNamespace(get=timeout))
    monkeypatch.setattr(modal.Dict,'from_name',lambda *a,**kw:SimpleNamespace(get=lambda *a:owner))
    monkeypatch.setattr('paperlab.fly_credit_reset_cloud.verify_reference',lambda *a:None)
    return calls


def test_timeout_reattaches_exact_saved_call(tmp_path,monkeypatch):
    calls=cloud_fake(monkeypatch)
    for _ in range(2):cloud_run(payload(),tmp_path,'reference','data')
    assert len(calls)==1
    assert json.loads((tmp_path/'cloud-call.json').read_text())['call_id']=='fc-credit-once'


def test_uncertain_submission_never_retries(tmp_path,monkeypatch):
    calls=cloud_fake(monkeypatch,lost=True)
    with pytest.raises(ConnectionError):cloud_run(payload(),tmp_path,'ref','data')
    with pytest.raises(RuntimeError,match='Uncertain'):cloud_run(payload(),tmp_path,'ref','data')
    assert len(calls)==1


@pytest.mark.parametrize('owner,busy',[(None,True),({'call_id':'fc-live'},False)])
def test_busy_worker_creates_no_call_or_receipt(tmp_path,monkeypatch,owner,busy):
    calls=cloud_fake(monkeypatch,owner=owner,busy=busy)
    with pytest.raises(RuntimeError,match='active work'):cloud_run(payload(),tmp_path,'ref','data')
    assert not calls and not (tmp_path/'cloud-call.json').exists()


def test_full_bin_control_cannot_hide_retimed_spikes_in_equal_totals(tmp_path):
    values={k:np.zeros((50,2)) for k in ('ms','counts','weights','u','w','initial_weights',
        'neuron_ids','plastic_edges','plastic_pre','plastic_post','kc','dan')}
    values['counts'][0,0]=1;np.savez(tmp_path/'a.npz',**values)
    values['counts'][0,0]=0;values['counts'][1,0]=1;np.savez(tmp_path/'b.npz',**values)
    with pytest.raises(ValueError,match='counts'):compare_trace(tmp_path/'a.npz',tmp_path/'b.npz')


@pytest.mark.skipif(not os.environ.get('FLY_TRACE_DATA'),reason='requires full prepared graph')
def test_native_frozen_rate_reset_changes_only_learning_history(tmp_path):
    from dataclasses import replace
    from paperlab.fly_trace import TraceLab,Assay,input_frames
    lab=TraceLab(os.environ['FLY_TRACE_DATA']);b=lab.brain
    images=input_frames(Assay(steps=3));results=[]
    for mode in ('carry','reset_rates'):
        b.reset();b.weights_frozen=True;lab.fly.controller.s=replace(lab.fly.controller.s,learning=False)
        counts=[];rates=[]
        for i,rgb in enumerate(images,1):
            metadata=apply_boundary(b,mode,i,tmp_path/f'{mode}-{i}.npz')
            event=lab.fly.controller.observe(rgb,'reward' if i>1 else 'none')
            counts.append(event['spike_sha256']);rates.append(b.rate_dan.copy())
        results.append((counts,rates))
    assert results[0][0]==results[1][0]
    assert not np.array_equal(results[0][1][-1],results[1][1][-1])
