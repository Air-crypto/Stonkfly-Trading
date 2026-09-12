import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from paperlab.core import digest
from paperlab.fly_market_activity import array_hash, dynamic_state
from paperlab.fly_market_activity_audit import DYNAMIC_FIELDS, read_arrays
from paperlab.fly_market_study import learned_state
from paperlab.fly_selective_trace import apply_boundary, compare_recording, preflight, run, validate
from paperlab.fly_selective_trace_audit import audit_boundary
from test_fly_credit_reset import payload as parent_payload


def payload():
    return {'protocol_json':Path('reports/fly-selective-trace-protocol-01.json').read_text(),
        'parent_payload':parent_payload(),
        'parent_audit_json':Path('reports/fly-credit-reset-audit-01.json').read_text()}


def brain(fresh=False):
    initial={k:np.zeros(4,dtype=np.float64) for k in sorted(DYNAMIC_FIELDS)}
    initial['v'][:]=-60
    values={k:v.copy() if fresh else v+2 for k,v in initial.items()}
    return SimpleNamespace(initial=initial,**values,memory_u=np.array([.2,.3]),
        memory_w=np.array([.1,.2]),weight=np.array([1.,2.,3.],dtype=np.float32),
        circuit={'edges':np.array([0,2])},cursor=0 if fresh else 5000,
        sim_ms=0. if fresh else 500.,total_spikes=0 if fresh else 50)


def test_complete_protocol_preserves_controls_then_separates_histories():
    p,a,_,_=validate(payload())
    assert len(p['arms'])==12
    assert list(p['arms'])[:6]==list(a['protocol']['arms'])
    assert [x['boundary'] for x in p['arms'].values()]==['carry']*3+['reset_rates']*3+['reset_kc']*3+['reset_dan']*3
    assert p['limits']['neural_observations']==36


@pytest.mark.parametrize('key',['protocol_json','parent_audit_json','parent_payload'])
def test_reference_changes_are_rejected(key):
    p=payload()
    if key=='parent_payload':p[key]['reference_json']+=' '
    else:p[key]+=' '
    with pytest.raises(ValueError):validate(p)


@pytest.mark.parametrize('mode,targets',[('carry',set()),('reset_rates',{'rate_kc','rate_dan'}),
    ('reset_kc',{'rate_kc'}),('reset_dan',{'rate_dan'})])
@pytest.mark.parametrize('observation',[1,2,3])
def test_selective_boundaries_preserve_unrelated_state_and_first_image(tmp_path,mode,targets,observation):
    b=brain(fresh=observation==1);before=dynamic_state(b);memory=learned_state(b);weights=b.weight.copy()
    m=apply_boundary(b,mode,observation,tmp_path/'b.npz')
    check=audit_boundary(tmp_path/'b.npz',m,b.initial,memory,mode,observation,weights)
    expected=set() if observation==1 else targets
    assert set(check['changed_fields'])==expected
    assert len(check['preserved_fields'])==21-len(expected)
    for k in before:np.testing.assert_array_equal(getattr(b,k),b.initial[k] if k in expected else before[k])
    np.testing.assert_array_equal(b.weight,weights)


@pytest.mark.parametrize('key',['after__rate_kc','after__rate_dan','after__v','memory_after__w'])
def test_independent_audit_rejects_corruption_even_after_rehashing(tmp_path,key):
    b=brain();memory=learned_state(b);path=tmp_path/'b.npz'
    m=apply_boundary(b,'reset_kc',2,path);a=read_arrays(path);a[key][0]+=1
    np.savez_compressed(path,**a);m['artifact_sha256']=digest(path)
    if key.startswith('after__'):m['after_sha256'][key.removeprefix('after__')]=array_hash(a[key])
    with pytest.raises(ValueError):audit_boundary(path,m,b.initial,memory,'reset_kc',2,b.weight)


@pytest.mark.parametrize('change',['clock','targets','weights','missing','dtype'])
def test_independent_audit_checks_more_than_mutated_array_values(tmp_path,change):
    b=brain();memory=learned_state(b);path=tmp_path/'b.npz';weight=b.weight.copy()
    m=apply_boundary(b,'reset_dan',2,path)
    if change=='clock':m['after_clock']['cursor']+=1
    elif change=='targets':m['target_fields']=['rate_kc','rate_dan']
    elif change=='weights':weight[1]+=1  # A nonplastic connection.
    else:
        a=read_arrays(path)
        if change=='missing':del a['before__queue']
        else:a['after__rate_dan']=a['after__rate_dan'].astype(np.float32)
        np.savez_compressed(path,**a);m['artifact_sha256']=digest(path)
    with pytest.raises(ValueError):audit_boundary(path,m,b.initial,memory,'reset_dan',2,weight)


@pytest.mark.parametrize('mode,index',[('unknown',2),('reset_kc',True),('reset_dan',0),('carry',4)])
def test_invalid_request_does_not_mutate_brain(tmp_path,mode,index):
    b=brain();before=dynamic_state(b)
    with pytest.raises(ValueError):apply_boundary(b,mode,index,tmp_path/'b.npz')
    assert not (tmp_path/'b.npz').exists()
    for k,v in before.items():np.testing.assert_array_equal(getattr(b,k),v)


def test_existing_artifact_and_nonfresh_first_observation_do_not_run(tmp_path):
    b=brain();path=tmp_path/'b.npz'
    with pytest.raises(ValueError,match='fresh'):apply_boundary(b,'reset_kc',1,path)
    path.write_bytes(b'preserved')
    with pytest.raises(ValueError,match='overwrite'):apply_boundary(b,'reset_kc',2,path)
    assert path.read_bytes()==b'preserved'


def test_exact_counts_cannot_hide_different_sampled_voltage(tmp_path):
    values={k:np.zeros((50,2)) for k in ('ms','counts','weights','u','w','initial_weights',
        'neuron_ids','plastic_edges','plastic_pre','plastic_post','kc','dan','voltage')}
    np.savez(tmp_path/'a.npz',**values);values['voltage'][0,0]=1;np.savez(tmp_path/'b.npz',**values)
    with pytest.raises(ValueError,match='voltage'):compare_recording(tmp_path/'a.npz',tmp_path/'b.npz')


def test_pending_study_blocks_before_model_construction_or_output(tmp_path,monkeypatch):
    def forbidden(*a,**kw):raise AssertionError('Must not construct the native model')
    monkeypatch.setattr('paperlab.fly_selective_trace.TraceLab',forbidden)
    with pytest.raises(FileNotFoundError):
        run(payload(),tmp_path/'parent','unused',tmp_path/'out',completed_study=tmp_path/'pending')
    assert not (tmp_path/'out').exists()


def test_synthetic_completed_study_cannot_release_execution(tmp_path,monkeypatch):
    from paperlab.fly_selective_trace import require_completed_study
    registration=json.loads(Path('reports/fly-market-study-11-preregistration.json').read_text())
    monkeypatch.setattr('paperlab.fly_online_figure.evidence',lambda root:({'registration':registration},None,None,True))
    with pytest.raises(ValueError,match='actual study'):require_completed_study(tmp_path)


def test_retained_native_boundary_preflight_without_a_model(tmp_path,monkeypatch):
    location=os.environ.get('FLY_CREDIT_RESET_RECORDINGS')
    if not location:pytest.skip('Requires retained cloud arrays; no neural propagation')
    def forbidden(*a,**kw):raise AssertionError('Preflight must not construct a native model')
    monkeypatch.setattr('paperlab.fly_selective_trace.TraceLab',forbidden)
    r=preflight(payload(),Path(location)/'artifacts',tmp_path/'out')
    assert r['parent_artifacts_verified']==84 and len(r['boundary_checks'])==24
    assert r['dynamic_fields']==21 and r['full_weights']==25582938
    assert r['neural_observations']==r['cloud_submissions']==0
    for check in r['boundary_checks']:
        assert check['all_weights_preserved'] and check['memory_preserved'] and check['clock_preserved']
    with pytest.raises(ValueError,match='Preserve'):preflight(payload(),Path(location)/'artifacts',tmp_path/'out')
