"""Registered eta propagation in the pure auditor; no native brain or RPC."""
import numpy as np
import pytest

from paperlab.fly_credit_audit import rule_module
from paperlab.fly_learning_scale_credit import reconstruct
from paperlab.fly_learning_scale import validate
from test_fly_learning_scale_cloud import payload


def fixture(eta, learning=True):
    rule=rule_module();baseline=np.array([2.],dtype=np.float32)
    circuit={'edges':np.array([7]),'pre':np.array([0]),'post':np.array([1]),
             'dan':np.array([1]),'gain':np.array([[.02]],dtype=np.float32)}
    counts=np.zeros((50,2),dtype=np.int32);counts[::3,0]=1;counts[1::4,1]=1
    initial={'kc':np.zeros(1),'dan':np.zeros(1),'u':np.zeros(1),'w':np.zeros(1),'weights':baseline.copy()}
    state={k:v.copy() for k,v in initial.items()};rows={k:[] for k in ('kc','dan','u','w','weights')}
    for row in counts:
        rule.advance(state['kc'],state['dan'],state['u'],state['w'],row[[0]]/.01,row[[1]]/.01,
                     circuit['gain'],.01,eta,learning=learning,frozen=not learning)
        state['weights']=(baseline*(1+state['w'])).astype(np.float32)
        for key in rows:rows[key].append(state[key].copy())
    arrays={k:np.asarray(v) for k,v in rows.items()}
    arrays.update(counts=counts,neuron_ids=np.array([10,20]),ms=np.arange(10,501,10),
                  plastic_edges=circuit['edges'],plastic_pre=circuit['pre'],plastic_post=circuit['post'],initial_weights=baseline)
    return arrays,initial,circuit,baseline,rule


@pytest.mark.parametrize('eta',[.001,.0001])
@pytest.mark.parametrize('learning',[True,False])
def test_reconstructs_both_registered_rates_and_freezing(eta,learning):
    a,state,c,b,rule=fixture(eta,learning)
    view,metrics=reconstruct(a,state,c,b,rule,learning,np.array([0]),eta=eta)
    assert view['eta']==eta and metrics['all_weights_exact']
    np.testing.assert_array_equal(state['weights'],a['weights'][-1])
    assert bool(np.any(view['total']))==learning


def test_lower_eta_recording_cannot_be_audited_as_original_eta():
    a,state,c,b,rule=fixture(.0001)
    with pytest.raises(ValueError,match='does not reproduce|do not reproduce'):
        reconstruct(a,state,c,b,rule,True,np.array([0]),eta=.001)


def test_unregistered_eta_is_rejected():
    a,state,c,b,rule=fixture(.001)
    with pytest.raises(ValueError,match='Unregistered'):
        reconstruct(a,state,c,b,rule,True,np.array([0]),eta=.0005)


def test_eta_scaling_is_linear_only_when_recorded_firing_is_fixed():
    a,*_=fixture(.001);b,*_=fixture(.0001)
    np.testing.assert_allclose(b['u'],a['u']/10,rtol=1e-12,atol=1e-14)
    np.testing.assert_allclose(b['w'],a['w']/10,rtol=1e-12,atol=1e-14)


def test_protocol_only_changes_eta_after_all_controls():
    p,parent,_,_=validate(payload())
    assert len(p['arms'])==5 and p['execution_order'][:3]==p['control_arms']
    for name,a in p['arms'].items():
        assert a['boundary']=='carry'
        assert a['eta']==(.001 if name in p['control_arms'] else .0001)
        if name in p['control_arms']:
            assert {k:v for k,v in a.items() if k!='eta'}==parent['protocol']['arms'][name]
    assert p['limits']['native_observations']==15 and p['limits']['cloud_submissions']==1
