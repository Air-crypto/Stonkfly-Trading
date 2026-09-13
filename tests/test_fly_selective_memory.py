"""Memory-filter algebra checked against numerical ODE integration, no brain."""
import numpy as np
import pytest

from paperlab.fly_selective_memory import decompose, describe, response, validate_report, GROUPS, MODES, COMPONENTS


def integrate(u0, w0, drives):
    # Independently integrate du/dt = -u/1800 + D and dw/dt = (u-w)/.05.
    # RK4 takes 100 substeps per recorded bin. No production filter coefficients.
    state = np.stack((u0, w0)).astype(float); history=[]; h=.0001
    for drive in drives:
        def f(s):return np.stack((-s[0]/1800+drive, (s[0]-s[1])/.05))
        for _ in range(100):
            k1=f(state);k2=f(state+h*k1/2);k3=f(state+h*k2/2);k4=f(state+h*k3)
            state += h*(k1+2*k2+2*k3+k4)/6
        history.append(state[1].copy())
    return np.asarray(history)


def test_piecewise_components_match_independent_ode():
    u=np.array([.2,-.1]);w=np.array([-.05,.15])
    old=np.array([[.2,-.3],[0,0],[-.4,.8],[.5,-.2],[0,.2],[.1,0]])
    new=np.array([[-.1,.3],[.6,-.1],[.1,0],[0,0],[.2,-.2],[0,.3]])
    parts=decompose(u,w,old,new)
    np.testing.assert_allclose(w+sum(parts.values()),integrate(u,w,old+new),atol=2e-11,rtol=1e-10)
    zeros=np.zeros_like(u)
    np.testing.assert_allclose(parts['stored_memory'],integrate(u,w,np.zeros_like(old))-w,atol=2e-11,rtol=1e-10)
    np.testing.assert_allclose(parts['earlier_trace_drive'],integrate(zeros,zeros,old),atol=2e-11,rtol=1e-10)
    np.testing.assert_allclose(parts['current_trace_drive'],integrate(zeros,zeros,new),atol=2e-11,rtol=1e-10)


def test_opposing_drives_cancel_without_removing_stored_memory():
    old=np.array([[1.,-2.],[-3.,4.],[2.,1.]])
    p=decompose([.2,-.1],[.1,.1],old,-old)
    np.testing.assert_array_equal(p['earlier_trace_drive'],-p['current_trace_drive'])
    assert np.any(p['stored_memory'])


def test_frozen_means_no_passive_relaxation_either():
    parts=decompose([.2],[.1],[[1.],[2.]],[[3.],[4.]],frozen=True)
    assert all(not np.any(v) for v in parts.values())


def test_signed_projections_sum_but_component_norms_do_not():
    actual=np.array([3.,4.]);v={'a':np.array([6.,8.]),'b':np.array([-3.,-4.])}
    d=describe(v,actual)
    assert d['observed_change_l2']==5.
    assert [d['components'][k]['signed_projection'] for k in ('a','b')]==[10.,-5.]
    assert sum(c['l2'] for c in d['components'].values())==15.
    assert all(v['signed_projection']==0 for v in describe({'a':np.zeros(2)},np.zeros(2))['components'].values())


@pytest.mark.parametrize('time,tu,tw',[(-1.,1800.,.05),(float('nan'),1800.,.05),(.1,0.,.05),(.1,1.,1.)])
def test_invalid_response_parameters(time,tu,tw):
    with pytest.raises(ValueError):response(time,tu,tw)


@pytest.mark.parametrize('args',[
    ([0.],[0.],[[float('nan')]],[[0.]]),
    ([0.,1.],[0.],[[0.]],[[0.]]),
    ([0.],[0.],[[0.]],[[0.],[1.]]),
])
def test_invalid_component_arrays(args):
    with pytest.raises(ValueError):decompose(*args)


@pytest.mark.parametrize('dt',[0.,-.01,.02,float('nan')])
def test_invalid_bin_duration(dt):
    with pytest.raises(ValueError):decompose([0.],[0.],[[0.]],[[0.]],dt=dt)


def report_fixture():
    import copy
    point={'observed_change_l2':0.,'components':{k:{'l2':0.,'signed_projection':0.} for k in COMPONENTS}}
    return {'status':'selective_memory_components_reconstructed','verification':{
        'observations':36,'bins':1800,'edges_per_bin':7835,
        **{k:True for k in ('all_recorded_drive_integrals_match_audit','all_recorded_memory_filters_reconstructed',
        'all_weight_residuals_within_endpoint_precision','all_final_projection_sums_verified','no_saturated_memory_used')}},
        'arms':{g+'_'+m:[{'observation':i,'series':[copy.deepcopy(point) for _ in range(50)],
                         'final':copy.deepcopy(point)} for i in (1,2,3)] for g in GROUPS for m in MODES}}


def test_figure_rejects_changed_final_value():
    r=report_fixture();validate_report(r)
    r['arms'][GROUPS[0]+'_'+MODES[0]][2]['final']['observed_change_l2']=1.
    with pytest.raises(ValueError,match='inconsistent final'):validate_report(r)


def test_figure_rejects_balanced_but_impossible_projection():
    r=report_fixture();p=r['arms'][GROUPS[0]+'_'+MODES[0]][0]['series'][0]
    p['components']['stored_memory']['signed_projection']=1.
    p['components']['earlier_trace_drive']['signed_projection']=-1.
    with pytest.raises(ValueError,match='exceeds'):validate_report(r)


def test_figure_rejects_missing_condition():
    r=report_fixture();r['arms'].pop(GROUPS[-1]+'_'+MODES[-1])
    with pytest.raises(ValueError,match='complete'):validate_report(r)
