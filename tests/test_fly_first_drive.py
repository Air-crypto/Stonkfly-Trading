import math
import os
from pathlib import Path

import numpy as np
import pytest

from paperlab.fly_credit_audit import rule_module
from paperlab.fly_first_drive import inspect, substitutions, terms


def test_silent_kc_with_carried_trace_has_negative_dan_term():
    p=rule_module().PARAMETERS;r=terms(0,[100,0],2,[3,4],[.25,.75],p)
    assert np.array_equal(r['current_kc_times_dan_trace'],[0,0])
    assert r['total']==pytest.approx(-.001*100*.25*2*math.exp(-.005))
    cleared=terms(0,[100,0],0,[3,4],[.25,.75],p)
    assert cleared['total']==0


def test_zero_histories_and_equal_trace_constants_cancel_same_bin_products():
    r=terms(100,[100,200],0,[0,0],[.25,.75],rule_module().PARAMETERS)
    assert abs(r['total'])<1e-14
    assert np.all(r['current_kc_times_dan_trace']>0)
    assert np.all(r['current_dan_times_kc_trace']<0)


@pytest.mark.parametrize('args',[(0,[100],2,[3,4],[.25,.75]),
                                (0,[[100]],2,[[3]],[[.25]]),
                                (0,[float('nan')],2,[3],[.25]),
                                (float('inf'),[100],2,[3],[.25])])
def test_invalid_driver_alignment_or_nonfinite_values_are_rejected(args):
    with pytest.raises(ValueError,match='finite aligned'):terms(*args,rule_module().PARAMETERS)


def test_substitutions_require_same_starting_connection_memory():
    carry={k:np.ones(1) for k in ('kc','dan','u','w','weights')}
    reset={k:a.copy() for k,a in carry.items()};reset['w'][0]=0
    with pytest.raises(ValueError,match='identical connection memory'):
        substitutions(carry,reset,None,None,None,None)


def test_retained_first_drives_and_one_bin_substitutions():
    value=os.environ.get('FLY_CREDIT_RESET_RECORDINGS')
    if not value:pytest.skip('Set FLY_CREDIT_RESET_RECORDINGS to the retained cloud audit/artifacts directory; no model run')
    root=Path(value);r=inspect(root/'audit.json',root/'artifacts')
    assert r['neural_observations']==0
    assert [p['end_ms_in_observation'] for p in r['pairs']]==[10,40]
    for p in r['pairs']:
        assert p['starting_connection_memory_identical'] and p['current_counts_identical_through_this_bin']
        assert len(p['entries'])==5
        for entry in p['entries']:
            assert entry['carry']['source_spikes']==entry['reset']['source_spikes']==0
            assert entry['carry']['drive_u_per_second']<0 and entry['reset']['drive_u_per_second']==0
            assert {d['id'] for d in entry['carry']['drivers']}=={'11327','11900'}
            assert all(d['type']=='PPL101' for d in entry['carry']['drivers'])
        for row in p['substitutions']:
            assert row['weight_count']==7835
            assert row['weights_equal_recorded_reset']==(row['name'] in ('reset_kc_history_only','reset_both_histories'))
            assert row['weights_equal_recorded_carry']==(row['name'] in ('carry','reset_dan_history_only'))
    first=r['pairs'][0]['entries'][0]['carry']
    assert first['kc_mid_trace_hz']==pytest.approx(5.264478727569794)
    assert first['drive_u_per_second']==pytest.approx(-.5264478727569795)
