import copy
import math

import numpy as np
import pytest

from paperlab.fly_credit_audit import reconstruct, rule_module, split_drive


def fixture(learning=True):
    rule = rule_module()
    circuit = {'edges': np.array([2, 9]), 'pre': np.array([0, 1]),
               'post': np.array([1, 0]), 'dan': np.array([2]), 'gain': np.array([[.4, .6]])}
    baseline = np.array([5, 7], dtype=np.float32)
    state = {'weights': baseline.copy(), 'u': np.zeros(2), 'w': np.zeros(2),
             'kc': np.zeros(2), 'dan': np.zeros(1)}
    initial = copy.deepcopy(state)
    counts = np.random.default_rng(81).integers(0, 3, (50, 3), dtype=np.int32)
    # Both separated firing and coincident firing occur; no native propagation.
    arrays = {'neuron_ids': np.array([10, 20, 30]), 'counts': counts,
              'plastic_edges': circuit['edges'].copy(), 'plastic_pre': circuit['pre'].copy(),
              'plastic_post': circuit['post'].copy(), 'initial_weights': baseline.copy(),
              'ms': np.arange(10, 501, 10)}
    recorded = {k: [] for k in ('kc', 'dan', 'u', 'w', 'weights')}
    for row in counts:
        rule.advance(state['kc'], state['dan'], state['u'], state['w'],
            row[circuit['pre']]/.01, row[circuit['dan']]/.01, circuit['gain'],
            .01, .001, learning=learning, frozen=not learning)
        state['weights'] = (baseline*(1+state['w'])).astype(np.float32)
        for key in recorded:
            recorded[key].append(state[key].copy())
    arrays.update({k: np.asarray(v) for k,v in recorded.items()})
    return arrays, initial, circuit, baseline, rule


def test_prior_traces_decay_and_can_oppose_current_trace_contribution():
    p = rule_module().PARAMETERS
    # Constant 10 Hz current rates and traces: total drive is zero, while the
    # initially asymmetric history splits into opposing nonzero components.
    args = [np.array([10.]), np.array([10.]), np.array([20.]), np.array([0.]),
            np.array([10.]), np.array([10.]), np.ones((1, 1)), .2, .001, p]
    previous, current, total = split_drive(*args)
    assert previous[0] == pytest.approx(-.2*math.exp(-.205))
    assert current[0] == pytest.approx(-previous[0])
    assert total[0] == 0


def test_split_respects_independent_trace_constants():
    p = {**rule_module().PARAMETERS, 'trace_dan_seconds': .4}
    previous, current, total = split_drive(np.array([3.]), np.array([4.]),
        np.array([3.]), np.array([4.]), np.array([10.]), np.array([20.]),
        np.ones((1,1)), 0, .001, p)
    assert previous[0] == pytest.approx(.001*(40*math.exp(-.005/.4)-60*math.exp(-.005)))
    assert current[0] != 0  # Unequal time constants do not cancel same-bin terms.
    assert total[0] == pytest.approx(previous[0]+current[0])


@pytest.mark.parametrize('learning', [True, False])
def test_all_bins_reconstruct_and_selected_weights_remain_observed(learning):
    a, state, circuit, baseline, rule = fixture(learning)
    view, summary = reconstruct(a, state, circuit, baseline, rule, learning, np.array([1]))
    assert len(view['total']) == 50
    assert np.max(np.abs(view['earlier'])) == 0
    assert summary['all_weights_exact']
    assert state['kc'].sum() > 0  # Frozen traces still evolve; only memory freezes.
    delta = np.diff(np.r_[baseline[1], a['weights'][:,1]].astype(float))
    np.testing.assert_array_equal(np.asarray(view['weight_step'])[:,0], delta)
    if not learning:
        assert all(value == 0 for value in summary['integrated_absolute_drive'].values())
        assert np.max(np.abs(view['weight_step'])) == 0


@pytest.mark.parametrize('key', ['kc', 'dan', 'u', 'w', 'weights', 'initial_weights',
                                'plastic_edges', 'plastic_pre', 'plastic_post', 'counts'])
def test_rejects_corrupted_recording_even_if_display_selection_omits_it(key):
    a, state, circuit, baseline, rule = fixture()
    a[key].flat[0] += 1
    with pytest.raises(ValueError):
        reconstruct(a, state, circuit, baseline, rule, True, np.array([1]))


def test_rejects_nonfinite_memory():
    a, state, circuit, baseline, rule = fixture()
    a['u'][30,0] = np.nan
    with pytest.raises(ValueError, match='Invalid full recorded array'):
        reconstruct(a, state, circuit, baseline, rule, True, np.array([1]))


def test_boundary_state_is_required_not_fitted_to_recorded_end_values():
    a, state, circuit, baseline, rule = fixture()
    state['kc'][0] = 9
    with pytest.raises(ValueError, match='does not reproduce'):
        reconstruct(a, state, circuit, baseline, rule, True, np.array([1]))
