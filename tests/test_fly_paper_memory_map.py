import numpy as np
import pytest

from paperlab.fly_paper_memory_map import recipient_summary, compare_memories


def state(weights):
    weights = np.asarray(weights, dtype=np.float32)
    return {'weights': weights, 'u': np.zeros(len(weights)), 'w': np.zeros(len(weights))}


def test_opposing_changes_do_not_disappear_behind_a_zero_net_sum():
    baseline = np.array([1, 10, 4], dtype=np.float32)
    learned = state([2, 9, 2]);before = {k: v.copy() for k, v in learned.items()}
    rows = recipient_summary(baseline, learned, np.array([7, 7, 11]), {'7': 'MBON07', '11': 'MBON11'})
    assert rows[0]['net_weight_change_pct'] == 0
    assert rows[0]['changed'] == 2 and rows[0]['increased'] == rows[0]['decreased'] == 1
    assert rows[0]['weight_delta_l2'] == pytest.approx(2**.5)
    assert rows[0]['efficacy_quantiles'][0] == pytest.approx(.9)
    assert rows[1]['net_weight_change_pct'] == -50
    assert sum(r['edges'] for r in rows) == 3
    assert all(np.array_equal(before[k], learned[k]) for k in learned)


def test_cross_pool_memory_similarity_handles_reversals_and_unchanged_weights():
    baseline = np.array([2, 2, 2], dtype=np.float32)
    a = state([3, 1, 2]);b = state([1, 3, 2])
    result = compare_memories(baseline, a, b)
    assert result['weight_delta_cosine'] == pytest.approx(-1)
    assert result['changed_in_both'] == 2
    assert result['same_sign_fraction_when_both_changed'] == 0
    zero = compare_memories(baseline, state(baseline), a)
    assert zero['weight_delta_cosine'] is None
    assert zero['same_sign_fraction_when_both_changed'] is None


@pytest.mark.parametrize('problem', ['zero_baseline', 'nonfinite', 'shape', 'annotation'])
def test_recipient_map_rejects_invalid_or_unmapped_memory(problem):
    baseline = np.ones(2, dtype=np.float32);learned = state([1, 2]);labels = {'7': 'MBON07'}
    if problem == 'zero_baseline':baseline[0] = 0
    if problem == 'nonfinite':learned['u'][0] = np.nan
    if problem == 'shape':learned['w'] = np.zeros(3)
    if problem == 'annotation':labels = {}
    with pytest.raises(ValueError):recipient_summary(baseline, learned, np.array([7, 7]), labels)
