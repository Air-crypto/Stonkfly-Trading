import numpy as np
import pytest

from paperlab.fly_stimulation_timing import count_difference


def test_timing_difference_survives_identical_observation_totals():
    a = np.zeros((50, 2), dtype=np.int32);b = a.copy()
    a[3, 0] = 1;b[4, 0] = 1
    result = count_difference(a, b)
    assert result['observation_totals_equal']
    assert result['first_difference'] == {'bin_index': 3, 'start_ms': 30, 'end_ms': 40}
    assert sum(result['changed_neurons_per_bin']) == 2


def test_equal_bins_have_no_invented_divergence():
    a = np.ones((50, 4), dtype=np.int32)
    assert count_difference(a, a)['first_difference'] is None


@pytest.mark.parametrize('bad', [np.zeros((49, 2), dtype=int), np.full((50, 2), -1), np.zeros((50, 2))])
def test_reject_non_native_or_misaligned_counts(bad):
    with pytest.raises(ValueError):count_difference(np.zeros((50, 2), dtype=int), bad)
