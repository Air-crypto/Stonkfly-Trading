import numpy as np
import pytest

from paperlab.fly_paper_activation import activation


def test_shared_sources_and_recipients_are_not_counted_once_per_edge():
    counts=np.array([3,0,2,7])
    # Source 0 feeds both recipients, while recipient 3 has two incoming edges.
    actual=activation(counts,[0,0,1],[2,3,3],np.array([True,False,True]))
    assert actual=={'source_spikes':3,'active_sources':1,'active_edges':2,
                    'active_changed_edges':1,'recipient_spikes':9,'active_recipients':2}


def test_active_changed_edges_can_have_silent_recipients():
    actual=activation(np.array([5,0,0]),[0,0],[1,2],np.array([True,True]))
    assert actual['active_changed_edges']==2 and actual['recipient_spikes']==0


@pytest.mark.parametrize('problem',['negative_count','fractional_count','index','length','mask'])
def test_reject_invalid_native_counts_and_mapping(problem):
    counts=np.array([1,0,2]);pre=np.array([0,1]);post=np.array([2,2]);changed=np.array([True,False])
    if problem=='negative_count':counts[0]=-1
    if problem=='fractional_count':counts=counts.astype(float)
    if problem=='index':post[0]=3
    if problem=='length':pre=pre[:1]
    if problem=='mask':changed=changed.astype(int)
    with pytest.raises(ValueError):activation(counts,pre,post,changed)
