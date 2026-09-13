import numpy as np
import pytest

from paperlab.fly_rate_checkpoint_audit import verify_weights


@pytest.mark.parametrize('corrupt',[None,'export','nonplastic','efficacy','bounds'])
def test_full_checkpoint_and_export_must_reconcile(corrupt):
    expected=np.array([1.,2.,3.,4.],dtype=np.float32);edges=np.array([1,3])
    baseline=expected[edges].copy();memory={'u':np.array([.2,-.3]),'w':np.array([.3,-.1])}
    memory['weights']=(baseline*(1+memory['w'])).astype(np.float32)
    actual=expected.copy();actual[edges]=memory['weights']
    if corrupt=='export':memory['weights'][0]+=.1
    if corrupt=='nonplastic':actual[0]+=.1
    if corrupt=='efficacy':memory['w'][0]+=.1
    if corrupt=='bounds':memory['u'][0]=1.1
    if corrupt:
        with pytest.raises(ValueError):verify_weights(actual,expected,edges,baseline,memory)
    else:
        checked=verify_weights(actual,expected,edges,baseline,memory)
        assert checked['plastic_edges']==checked['nonplastic_edges']==checked['changed_weights']==2
