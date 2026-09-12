import json
from pathlib import Path

import numpy as np
import pytest

from paperlab.fly_market_memory_audit import audit
from paperlab.fly_market_study import INFERENCE_ARMS, memory_signature, signature


def fixture(root):
    plan=json.loads(Path('reports/fly-market-study-04-plan.json').read_text())['plan']
    plan.update(schema=4,arms=INFERENCE_ARMS);plan.pop('visual_encoding',None)
    envelope={'plan':plan,'sha256':signature(plan)}
    report={'plan_sha256':envelope['sha256'],'phase_diagnostics':{}}
    pristine={'weights':np.ones(2,dtype=np.float32),'u':np.zeros(2,dtype=np.float32),'w':np.zeros(2,dtype=np.float32)}
    for i,pool in enumerate(plan['cohort']):
        report['phase_diagnostics'][pool]={}
        for name,arm in plan['arms'].items():
            state={k:v+(.1 if arm['train'] else 0) for k,v in pristine.items()}
            initial=memory_signature(pristine);trained=memory_signature(state)
            folder=root/f'pool{i}-{name}';folder.mkdir()
            np.savez_compressed(root/f'pool{i}-{name}-training-memory.npz',**state)
            np.savez_compressed(folder/'initial-memory.npz',**state)
            report['phase_diagnostics'][pool][name]={
                'training':{'initial_memory_sha256':initial,'final_memory_sha256':trained},
                'development':{'initial_memory_sha256':trained,'final_memory_sha256':('updated' if arm['online'] else trained)},
                'test':{'initial_memory_sha256':trained,'final_memory_sha256':('updated' if arm['online'] else trained)}}
    return envelope,report


def test_memory_audit_checks_saved_training_and_inference_arrays(tmp_path):
    plan,report=fixture(tmp_path);result=audit(plan,report,tmp_path)
    assert len(result['checkpoint_sha256'])==12
    for pool,arms in result['pools'].items():
        assert arms['pristine_frozen']['training_changed_memory'] is False
        assert arms['trained_frozen']['training_changed_weights']==2
        assert arms['trained_frozen']['training_delta_l2']['u']>0
        assert arms['trained_frozen']['frozen_inference_reported_memory_unchanged'] is True
        assert arms['online_original']['frozen_inference_reported_memory_unchanged'] is None


@pytest.mark.parametrize('mutation',['checkpoint','restored','frozen','initial','plan','invalid_values'])
def test_memory_audit_rejects_changed_or_mislabeled_state(tmp_path,mutation):
    plan,report=fixture(tmp_path);pool=plan['plan']['cohort'][0];rows=report['phase_diagnostics'][pool]['trained_frozen']
    if mutation=='checkpoint':rows['training']['final_memory_sha256']='wrong'
    if mutation=='restored':rows['test']['initial_memory_sha256']='wrong'
    if mutation=='frozen':rows['test']['final_memory_sha256']='wrong'
    if mutation=='initial':rows['training']['initial_memory_sha256']='wrong'
    if mutation=='plan':report['plan_sha256']='wrong'
    if mutation=='invalid_values':np.savez_compressed(tmp_path/'pool0-trained_frozen/initial-memory.npz',weights=np.array([np.nan]),u=np.zeros(1),w=np.zeros(1))
    with pytest.raises(ValueError):audit(plan,report,tmp_path)
