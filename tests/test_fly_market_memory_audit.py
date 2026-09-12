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


def restoration_fixture(root):
    from paperlab.fly_market_study import RESTORATION_ARMS,RESTORATION_TIMING
    plan=json.loads(Path('reports/fly-market-study-05-plan.json').read_text())['plan']
    plan.update(schema=5,arms=RESTORATION_ARMS,restoration_timing=RESTORATION_TIMING)
    envelope={'plan':plan,'sha256':signature(plan)};report={'plan_sha256':envelope['sha256'],'phase_diagnostics':{}}
    posts=np.array([10704,11402,12859]);edges=np.array([11,12,13]);np.savez_compressed(root/'plastic-map.npz',post_ids=posts,edge_ids=edges)
    pristine={'weights':np.ones(3,dtype=np.float32),'u':np.zeros(3,dtype=np.float32),'w':np.zeros(3,dtype=np.float32)}
    for i,pool in enumerate(plan['cohort']):
        report['phase_diagnostics'][pool]={}
        for name,arm in plan['arms'].items():
            trained={k:v+(.1 if arm['train'] else 0) for k,v in pristine.items()};targets=arm.get('restore_post_ids',[])
            mask=np.isin(posts.astype(str),targets);initial={k:np.where(mask,pristine[k],v) for k,v in trained.items()}
            folder=root/f'pool{i}-{name}';folder.mkdir()
            np.savez_compressed(root/f'pool{i}-{name}-training-memory.npz',**trained)
            np.savez_compressed(root/f'pool{i}-{name}-inference-memory.npz',**initial)
            np.savez_compressed(folder/'initial-memory.npz',**initial)
            train_hash=memory_signature(trained);initial_hash=memory_signature(initial)
            restoration={'post_ids':targets,'edge_ids':[str(e) for e in edges[mask]],'edge_count':int(mask.sum()),'training_memory_sha256':train_hash,'inference_memory_sha256':initial_hash}
            report['phase_diagnostics'][pool][name]={
                'training':{'initial_memory_sha256':memory_signature(pristine),'final_memory_sha256':train_hash,
                            'training_compute_source':'trained_frozen' if arm['train'] else 'pristine_frozen',
                            'training_compute_reused':name not in ('pristine_frozen','trained_frozen')},
                'development':{'initial_memory_sha256':initial_hash,'final_memory_sha256':initial_hash,'restoration':dict(restoration)},
                'test':{'initial_memory_sha256':initial_hash,'final_memory_sha256':initial_hash,'restoration':dict(restoration)}}
    return envelope,report


def test_restoration_memory_audit_checks_every_untargeted_value(tmp_path):
    plan,report=restoration_fixture(tmp_path);result=audit(plan,report,tmp_path)
    for arms in result['pools'].values():
        assert arms['restore_10704']['restored_edge_count']==1
        assert arms['restore_both']['restored_edge_count']==2
        assert arms['restore_10704']['test_checkpoint_matches_training'] is False
        assert arms['restore_10704']['test_checkpoint_matches_inference'] is True
        assert arms['trained_frozen']['test_checkpoint_matches_training'] is True


@pytest.mark.parametrize('mutation',['untargeted','targeted','test_start','development','mapping','metadata','cache_source'])
def test_restoration_audit_rejects_accidental_resets_and_leakage(tmp_path,mutation):
    plan,report=restoration_fixture(tmp_path);pool=plan['plan']['cohort'][0];rows=report['phase_diagnostics'][pool]['restore_10704']
    if mutation in ('untargeted','targeted','test_start'):
        path=tmp_path/('pool0-restore_10704/initial-memory.npz' if mutation=='test_start' else 'pool0-restore_10704-inference-memory.npz')
        with np.load(path) as f:state={k:f[k].copy() for k in f.files}
        state['u'][0 if mutation=='targeted' else -1]+=.3;np.savez_compressed(path,**state)
    if mutation=='development':rows['development']['initial_memory_sha256']=rows['training']['final_memory_sha256']
    if mutation=='mapping':np.savez_compressed(tmp_path/'plastic-map.npz',post_ids=np.array([11402,10704,12859]),edge_ids=np.array([11,12,13]))
    if mutation=='metadata':rows['test']['restoration']['edge_count']=999
    if mutation=='cache_source':rows['training']['training_compute_source']='pristine_frozen'
    with pytest.raises(ValueError):audit(plan,report,tmp_path)
