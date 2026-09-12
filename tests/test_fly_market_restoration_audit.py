import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from paperlab.fly_market_restoration_audit import audit,figure


def fixture():
    return (json.loads(Path('reports/fly-market-restoration-study-01.json').read_text()),
            Path('reports/fly-market-study-05.json').read_text(),
            json.loads(Path('reports/fly-market-study-05-plan.json').read_text()))


def test_restoration_audit_distinguishes_recovered_actions_and_spike_counts():
    study,reference,plan=fixture();r=audit(study,reference,plan)
    assert r['both_restoration_matches_pristine_memory']
    assert r['joint_readout_contrast_hz']==[0,-6,0]
    assert r['comparisons']['restore_both']['matching_pristine_spike_counts']==3
    assert r['comparisons']['restore_10704']['matching_pristine_spike_counts']==1
    assert r['comparisons']['restore_11402']['matching_pristine_actions']==3
    ET.fromstring(figure(study))


@pytest.mark.parametrize('mutation',['control','memory','training','input','updates','mask','overlap','config'])
def test_restoration_audit_rejects_broken_controls_or_masks(mutation):
    study,reference,plan=fixture();reports=study['reports'];r=reports['restore_10704']
    if mutation=='control':reports['trained_frozen']['events'][1]['side']='BUY'
    if mutation=='memory':r['final_memory_sha256']='changed'
    if mutation=='training':study['training_events'][1]['spike_sha256']='changed'
    if mutation=='input':r['events'][1]['input_sha256']='changed'
    if mutation=='updates':r['events'][1]['diagnostics']['weight_delta_l2']=1
    if mutation=='mask':r['restoration']['edge_ids'].pop()
    if mutation=='overlap':r['restoration']['edge_ids'][0]=reports['restore_11402']['restoration']['edge_ids'][0]
    if mutation=='config':r['config']['news']='positive'
    with pytest.raises(ValueError):audit(study,reference,plan)


def factorial_fixture(root):
    import copy,hashlib
    import numpy as np
    from paperlab.fly_market_restoration import FACTORIAL_ARMS,MBON07
    from paperlab.fly_market_study import memory_signature
    reference=json.loads(Path('reports/fly-market-study-06.json').read_text());protocol=json.loads(Path('reports/fly-market-restoration-protocol-02.json').read_text());plan=json.loads(Path('reports/fly-market-study-06-plan.json').read_text());rows=reference['phase_diagnostics'][protocol['pool']]
    # A six-edge fixture with one edge per target; real cloud checkpoints remain private.
    posts=np.array([10704,11402,*map(int,MBON07)]);edges=np.arange(6)
    pristine={k:np.ones(6,dtype=np.float32) for k in ['weights','u','w']};trained={k:v+1 for k,v in pristine.items()}
    np.savez_compressed(root/'plastic-map.npz',post_ids=posts,edge_ids=edges)
    np.savez_compressed(root/'pristine-memory.npz',**pristine);np.savez_compressed(root/'trained-memory.npz',**trained)
    rows['trained_frozen']['training']['final_memory_sha256']=memory_signature(trained);reports={}
    for name,arm in FACTORIAL_ARMS.items():
        mask=np.isin(posts.astype(str),arm['restore_post_ids']);base=trained if arm['memory']=='trained' else pristine
        state={k:np.where(mask,pristine[k],base[k]) for k in base};folder=root/name;folder.mkdir();np.savez_compressed(folder/'initial-memory.npz',**state)
        ref=rows[name] if name in rows else rows['pristine_frozen']
        events=copy.deepcopy([d['neural'] for d in ref['test']['decisions'][:3]])
        for e in events:e['diagnostics']={'plasticity_enabled':False,'weight_delta_l2':0}
        reports[name]={'config':{**arm,'preset':'recorded_market','news':'none','eta':.001,'learning':False,'pulses':'none','view':'original'},'native_build':reference['native_build'],'graph':{'plastic_edges':6},'plan_sha256':plan['sha256'],
            'initial_memory_sha256':memory_signature(state),'final_memory_sha256':memory_signature(state),'training_memory_sha256':memory_signature(trained),
            'restoration':{'post_ids':arm['restore_post_ids'],'edge_ids':[str(e) for e in edges[mask]],'edge_count':int(mask.sum())},'events':events}
        if name in rows:rows[name]['test']['initial_memory_sha256']=memory_signature(state)
    text=json.dumps(reference);protocol['reference_report_sha256']=hashlib.sha256(text.encode()).hexdigest()
    for r in reports.values():r['reference_report_sha256']=protocol['reference_report_sha256']
    study={'protocol':protocol,'reports':reports,'training_events':[d['neural'] for d in rows['trained_frozen']['training']['decisions'][:3]]}
    return study,text,plan


def test_factorial_audit_checks_actual_memory_and_conditional_readout_effects(tmp_path):
    study,reference,plan=factorial_fixture(tmp_path);result=audit(study,reference,plan,tmp_path)
    assert result['verification']['all_memory_restorations_audited']
    assert result['verification']['all_five_reference_controls_reproduced']
    assert result['restoring_11402_effect_hz']['keep_MBON07_keep_10704']==[0,-4,2]
    assert len(result['checkpoint_sha256'])==12
    ET.fromstring(figure(study))
    with pytest.raises(ValueError,match='actual checkpoint'):audit(study,reference,plan)


@pytest.mark.parametrize('mutation',['untargeted','targeted','mapping','all_control','reference_control','count'])
def test_factorial_audit_rejects_wrong_restoration_and_lost_controls(tmp_path,mutation):
    import numpy as np
    study,reference,plan=factorial_fixture(tmp_path)
    if mutation in ('untargeted','targeted'):
        path=tmp_path/'restore_11402/initial-memory.npz'
        with np.load(path) as f:state={k:f[k].copy() for k in f.files}
        state['weights'][0 if mutation=='untargeted' else 1]+=1;np.savez_compressed(path,**state)
    if mutation=='mapping':
        path=tmp_path/'plastic-map.npz'
        with np.load(path) as f:state={k:f[k].copy() for k in f.files}
        state['post_ids'][0]=11402;np.savez_compressed(path,**state)
    if mutation=='all_control':study['reports']['restore_all']['events'][1]['spike_sha256']='changed'
    if mutation=='reference_control':study['reports']['restore_both']['events'][1]['side']='SELL'
    if mutation=='count':study['reports']['trained_frozen']['graph']['plastic_edges']=7
    with pytest.raises(ValueError):audit(study,reference,plan,tmp_path)


def test_nine_arm_gate_timing_reconciles_published_recordings():
    from paperlab.fly_pulse_audit import gate_timing
    study=json.loads(Path('reports/fly-market-restoration-study-02.json').read_text())
    views={name:json.loads((Path('examples/fly-debugger')/('marketrestore02-'+name)/'view.json').read_text()) for name in study['protocol']['arms']}
    timing=gate_timing(study,views,study['protocol']['arms'])
    assert len(timing['arms'])==9
    event=timing['arms']['restore_MBON07_10704'][2]
    assert event['gate_spikes']==1 and event['neurons']['10527']==[{'bin':11,'start_ms':1110.,'end_ms':1120.,'spikes':1}]
    assert timing['arms']['restore_MBON07'][2]['gate_spikes']==0
    views.pop('restore_all')
    with pytest.raises(ValueError,match='Missing timing controls'):gate_timing(study,views,study['protocol']['arms'])
