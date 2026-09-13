"""Reject mismatched evidence before publishing a study 12 diagnostic."""
from copy import deepcopy

import pytest

from paperlab.fly_market_study import signature
from paperlab.fly_rate_report import validate_chunk


def evidence():
    name = 'test-pool0-trained_online_low_eta'
    event = {'side':'BUY', 'gate_spikes':1, 'weight_delta_l2':.2, 'equity_reward_usd':-.5}
    summary = {'chunk':name, 'plan_sha256':'plan', 'artifact_sha256':{'trace/step-01.npz':'array'},
               'code_sha256':{'rule.py':'source'}, 'outcome':{'rows':[{'event':event}]+[{'event':None}]*24}}
    audited = {'chunk':name, 'plan_sha256':'plan', 'status':'paper_rate_chunk_audited',
        'artifact_sha256':deepcopy(summary['artifact_sha256']), 'executed_source_sha256':deepcopy(summary['code_sha256']),
        'verification':{'decision_slots':25, 'observations':1, 'native_bins':50, 'ledger_replayed':True,
            'feedback_reconstructed':True, 'all_boundaries_verified':True, 'all_weights_exact':True},
        'observations':[{'side':'BUY', 'gate_spikes':1, 'weight_norm_audit':{'reported':.2}, 'equity_reward_usd':-.5}]}
    report = {'plan_sha256':'plan', 'chunk_sha256':{name:signature(summary)}}
    receipt = {'chunk':name, 'status':'completed'}
    return report, summary, audited, receipt, name


def test_matching_audited_observation_is_accepted():
    validate_chunk(*evidence())


@pytest.mark.parametrize('damage', ['changed_summary', 'different_plan', 'different_source', 'different_array',
    'partial_bins', 'missing_observation', 'wrong_gate', 'wrong_update', 'wrong_reward', 'incomplete_weights', 'pending_call'])
def test_corrupt_or_partial_evidence_is_rejected(damage):
    report, summary, audited, receipt, name = evidence()
    if damage == 'changed_summary': summary['outcome']['rows'][0]['event']['side'] = 'SELL'
    elif damage == 'different_plan': audited['plan_sha256'] = 'other'
    elif damage == 'different_source': audited['executed_source_sha256']['rule.py'] = 'other'
    elif damage == 'different_array': audited['artifact_sha256']['trace/step-01.npz'] = 'other'
    elif damage == 'partial_bins': audited['verification']['native_bins'] = 49
    elif damage == 'missing_observation': audited['observations'] = []
    elif damage == 'wrong_gate': audited['observations'][0]['gate_spikes'] = 0
    elif damage == 'wrong_update': audited['observations'][0]['weight_norm_audit']['reported'] = .3
    elif damage == 'wrong_reward': audited['observations'][0]['equity_reward_usd'] = .5
    elif damage == 'incomplete_weights': audited['verification']['all_weights_exact'] = False
    elif damage == 'pending_call': receipt['status'] = 'running'
    with pytest.raises(ValueError): validate_chunk(report, summary, audited, receipt, name)


@pytest.fixture
def raw_case(tmp_path):
    import numpy as np
    from paperlab.core import atomic_json, digest
    root = tmp_path/'root'; output = tmp_path/'output'; output.mkdir()
    report = {'validation_only':True, 'chunks':{}}
    for low, arm in enumerate(('trained_online_carry','trained_online_low_eta')):
        name = 'development-pool0-'+arm; folder = root/'chunks'/name; (folder/'trace').mkdir(parents=True)
        np.savez(folder/'neuron-ids.npz', neuron_ids=np.array([10162,10059,10527,555871]))
        np.savez(folder/'circuit.npz', edges=np.array([8022240]))
        counts = np.zeros((50,4),dtype=int)
        if low: counts[0] = [1,2,1,0]
        memory = np.ones((50,1))
        np.savez(folder/'trace/step-01.npz', counts=counts, weights=memory, u=memory, w=memory)
        event = {'market_decision_ts':100, 'input_sha256':'image', 'stimulus':'none', 'stimulus_ms':0, 'equity_reward_usd':0}
        summary = {'outcome':{'rows':[{'event':event}]}, 'artifact_sha256':{
            file:digest(folder/file) for file in ('neuron-ids.npz','circuit.npz','trace/step-01.npz')}}
        atomic_json(folder/'summary.json',summary)
        action = {'decision_ts':100, 'decision':'BUY' if low else 'HOLD', 'left_hz':2*low, 'right_hz':4*low, 'gate_spikes':low}
        report['chunks'][name] = {'summary_sha256':digest(folder/'summary.json'), 'trace':{'decisions':[action]}}
    atomic_json(output/'report.json',report)
    return root, report, output


def test_case_reads_full_arrays_and_keeps_synthetic_label(raw_case):
    pytest.importorskip('matplotlib')
    from paperlab.fly_rate_report import export_case
    case = export_case(*raw_case)
    assert case['validation_only'] is True and case['observation']==0
    assert case['runs']['trained_online_low_eta']['counts']['10527'][0]==1
    assert case['matched_input_and_feedback_prefix'][0]['input_sha256']=='image'
    assert (raw_case[2]/'case.png').is_file()


@pytest.mark.parametrize('damage', ['raw_bytes','decoder','input_history','no_difference'])
def test_case_rejects_mismatched_raw_or_paired_evidence(raw_case, damage):
    import json
    from paperlab.core import atomic_json, digest
    from paperlab.fly_rate_report import export_case
    root,report,output = raw_case; name='development-pool0-trained_online_low_eta'
    folder=root/'chunks'/name
    if damage=='raw_bytes': (folder/'trace/step-01.npz').write_bytes(b'changed')
    elif damage=='decoder': report['chunks'][name]['trace']['decisions'][0]['gate_spikes']=2
    elif damage=='no_difference': report['chunks'][name]['trace']['decisions'][0]['decision']='HOLD'
    else:
        summary=json.loads((folder/'summary.json').read_text())
        summary['outcome']['rows'][0]['event']['input_sha256']='different image'
        atomic_json(folder/'summary.json',summary); report['chunks'][name]['summary_sha256']=digest(folder/'summary.json')
    with pytest.raises(ValueError): export_case(root,report,output)
