import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest

from paperlab.core import atomic_json
from paperlab.fly_credit_divergence import analyze, compare_arrays
from paperlab.fly_credit_selection import create_views


def arrays():
    ids=np.array([1000,1001,10162,10059,10527,555871,10704,11402],dtype=np.int64)
    return {'ms':np.arange(10,501,10,dtype=np.float64),'neuron_ids':ids,
        'plastic_edges':np.array([2000,2001]),'plastic_pre':np.array([0,1]),'plastic_post':np.array([6,7]),
        'initial_weights':np.ones(2,dtype=np.float32),'weights':np.ones((50,2),dtype=np.float32),
        'u':np.zeros((50,2)),'w':np.zeros((50,2)),'voltage':np.zeros((50,8),dtype=np.float32),
        'counts':np.zeros((50,8),dtype=np.int32)}


def test_identical_recordings_do_not_invent_a_first_difference():
    a=arrays();r=compare_arrays(a,copy.deepcopy(a))
    assert r['initial_weights_identical']
    assert all(v['first'] is None and sum(v['different_entities_per_bin'])==0 for v in r['fields'].values())


def test_first_bin_order_uses_sample_boundaries_and_detects_shifted_spikes():
    a=arrays();a['ms']+=500;a['counts'][14,0]=1;b=copy.deepcopy(a)
    for name in ('u','w','weights'):b[name][0,0]+=.1
    b['voltage'][4,6]=.25;b['counts'][13,0]=1;b['counts'][14,0]=0
    r=compare_arrays(a,b)
    assert [r['fields'][name]['first']['observation_end_ms'] for name in ('weights','voltage','counts')]==[10,50,140]
    assert r['fields']['counts']['first']['end_ms']==640
    assert r['fields']['counts']['different_entities_per_bin'][13:15]==[1,1]
    assert np.array_equal(a['counts'].sum(axis=0),b['counts'].sum(axis=0))
    assert r['first_weight_edges'][0]['edge']==2000 and r['first_weight_edges'][0]['target']=='10704'


def test_positive_direction_requires_a_gate_spike():
    a=arrays();a['counts'][0,3]=4;a['counts'][0,2]=1;a['counts'][37,4]=1
    b=copy.deepcopy(a);b['counts'][37,4]=0;r=compare_arrays(a,b)
    assert r['decoder']['carry']['side']=='BUY' and r['decoder']['reset']['side']=='HOLD'
    assert r['decoder']['carry']['difference_hz']==r['decoder']['reset']['difference_hz']==6
    assert r['decoder']['carry']['cumulative_gate_spikes'][36:39]==[0,1,1]


@pytest.mark.parametrize('bad',['identity','dtype','time','nan','negative_count'])
def test_misaligned_or_invalid_arrays_are_rejected(bad):
    a=arrays();b=copy.deepcopy(a)
    if bad=='identity':b['neuron_ids'][0]=1002
    if bad=='dtype':b['counts']=b['counts'].astype(np.int64)
    if bad=='time':a['ms'][2]=b['ms'][2]=35
    if bad=='nan':b['voltage'][0,0]=np.nan
    if bad=='negative_count':b['counts'][0,0]=-1
    with pytest.raises(ValueError):compare_arrays(a,b)


@pytest.fixture
def retained():
    value=os.environ.get('FLY_CREDIT_RESET_RECORDINGS')
    if not value:pytest.skip('Set FLY_CREDIT_RESET_RECORDINGS to the retained cloud audit/artifacts directory; no model run')
    return Path(value)


def test_retained_native_records_reproduce_the_complete_divergence(retained):
    r=analyze(retained/'audit.json',retained/'artifacts');pair=r['pairs'][0]
    assert [pair['observations'][1]['fields'][k]['first']['observation_end_ms'] for k in ('weights','voltage','counts')]==[10,50,140]
    edges=pair['observations'][1]['first_weight_edges']
    assert len(edges)==5 and {e['target'] for e in edges}=={'10704','11402'}
    assert pair['observations'][2]['decoder']['carry']['cell_spikes']['10527']==1
    assert pair['observations'][2]['decoder']['reset']['cell_spikes']['10527']==0
    assert all(o['fields']['voltage']['first'] is None for o in r['pairs'][2]['observations'])


def test_derived_online_views_copy_own_activity_and_reconstruct_added_credit(retained,tmp_path):
    ref=json.loads(Path('examples/fly-debugger/market10-pool0-trained_stimulated/view.json').read_text())
    for node in ref['nodes']:node['total_spikes']=999999
    for frame in ref['frames']:
        frame['counts']=[[999999]];frame['plastic_weights']=[[-999999]];frame['plastic_u']=[[-999999]]
    reference=tmp_path/'reference-labels-only.json';atomic_json(reference,ref)
    out=tmp_path/'views';metadata=create_views(retained/'audit.json',retained/'artifacts',reference,out)
    for name,info in metadata.items():
        v=json.loads((out/('creditdivergence01-'+name)/'view.json').read_text())
        old=json.loads((retained/'artifacts'/name/'view.json').read_text())
        assert v['report']==old['report'] and info['added_neurons']==['18540']
        node=next(i for i,n in enumerate(v['nodes']) if n['id']=='18540')
        edge=next(e for e in v['edges'] if e['id']==4110156);column=v['plastic_selection'].index(edge['plastic_index'])
        total=0
        for i,frame in enumerate(v['frames'],1):
            with np.load(retained/'artifacts'/name/f'step-{i:02}.npz',allow_pickle=False) as a:
                assert np.array_equal(np.array(frame['counts'])[:,node],a['counts'][:,7784])
                assert np.array_equal(np.array(frame['plastic_weights'])[:,column],a['weights'][:,8])
                assert np.array_equal(np.array(frame['plastic_u'])[:,column],a['u'][:,8])
                total+=int(a['counts'][:,7784].sum())
            assert frame['plastic_credit']['plastic_selection']==v['plastic_selection']
            assert frame['event']==old['frames'][i-1]['event']
        assert v['nodes'][node]['total_spikes']==total and total!=999999


def test_tampered_retained_trace_is_rejected_before_analysis(retained,tmp_path):
    for name in ('trained_online_recorded_carry','trained_online_recorded_reset_rates'):
        folder=tmp_path/name;folder.mkdir()
        (folder/'view.json').hardlink_to(retained/'artifacts'/name/'view.json')
    (tmp_path/'trained_online_recorded_carry/step-01.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='Audited artifact changed'):
        analyze(retained/'audit.json',tmp_path)
