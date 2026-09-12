import copy

import pytest

from paperlab.fly_decoder_audit import audit_view


def fixture(left=17, right=18, gate=3):
    event={'side':'BUY','left_hz':left*2.,'right_hz':right*2.,'difference_hz':(right-left)*2.,
           'gate_spikes':gate,'cell_ids':{'left':['10162'],'right':['10059'],'gate':['10527','555871']},'brain_ms':500.}
    event['side']='HOLD' if not gate or right==left else 'BUY' if right>left else 'SELL'
    counts=[[0,0,0,0] for _ in range(50)];counts[0]=[left,right,gate,0]
    return {'nodes':[{'id':i} for i in ['10162','10059','10527','555871']], 'report':{'events':[event]},
            'frames':[{'event':event,'times_ms':list(range(10,501,10)),'counts':counts}]}


def test_exact_threshold_is_one_spike_and_tie_holds():
    row=audit_view(fixture())['observations'][0]
    assert row['single_spike_changes_action']
    changed=[(r['body_id'],r['delta_spikes'],r['side']) for r in row['edits'] if r['action_changed']]
    assert changed==[('10162',1,'HOLD'),('10059',-1,'HOLD')]
    assert not audit_view(fixture(right=20))['observations'][0]['single_spike_changes_action']


def test_zero_gate_can_open_but_single_gate_can_close():
    row=audit_view(fixture(right=20,gate=0))['observations'][0]
    assert [(r['body_id'],r['delta_spikes'],r['side']) for r in row['edits'] if r['action_changed']]==[('10527',1,'BUY'),('555871',1,'BUY')]
    assert all(r['delta_spikes']==1 for r in row['edits'] if r['body_id'] in ['10527','555871'])
    row=audit_view(fixture(right=20,gate=1))['observations'][0]
    assert [(r['body_id'],r['delta_spikes'],r['side']) for r in row['edits'] if r['action_changed']]==[('10527',-1,'HOLD')]


def test_tie_needs_one_spike_for_either_direction_and_uses_ids():
    v=fixture(right=17); original=copy.deepcopy(v);r=audit_view(v)
    assert {e['side'] for e in r['observations'][0]['edits']}=={'BUY','SELL','HOLD'}
    assert v==original
    v['nodes'].reverse()
    for row in v['frames'][0]['counts']:row.reverse()
    assert audit_view(v)==r


@pytest.mark.parametrize('bad',['identity','event','rate','negative','fraction','time','missing'])
def test_inconsistent_recordings_fail(bad):
    v=fixture();f=v['frames'][0]
    if bad=='identity':v['nodes'][0]['id']='10059'
    if bad=='event':v['report']['events']=[{}]
    if bad=='rate':f['event']['left_hz']=3
    if bad=='negative':f['counts'][0][0]=-1
    if bad=='fraction':f['counts'][0][0]=.5
    if bad=='time':f['times_ms'][0]+=1
    if bad=='missing':v['nodes'][0]['id']='999'
    with pytest.raises(ValueError):audit_view(v)


def test_population_mean_controls_a_single_spike_contribution():
    v=fixture();v['nodes'].append({'id':'900'});v['frames'][0]['event']['cell_ids']['left'].append('900')
    for i,row in enumerate(v['frames'][0]['counts']):row.append(17 if i==0 else 0)
    row=audit_view(v)['observations'][0]
    added=next(e for e in row['edits'] if e['body_id']=='900' and e['delta_spikes']==1)
    assert added['difference_hz']==1 and added['side']=='HOLD'
