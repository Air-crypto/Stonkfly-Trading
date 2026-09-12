import hashlib
import json
from pathlib import Path

import pytest

from paperlab.fly_recipient_isolation_figure import evidence


def fixture():
    # Entirely synthetic display values. Never publish as model or market evidence.
    p=json.loads(Path('reports/fly-recipient-isolation-protocol-01.json').read_text());rows={}
    for i,name in enumerate(p['arms']):
        rows[name]=[]
        for j in range(3):
            side=('BUY','HOLD','SELL')[(i+j)%3]
            rows[name].append({'observation':j+1,'side':side,'difference_hz':{'BUY':2,'HOLD':0,'SELL':-2}[side],
                'gate_spikes':1,'recipient_counts':{'10704':i+j,'11402':15-i+j},
                'spike_sha256':hashlib.sha256(f'{name}-{j}'.encode()).hexdigest()})
    comparisons=[]
    for i in (0,1):
        for memory in ('pristine','trained'):
            base=f'pool{i}-{memory}-both'
            for target in ('only_10704','only_11402'):
                variant=f'pool{i}-{memory}-{target}'
                comparisons.append({'baseline':base,'variant':variant,'changed_neurons':[100,500,1000],
                    'different_actions':[x['side']!=y['side'] for x,y in zip(rows[base],rows[variant])]})
    return {'status':'verified','full_observations_verified':36,'native_bins_verified':1800,'protocol':p,
            'rows':rows,'controls_verified':list(p['arms'])[:4],
            'reference_control_artifact_sha256':{str(i):'f'*64 for i in range(12)},'isolation_comparisons':comparisons}


def test_plot_keeps_all_conditions_and_actions():
    a=fixture();assert len(evidence(a))==12
    assert len(a['isolation_comparisons'])==8


@pytest.mark.parametrize('field',['control','target','label','action','comparison','negative','totals','learning'])
def test_plot_rejects_incomplete_or_misrepresented_evidence(field):
    a=fixture();name='pool0-trained-only_11402'
    if field=='control':a['reference_control_artifact_sha256'].pop('0')
    if field=='target':a['protocol']['arms'][name]['target_ids']=['10704']
    if field=='label':a['protocol']['arms'][name]['memory']='pristine'
    if field=='action':a['rows'][name][0]['side']='INVALID'
    if field=='comparison':a['isolation_comparisons'].pop()
    if field=='negative':a['rows'][name][0]['recipient_counts']['11402']=-1
    if field=='totals':a['isolation_comparisons'][0]['changed_neurons'][0]=0
    if field=='learning':a['protocol']['inference']['learning']=True
    with pytest.raises(ValueError):evidence(a)
