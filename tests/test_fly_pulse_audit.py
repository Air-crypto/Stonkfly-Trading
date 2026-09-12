import copy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from paperlab.fly_pulse_audit import audit,figure


def fixture():
    return json.loads(Path('reports/fly-market-pulse-study-01.json').read_text()),Path('reports/fly-market-study-04.json').read_text()


def test_factorial_audit_covers_every_single_factor_edge():
    study,reference=fixture();result=audit(study,reference)
    assert len(result['comparisons'])==12
    assert {factor:sum(c['factor']==factor for c in result['comparisons']) for factor in ('memory','learning','pulses')}=={'memory':4,'learning':4,'pulses':4}
    ET.fromstring(figure(study))


@pytest.mark.parametrize('mutation',['reference','initial_memory','image','pulse','weight','control'])
def test_pulse_audit_rejects_broken_pairing(mutation):
    study,reference=fixture();study=copy.deepcopy(study)
    r=study['reports']['pristine_frozen_recorded']
    if mutation=='reference':reference+=' '
    if mutation=='initial_memory':r['initial_memory_sha256']='altered'
    if mutation=='image':r['events'][0]['input_sha256']='altered'
    if mutation=='pulse':r['events'][1]['stimulus']='none'
    if mutation=='weight':r['events'][0]['diagnostics']['weight_delta_l2']=1
    if mutation=='control':study['reports']['trained_online_recorded']['events'][2]['spike_sha256']='altered'
    with pytest.raises(ValueError):audit(study,reference)


def timing_fixture():
    from paperlab.fly_market_pulse import ARMS
    study,_=fixture()
    views={name:json.loads(Path(f'examples/fly-debugger/pulse01-{name}/view.json').read_text()) for name in ARMS}
    return study,views


def test_gate_timing_reconciles_each_control_and_observation():
    from paperlab.fly_pulse_audit import gate_timing
    study,views=timing_fixture();result=gate_timing(study,views)
    assert len(result['arms'])==8
    assert all(len(rows)==3 for rows in result['arms'].values())
    active=result['arms']['trained_online_recorded'][2]
    assert active['neurons']['10527']==[{'bin':37,'start_ms':1370.,'end_ms':1380.,'spikes':1}]
    assert result['arms']['trained_frozen_recorded'][2]['gate_spikes']==0


@pytest.mark.parametrize('mutation',['gate_count','missing_gate','time_grid','time_offset','report','event'])
def test_gate_timing_rejects_mislabeled_or_incomplete_recordings(mutation):
    from paperlab.fly_pulse_audit import gate_timing
    study,views=timing_fixture();view=views['trained_online_recorded'];frame=view['frames'][2]
    if mutation=='gate_count':frame['counts'][37][next(i for i,n in enumerate(view['nodes']) if n['id']=='10527')]=0
    if mutation=='missing_gate':next(n for n in view['nodes'] if n['id']=='10527')['id']='missing'
    if mutation=='time_grid':frame['times_ms'][37]+=1
    if mutation=='time_offset':frame['times_ms']=[t+100 for t in frame['times_ms']]
    if mutation=='report':view['report']['initial_memory_sha256']='changed'
    if mutation=='event':frame['event']['side']='HOLD'
    with pytest.raises(ValueError):gate_timing(study,views)
