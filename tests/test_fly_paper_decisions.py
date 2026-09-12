import copy
from urllib.parse import parse_qs

import pytest

from paperlab.fly_paper_decisions import compare, markdown
from test_fly_paper_figure import report_fixture


def fixture():
    r = report_fixture()
    for pool in r['phase_diagnostics'].values():
        for phases in pool.values():
            for lane in phases.values():
                for row in lane['decisions']:
                    row['quote_ts'] = row['decision_ts']
                    if row['neural'] is not None:
                        row['neural'] = {'left_hz': 2, 'right_hz': 4, 'difference_hz': 2,
                                         'gate_spikes': 1, 'side': 'BUY', 'plasticity_enabled': False,
                                         'weight_delta_l2': 0, 'input_sha256': 'b'*64,
                                         'spike_sha256': 'c'*64, 'total_spikes': 20,
                                         'news_features': [0]*50}
    return r


def test_count_changes_are_not_automatically_action_changes_and_links_skip_missing_slots():
    report = fixture()
    rows = report['phase_diagnostics']['one']['trained_frozen']['test']['decisions']
    rows[0]['neural'].update(spike_sha256='d'*64, total_spikes=21)
    rows[2]['neural'].update(gate_spikes=0, side='HOLD', spike_sha256='e'*64)
    result = compare(report)
    c = next(x for x in result['comparisons'] if x['pool_index']==0 and x['phase']=='test' and x['comparison']=='memory_carry')
    assert c['first_full_count_difference'] == 1800
    assert c['first_decoder_difference'] == c['first_side_difference'] == 2400
    assert c['rows'][0]['side_equal'] and not c['rows'][0]['full_counts_equal']
    assert c['rows'][1]['observation_index'] is None and c['rows'][1]['viewer_query'] is None
    assert parse_qs(c['rows'][2]['viewer_query'][1:]) == {
        'run':['pool0-trained_frozen'], 'step':['1'], 'compare':['pool0-pristine_frozen']}
    assert c['rows'][3]['terminal'] and c['rows'][3]['viewer_query'] is None


def test_terminal_fill_keeps_earlier_signal_time_and_does_not_invent_a_new_decision():
    report = fixture();lane=report['phase_diagnostics']['one']['trained_frozen']['test']
    lane['decisions'][3]['fill']={'status':'filled','side':'BUY','decision_ts':2400,'fee':'0.25'}
    lane.update(fills=1, fees=.25)
    original=copy.deepcopy(report);result=compare(report)
    c=next(x for x in result['comparisons'] if x['pool_index']==0 and x['phase']=='test' and x['comparison']=='memory_carry')
    assert c['first_fill_difference']==2700 and c['first_side_difference'] is None
    assert c['rows'][3]['variant'] is None
    assert 'BUY from 00:40:00 · fee $0.250' in markdown(result)
    assert report==original


@pytest.mark.parametrize('field',['quote','coverage','input','news','decoder','learning'])
def test_refuse_ambiguous_input_or_nonfrozen_decision_comparisons(field):
    report=fixture();row=report['phase_diagnostics']['one']['trained_frozen']['test']['decisions'][0]
    if field=='quote':row['quote_ts']+=1
    if field=='coverage':row['neural']=None
    if field=='input':row['neural']['input_sha256']='f'*64
    if field=='news':row['neural']['news_features'][0]=1
    if field=='decoder':row['neural']['side']='SELL'
    if field=='learning':row['neural']['plasticity_enabled']=True
    with pytest.raises(ValueError):compare(report)
