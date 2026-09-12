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
