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
