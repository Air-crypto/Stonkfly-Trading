import json
from pathlib import Path

import pytest

from paperlab.fly_restore_study import summarize


def fixture(tmp_path):
    study=json.loads(Path('reports/fly-restoration-study-01.json').read_text())
    protocol=json.loads(Path('reports/fly-restoration-protocol-01.json').read_text())
    (tmp_path/'protocol.json').write_text(json.dumps(protocol))
    for name,report in study['reports'].items():
        (tmp_path/name).mkdir()
        (tmp_path/name/'report.json').write_text(json.dumps(report))
    return study


def test_published_restoration_study_reaudits(tmp_path):
    expected=fixture(tmp_path)
    assert summarize(tmp_path)['summary']==expected['summary']


@pytest.mark.parametrize('mutation',['input','spikes','plasticity','training','build'])
def test_restoration_audit_rejects_broken_controls(tmp_path,mutation):
    fixture(tmp_path)
    path=tmp_path/'all/report.json'
    report=json.loads(path.read_text())
    if mutation=='input': report['events'][4]['input_sha256']='altered'
    if mutation=='spikes': report['events'][4]['spike_sha256']='altered'
    if mutation=='plasticity': report['events'][4]['diagnostics']['plasticity_enabled']=True
    if mutation=='training': report['events'][0]['spike_sha256']='altered'
    if mutation=='build': report['native_build']['binary_sha256']='altered'
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError): summarize(tmp_path)
