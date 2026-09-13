"""Resource and budget integration checks; no native construction or Modal RPCs."""
import ast
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from paperlab import fly_selective_trace_cloud as bridge
from paperlab.core import atomic_json, digest
from test_fly_selective_trace_cloud import worker, request
from test_fly_study_evidence_bundle import transport
from test_fly_online import sealed
from test_fly_online_figure import figure_evidence


@pytest.mark.parametrize('fail', [False, True])
def test_nonpreemptible_reservation_precedes_capture_and_failed_claim_cannot_repeat(worker, monkeypatch, fail):
    req, root, events, commit = worker; budget = root.parent/'budget.json'
    month = datetime.now(timezone.utc).strftime('%Y-%m')
    def capture(payload, reference, data, output, *, completed_study):
        assert events == ['commit', 'commit']  # Reservation, then persistent claim.
        assert json.loads(budget.read_text())['months'][month] == pytest.approx(.1608936)
        assert json.loads((root/bridge.CLAIM).read_text())['call_id'] == 'fc-test'
        events.append('capture')
        if fail: raise TimeoutError('Native deadline')
        protocol, _ = bridge.validate_request(req)
        for name in bridge.artifact_names(protocol):
            path = output/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('fixture')
        return {'fixture_capture': True}
    monkeypatch.setattr(bridge, 'capture', capture)
    invoke = lambda: bridge.run_budgeted_request(req, root.parent, 'data',
        call_id='fc-test', input_id='in-test', commit=commit)
    if fail:
        with pytest.raises(TimeoutError): invoke()
        assert json.loads(budget.read_text())['months'][month] == pytest.approx(.1608936)
        assert json.loads((root/bridge.CLAIM).read_text())['status'] == 'failed'
    else:
        result = invoke(); b = result['budget']
        assert b['reserved_usd'] == pytest.approx(.1608936) and b['price_multiplier'] == 3
        assert b['nonpreemptible'] and b['monthly_limit_usd'] == 25 and not b['provider_bill']
        assert b['estimated_compute_usd'] >= 10*3*2*(.0000131*2+.00000222*8)
        assert json.loads(budget.read_text())['months'][month] == pytest.approx(b['estimated_compute_usd'])
        assert json.loads((root/req['run_id']/'cloud-result.json').read_text()) == result
    before = digest(budget)
    with pytest.raises(ValueError, match='already claimed'): invoke()
    assert digest(budget) == before and events.count('capture') == 1


def test_bad_study_does_not_reserve_compute(worker):
    req, root, events, commit = worker
    req['selective_plan']['study_report_sha256'] = '0'*64
    with pytest.raises(ValueError, match='report differs'):
        bridge.run_budgeted_request(req, root.parent, 'data', call_id='fc-test', input_id='in-test', commit=commit)
    assert not events and not (root.parent/'budget.json').exists() and not (root/bridge.CLAIM).exists()


def test_shared_cap_stops_before_claim_or_capture(worker):
    req, root, events, commit = worker
    month = datetime.now(timezone.utc).strftime('%Y-%m')
    atomic_json(root.parent/'budget.json', {'first_month': month, 'months': {month: 18.75}})
    result = bridge.run_budgeted_request(req, root.parent, 'data', call_id='fc-test', input_id='in-test', commit=commit)
    assert result == {'status': 'budget_stopped'} and not events and not (root/bridge.CLAIM).exists()


def test_unscheduled_worker_shares_lease_and_imports_in_packaged_layout(tmp_path):
    tree = ast.parse(Path('selective_cloud.py').read_text())
    worker = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'worker')
    options = {k.arg: k.value for k in worker.decorator_list[0].keywords}
    assert 'schedule' not in options
    for name, value in {'nonpreemptible': True, 'cpu': (2, 2), 'memory': (8192, 8192),
        'timeout': 600, 'retries': 0, 'max_containers': 1, 'min_containers': 0,
        'single_use_containers': True}.items():
        assert ast.literal_eval(options[name]) == value
    call = worker.body[0].value
    assert isinstance(call, ast.Call) and call.func.id == 'exclusive' and ast.literal_eval(call.args[0]) == 'worker'
    for name in ('cloud.py', 'selective_cloud.py'): (tmp_path/name).write_bytes(Path(name).read_bytes())
    result = subprocess.run([sys.executable, '-I', '-c',
        'import sys;sys.path.insert(0,sys.argv[1]);import selective_cloud;print("import passed")', str(tmp_path)],
        cwd=tmp_path, env={**os.environ, 'PAPERLAB_FLY': '1', 'PAPERLAB_UNIVERSE': '1',
            'PAPERLAB_PAPER_STUDY': '0', 'PAPERLAB_ONLINE_STUDY': '0'}, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert 'import passed' in result.stdout
