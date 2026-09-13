"""Non-preemptible completion order and accounting; no native trajectories."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from paperlab import fly_online_completion as completion
from paperlab.core import atomic_json, digest
from paperlab.fly_online_protocol import chunk_name, chunk_order
from test_fly_online_recovery import case as parent_case


@pytest.fixture
def case(parent_case, monkeypatch):
    old = parent_case
    for i in range(4): old.invoke(i)
    p = json.loads(Path('reports/fly-online-completion-protocol-11.json').read_text())
    root = old.state/completion.DIRECTORY
    monkeypatch.setattr(completion, 'reference', lambda *a: (p, old.p, old.env, old.sources, {'vectors_reconstructed': 49}))
    calls = []; commits = []
    def run(env, memory, news, data, out, **kw):
        name = chunk_name(kw['stage'], kw['pool_index'], kw['arm'])
        receipt = json.loads((out.parent/'receipt.json').read_text())
        assert receipt['nonpreemptible'] and receipt['price_multiplier'] == 3
        assert receipt['reserved_usd'] == pytest.approx(.1608936) and commits
        if kw['stage'] == 'test':
            assert len(kw['development']) == 8 and (root/'selection-receipt.json').exists()
        else: assert kw['development'] is None
        sample = json.loads((old.original/'chunks'/chunk_name(*chunk_order()[0])/'artifacts/summary.json').read_text())
        sample.update(chunk=name, stage=kw['stage'], pool_index=kw['pool_index'], arm=kw['arm'])
        sample['outcome']['equity'] = 251 if kw['arm'] == 'trained_online_reset_rates' else 249
        atomic_json(out/'summary.json', sample); calls.append(name); return sample
    def invoke(i=0, runner=run):
        return completion.execute(old.state, 'spec', 'evidence', Path.cwd(), call_id=f'completion-{i}',
            input_id=f'completion-input-{i}', commit=lambda: commits.append(True), run=runner)
    return SimpleNamespace(old=old, state=old.state, root=root, invoke=invoke, calls=calls)


def test_ten_fresh_conditions_preserve_all_prior_bytes_and_select_before_test(case):
    c = case
    before = {str(p): digest(p) for base in (c.old.original, c.old.root) for p in base.rglob('*') if p.is_file()}
    for i, condition in enumerate(chunk_order()[6:]):
        result = c.invoke(i)
        assert result['status'] == 'completion_chunk_completed' and result['completed_chunks'] == i + 7
        assert c.calls[-1] == chunk_name(*condition)
    final = c.invoke(10)
    assert final['status'] == 'paper_online_completion_captured' and final['audited'] is False
    assert final['parent_recovery'] == 'preempted_incomplete' and final['new_chunks'] == 10
    assert before == {str(p): digest(p) for base in (c.old.original, c.old.root) for p in base.rglob('*') if p.is_file()}
    assert c.invoke(11)['status'] == 'paper_online_completion_captured' and len(c.calls) == 10


@pytest.mark.parametrize('exception', [RuntimeError, KeyboardInterrupt])
def test_interrupted_capture_is_durable_and_never_retried(case, exception):
    calls = []
    def fail(*a, **k): calls.append(1); raise exception('fixture interruption')
    with pytest.raises(exception): case.invoke(runner=fail)
    result = case.invoke(1)
    assert result['status'] == 'completion_chunk_unresolved' and result['receipt']['status'] == 'failed'
    assert len(calls) == 1 and result['receipt']['reserved_usd'] == pytest.approx(.1608936)


def test_reservation_and_settlement_charge_three_times_with_existing_safety_factor(tmp_path):
    from paperlab.budget import settle
    path = tmp_path/'budget.json'; r = completion.reservation(path)
    assert r['reserve'] == pytest.approx(610*3*2*(.0000131*2+.00000222*8))
    assert r['rate'] == pytest.approx(3*2*(.0000131*2+.00000222*8))
    result = settle(path, r, 100)
    assert result['estimated_compute_usd'] == pytest.approx(110*r['rate'])
    assert result['monthly_reserved_usd'] == pytest.approx(result['estimated_compute_usd'])
    assert result['monthly_limit_usd'] == 25 and result['provider_bill'] is False


def test_monthly_budget_prevents_new_claim(case):
    from datetime import datetime, timezone
    month = datetime.now(timezone.utc).strftime('%Y-%m')
    atomic_json(case.state/'budget.json', {'first_month': month, 'months': {month: 18.75}})
    assert case.invoke()['status'] == 'budget_stopped'
    assert not case.calls and not (case.root/'chunks').exists()


def test_partial_selection_cannot_start_held_out_capture(case):
    for i in range(2): case.invoke(i)
    atomic_json(case.root/'selection.json', {'incomplete': True})
    with pytest.raises(ValueError): case.invoke(2)
    assert len(case.calls) == 2


def test_reference_failure_precedes_any_new_reservation(case, monkeypatch):
    before = digest(case.state/'budget.json')
    def fail(*a): raise ValueError('Prior failure evidence differs')
    monkeypatch.setattr(completion, 'reference', fail)
    with pytest.raises(ValueError, match='Prior failure'): case.invoke()
    assert digest(case.state/'budget.json') == before and not case.calls


def test_cloud_worker_has_nonpreemptible_resources_and_imports_in_isolation(tmp_path):
    import os
    import subprocess
    import sys
    tree = ast.parse(Path('completion_cloud.py').read_text())
    worker = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'worker')
    args = {k.arg: k.value for k in worker.decorator_list[0].keywords}
    assert ast.literal_eval(args['nonpreemptible']) is True
    assert ast.literal_eval(args['cpu']) == (2, 2) and ast.literal_eval(args['memory']) == (8192, 8192)
    assert ast.literal_eval(args['timeout']) == 600 and ast.literal_eval(args['retries']) == 0
    for name in ('cloud.py', 'recovery_cloud.py', 'completion_cloud.py'):
        (tmp_path/name).write_bytes(Path(name).read_bytes())
    result = subprocess.run([sys.executable, '-I', '-c',
        'import sys;sys.path.insert(0,sys.argv[1]);import completion_cloud;print("import passed")', str(tmp_path)],
        cwd=tmp_path, env={**os.environ, 'PAPERLAB_FLY':'1', 'PAPERLAB_UNIVERSE':'1',
            'PAPERLAB_PAPER_STUDY':'1', 'PAPERLAB_ONLINE_STUDY':'1'}, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert 'import passed' in result.stdout
