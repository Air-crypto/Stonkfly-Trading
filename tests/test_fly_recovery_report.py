"""Recovery wiring fixtures are synthetic; none can pass the real release gate."""
import json
from pathlib import Path

import pytest

from paperlab import fly_recovery_report as recovery
from paperlab.core import atomic_json, digest
from paperlab.fly_market_study import signature
from paperlab.fly_online_figure import evidence
from paperlab.fly_online_protocol import chunk_name, chunk_order
from paperlab.fly_online_recovery import POLICY
from paperlab.fly_online_schedule import aggregate
from paperlab.fly_online_study import SOURCE_FILES, select_development
from paperlab.fly_selective_trace import require_completed_study
from test_fly_online import sealed
from test_fly_online_figure import figure_evidence


def raw(value):
    return json.dumps(value, indent=2) + '\n'


@pytest.fixture
def amended(figure_evidence, monkeypatch):
    root = figure_evidence
    read = lambda path: json.loads((root/path).read_text())
    envelope = read('plan.json'); r = envelope['plan']['registration']; report = read('report.json')
    sources = {'paperlab/' + name: 'a' * 64 for name in SOURCE_FILES}
    for i in range(56 - len(sources)):
        sources[f'fixture-source-{i}.py'] = 'b' * 64
    names = [chunk_name(*c) for c in chunk_order()]
    created = max(r['end'] + 2000, 1789268000)
    failed = {'status': 'failed', 'claimed_at': r['end'] + 900}
    halt = {'at': created - 10}
    history = {
        'fly-online-failure-11.json': raw({'failure': {'receipt': failed, 'call_state': 'failed',
                                                     'observed_at': created - 20}}),
        'fly-online-recovery-preflight-failure-11.json': raw({'halt': halt,
            'halt_sha256': recovery.text_sha(raw(halt)), 'terminal_call': {'status': 'terminal_exception'}}),
        'fly-news-process-probe-11.json': raw({'synthetic_fixture': True})}
    p = {'version': 2, 'created_at': created, 'policy': POLICY, 'seed': '0',
        'plan_sha256': envelope['sha256'], 'original_sources_sha256': signature(sources),
        'failed_receipt_sha256': recovery.text_sha(raw(failed)), 'controls': {},
        'code_sha256': {'paperlab/fly_online_recovery.py': 'c'*64, 'recovery_cloud.py': 'd'*64},
        'evidence_sha256': {}}
    news = {'news_snapshot_sha256': envelope['plan']['news_snapshot_sha256'],
            'vectors_reconstructed': len(envelope['plan']['news_features'])}
    armed = {'armed_at': r['recorded_at'], 'registration_sha256': recovery.text_sha(raw(r)),
        'registration_signature': signature(r), 'source_sha256': signature(sources),
        'call_id': 'fixture-arming', 'input_id': 'fixture-arming-input'}
    proof = {'sources': raw(sources), 'failed_receipt': raw(failed), 'prior_halt': raw(halt),
        'evidence': history, 'preregistration': raw(r), 'armed': raw(armed),
        'sealing': raw({'status': 'completed', 'plan_sha256': envelope['sha256'],
            'registration_sha256': armed['registration_sha256'], 'claimed_at': r['end'],
            'call_id': 'fixture-seal', 'input_id': 'fixture-seal-input'}),
        'input_check': raw({'seed': '0', 'interpreter_start': 'fresh-subprocess',
            'verified_hash_probe': 7584921261715552910, 'original_inputs_changed': False,
            'news': news, 'checked_at': created, 'call_id': 'fixture-2', 'input_id': 'fixture-input-2'}),
        'call_results': {}, 'projections': {}, 'all_recorded_artifacts_rehashed': True}
    summaries = {}; receipts = {}; selected = None
    for i, (stage, pool, arm) in enumerate(chunk_order()):
        if i == 8:
            selected = select_development(envelope, summaries.copy())
        name = names[i]; summary_path = f'chunks/{name}/summary.json'; summary = read(summary_path)
        summary.update(code_sha256={f: sources['paperlab/' + f] for f in SOURCE_FILES},
            artifact_sha256={'initial-dynamics.npz': 'e'*64, 'pristine-memory.npz': 'f'*64},
            native_build={'binary_sha256': '1'*64}, graph={'neurons': 166700, 'edges': 25582938, 'plastic_edges': 7835},
            news_audit=news, selection=selected)
        atomic_json(root/summary_path, summary); summaries[name] = summary
        audit_path = f'audits/{name}.json'; audit = read(audit_path)
        audit.update(artifact_sha256=summary['artifact_sha256'], executed_source_sha256=summary['code_sha256'])
        atomic_json(root/audit_path, audit); report['audit_sha256'][name] = digest(root/audit_path)
        claimed = r['end'] + i*300 if i < 2 else created + 1 + (i-2)*300
        receipt = {'status': 'completed', 'chunk': name, 'call_id': f'fixture-{i}', 'input_id': f'fixture-input-{i}',
            'plan_sha256': envelope['sha256'], 'source_sha256': signature(sources),
            'remote_path': f'/state/{recovery.ORIGINAL if i < 2 else recovery.DIRECTORY}/chunks/{name}/artifacts',
            'dispatch': 'scheduled-worker-after-paper-cycle' if i < 2 else 'separate-recovery-worker-shared-lease',
            'claimed_at': claimed, 'completed_at': claimed + 120, 'summary_sha256': digest(root/summary_path)}
        projection = {'status': 'online_view_projection_audited', 'validation_only': False, 'chunk': name,
            'plan_sha256': envelope['sha256'], 'summary_sha256': receipt['summary_sha256'],
            'native_audit_sha256': report['audit_sha256'][name], 'artifact_sha256': summary['artifact_sha256'],
            'projection': {'all_topology_verified': True, 'all_plotted_series_verified': True},
            'graph_data_sha256': {'graph': '2'*64}, 'synthetic_fixture': True}
        proof['projections'][name] = raw(projection); receipts[name] = receipt
        if i < 2:
            atomic_json(root/f'receipts/{name}-completed.json', receipt)
            p['controls'][name] = {'receipt_sha256': digest(root/f'receipts/{name}-completed.json'),
                'summary_sha256': receipt['summary_sha256'], 'projection_sha256': recovery.text_sha(raw(projection))}
            history['fly-online-control-' + ('first' if i == 0 else 'second') + '-11.json'] = raw(projection)
    p['evidence_sha256'] = {name: recovery.text_sha(value) for name, value in history.items()}
    proof['protocol'] = raw(p)
    # Only the immutable protocol pin is replaced. Its schema and all downstream
    # relationships use the real validator; the plan still forces fixture status.
    monkeypatch.setattr(recovery, 'PROTOCOL_SHA256', recovery.text_sha(raw(p)))
    for i, name in enumerate(names):
        receipt = receipts[name]
        if i >= 2:
            receipt.update(amendment_sha256=signature(p), reserved_usd=.0536312,
                budget={'estimated_compute_usd': .01, 'monthly_limit_usd': 25, 'provider_bill': False})
        atomic_json(root/f'receipts/{name}-completed.json', receipt)
        result = ({'status': 'paper_online_chunk_completed', 'chunk': name, 'receipt': receipt} if i < 2 else
                  {'status': 'recovery_chunk_completed', 'completed_chunks': i + 1, 'receipt': receipt})
        proof['call_results'][name] = raw({'study_11': result} if i < 2 else result)
    selection_raw = raw(selected); proof['selection'] = selection_raw
    agg = aggregate(envelope, summaries, selected)
    proof['cloud_summary'] = raw({'status': 'paper_online_recovery_captured', 'audited': False, 'amendment': p,
        'original_attempt': 'failed', 'reused_controls': names[:2], 'new_chunks': 14, 'aggregate': agg})
    report.update(**{k: agg[k] for k in ('total_equity', 'observations', 'selection', 'chunk_sha256')})
    report.update(status=recovery.STATUS, original_attempt='failed', reused_controls=names[:2], new_chunks=14,
        recovery=proof, cloud_registration=armed, selection_receipt={'call_id': 'fixture-8',
            'input_id': 'fixture-input-8', 'selected_at': receipts[names[8]]['claimed_at'],
            'selection_sha256': recovery.text_sha(selection_raw), 'development_chunk_sha256': selected['development_chunk_sha256']})
    atomic_json(root/'report.json', report)
    return root


def test_amendment_checks_all_sixteen_calls_without_erasing_fixture_status(amended):
    report, lanes, _, fixture = evidence(amended)
    assert fixture and len(lanes) == 16 and report['original_attempt'] == 'failed'
    with pytest.raises(ValueError, match='actual study'):
        require_completed_study(amended)


@pytest.mark.parametrize('change', ['protocol', 'failed', 'halt', 'source', 'seed', 'call',
    'projection', 'cloud_summary', 'selection_bytes', 'relabel', 'incomplete', 'control_receipt'])
def test_recovery_cannot_hide_failures_or_change_recorded_provenance(amended, change):
    path = amended/'report.json'; report = json.loads(path.read_text()); proof = report['recovery']
    if change == 'protocol': proof['protocol'] += ' '
    if change == 'failed': proof['failed_receipt'] += ' '
    if change == 'halt': proof['prior_halt'] += ' '
    if change == 'source': proof['sources'] = '{}'
    if change == 'seed':
        check = json.loads(proof['input_check']); check['interpreter_start'] = 'image-env'; proof['input_check'] = raw(check)
    if change == 'call':
        name = 'test-pool1-trained_online_carry'; result = json.loads(proof['call_results'][name])
        result['receipt']['input_id'] = 'another-input'; proof['call_results'][name] = raw(result)
    if change == 'projection':
        name = 'test-pool0-trained_frozen'; p = json.loads(proof['projections'][name])
        p['native_audit_sha256'] = '0'*64; proof['projections'][name] = raw(p)
    if change == 'cloud_summary':
        cloud = json.loads(proof['cloud_summary']); cloud['new_chunks'] = 13; proof['cloud_summary'] = raw(cloud)
    if change == 'selection_bytes': proof['selection'] += ' '
    if change == 'relabel': report['status'] = 'paper_online_study_audited'
    if change == 'incomplete': report['audited'] = False
    if change == 'control_receipt':
        receipt = amended/'receipts/development-pool0-trained_frozen-completed.json'
        receipt.write_text(receipt.read_text() + ' ')
    atomic_json(path, report)
    with pytest.raises(ValueError): evidence(amended)


def test_finalizer_does_not_create_output_when_capture_is_incomplete(tmp_path):
    out = tmp_path/'not-complete'
    with pytest.raises(FileNotFoundError):
        recovery.build(tmp_path, tmp_path, [tmp_path, tmp_path], tmp_path, tmp_path/'prices.db', out)
    assert not out.exists()


@pytest.fixture
def completed(amended, monkeypatch):
    from paperlab.fly_online_completion import POLICY, DIRECTORY, PARENT_PROTOCOL_SHA256
    root = amended; report = json.loads((root/'report.json').read_text()); proof = report['recovery']
    parent = json.loads(proof['protocol']); names = [chunk_name(*c) for c in chunk_order()]
    failure = {'receipt': {'chunk': names[6], 'status': 'claimed', 'amendment_sha256': signature(parent),
        'call_id': 'fixture-preempted-call'}, 'terminal_result': {'status': 'writer_busy', 'writer': 'worker'},
        'provider_event': 'Container terminated due to preemption', 'recovery_app_stopped': True,
        'observed_at': parent['created_at'] + 1500, 'complete_summary_present': False, 'development_selection_present': False}
    cp = {'version': 1, 'created_at': parent['created_at'] + 2000, 'policy': POLICY,
        'parent_protocol_sha256': PARENT_PROTOCOL_SHA256,
        'preemption_report_sha256': recovery.text_sha(raw(failure)), 'retained': {},
        'code_sha256': {'paperlab/fly_online_completion.py': 'a'*64, 'completion_cloud.py': 'b'*64}}
    for name in names[:6]:
        rp = root/f'receipts/{name}-completed.json'; receipt = json.loads(rp.read_text())
        cp['retained'][name] = {'receipt_sha256': digest(rp), 'summary_sha256': receipt['summary_sha256'],
                              'projection_sha256': recovery.text_sha(proof['projections'][name])}
    monkeypatch.setattr(recovery, 'COMPLETION_PROTOCOL_SHA256', recovery.text_sha(raw(cp)))
    news = json.loads(proof['input_check'])
    news.update(call_id='completion-6', input_id='completion-input-6', checked_at=cp['created_at'])
    proof['completion'] = {'protocol': raw(cp), 'preemption_report': raw(failure), 'input_check': raw(news)}
    for i, name in enumerate(names[6:], 6):
        path = root/f'receipts/{name}-completed.json'; receipt = json.loads(path.read_text())
        claim = cp['created_at'] + 1 + (i-6)*300
        receipt.update(call_id=f'completion-{i}', input_id=f'completion-input-{i}', claimed_at=claim,
            completed_at=claim+120, amendment_sha256=signature(cp), remote_path=f'/state/{DIRECTORY}/chunks/{name}/artifacts',
            dispatch='nonpreemptible-completion-worker-shared-lease', reserved_usd=.1608936,
            nonpreemptible=True, price_multiplier=3)
        atomic_json(path, receipt)
        proof['call_results'][name] = raw({'status': 'completion_chunk_completed', 'completed_chunks': i+1, 'receipt': receipt})
    report['selection_receipt'].update(call_id='completion-8', input_id='completion-input-8',
        selected_at=cp['created_at']+601)
    cloud = json.loads(proof['cloud_summary'])
    cloud.update(status='paper_online_completion_captured', amendment=cp, parent_amendment=parent,
        parent_recovery='preempted_incomplete', reused_conditions=names[:6], new_chunks=10)
    proof['cloud_summary'] = raw(cloud)
    report.update(status=recovery.COMPLETION_STATUS, parent_recovery='preempted_incomplete',
                  completion_reused_conditions=names[:6], completion_new_chunks=10)
    atomic_json(root/'report.json', report)
    return root


def test_nonpreemptible_completion_preserves_both_amendments_and_stays_a_fixture(completed):
    report, lanes, _, fixture = evidence(completed)
    assert fixture and len(lanes) == 16 and report['completion_new_chunks'] == 10
    with pytest.raises(ValueError, match='actual study'): require_completed_study(completed)


@pytest.mark.parametrize('change', ['missing_amendment', 'relabel_parent', 'price_multiplier', 'retained_receipt', 'unselected_test_owner'])
def test_completion_cannot_hide_preemption_or_understate_its_cost(completed, change):
    path = completed/'report.json'; report = json.loads(path.read_text()); proof = report['recovery']
    if change == 'missing_amendment': proof.pop('completion')
    if change == 'relabel_parent': report['status'] = recovery.STATUS
    if change == 'price_multiplier':
        name = 'test-pool0-trained_online_carry'; rp = completed/f'receipts/{name}-completed.json'
        receipt = json.loads(rp.read_text()); receipt['price_multiplier'] = 1; atomic_json(rp, receipt)
        call = json.loads(proof['call_results'][name]); call['receipt'] = receipt; proof['call_results'][name] = raw(call)
    if change == 'retained_receipt':
        rp = completed/'receipts/development-pool1-pristine_frozen-completed.json'; rp.write_text(rp.read_text() + ' ')
    if change == 'unselected_test_owner': report['selection_receipt']['call_id'] = 'unrelated-call'
    atomic_json(path, report)
    with pytest.raises(ValueError): evidence(completed)
