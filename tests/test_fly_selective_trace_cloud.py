"""Transport and once-only execution tests; no native propagation or cloud calls.

Positive transport cases use synthetic completed-study evidence with only the
release decision mocked. The actual evidence consistency checks still run.
"""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from paperlab import fly_selective_trace_cloud as bridge
from paperlab.core import atomic_json, digest
from paperlab.fly_market_study import signature
from test_fly_selective_trace import payload
from test_fly_study_evidence_bundle import transport
from test_fly_online import sealed
from test_fly_online_figure import figure_evidence


def request(release):
    return {'run_id': 'assay-selective-test', 'selective_plan': {
        'payload': payload(), 'bundle_sha256': release['bundle_sha256'],
        'study_report_sha256': release['report_sha256'], 'source_sha256': bridge.source_hashes()}}


def forbidden(*args, **kwargs):
    raise AssertionError('No real network or native construction is allowed in this test')


def test_actual_incomplete_study_cannot_reach_modal(tmp_path, monkeypatch):
    import modal
    monkeypatch.setattr(modal.Function, 'from_name', forbidden)
    monkeypatch.setattr('paperlab.fly_selective_trace.TraceLab', forbidden)
    out = tmp_path/'output'
    with pytest.raises(FileNotFoundError):
        bridge.cloud_run(payload(), out, 'unused', 'unused', completed_study=tmp_path)
    assert not out.exists()


def test_actual_complete_synthetic_study_cannot_reach_modal(figure_evidence, tmp_path, monkeypatch):
    import modal
    monkeypatch.setattr(modal.Function, 'from_name', forbidden)
    monkeypatch.setattr('paperlab.fly_selective_trace.TraceLab', forbidden)
    with pytest.raises(ValueError, match='actual study'):
        bridge.cloud_run(payload(), tmp_path/'out', 'unused', 'unused', completed_study=figure_evidence)
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('change', ['path', 'extra', 'protocol', 'bundle', 'report', 'source'])
def test_request_is_bound_to_protocol_sources_and_safe_identifiers(transport, change):
    req = request(transport[2])
    if change == 'path': req['run_id'] = 'assay-selective-../../escape'
    elif change == 'extra': req['config'] = {}
    elif change == 'protocol': req['selective_plan']['payload']['protocol_json'] += ' '
    elif change == 'source': req['selective_plan']['source_sha256'] = {}
    else: req['selective_plan'][{'bundle': 'bundle_sha256', 'report': 'study_report_sha256'}[change]] = '../bad'
    with pytest.raises(ValueError): bridge.validate_request(req)


class Volume:
    def __init__(self):
        self.files = {}; self.uploads = []; self.reads = []; self.fail_once = None
        self.read_file = ReadFile(self)

    def batch_upload(self): return self
    def __enter__(self): return self
    def __exit__(self, *args): return False

    def put_file(self, local, remote):
        self.uploads.append(remote)
        self.files[remote] = Path(local).read_bytes()

    def read(self, name):
        self.reads.append(name)
        if name not in self.files: raise FileNotFoundError(name)
        raw = self.files[name]
        yield raw[:len(raw)//2]
        if self.fail_once == name:
            self.fail_once = None
            raise ConnectionError('Interrupted artifact stream')
        yield raw[len(raw)//2:]


class ReadFile:
    def __init__(self, volume): self.volume = volume
    def __call__(self, name): return self.volume.read(name)
    async def aio(self, name):
        for block in self.volume.read(name): yield block


@pytest.fixture
def cloud(transport, monkeypatch):
    import modal
    state = SimpleNamespace(volume=Volume(), calls=[], observed=[], result=None, lost=False,
                            owner=None, busy=False, remote_error=False)
    def spawn(**kwargs):
        state.calls.append(kwargs)
        if state.lost: raise ConnectionError('Submission response lost')
        return SimpleNamespace(object_id='fc-selective-once')
    def get(timeout):
        assert 0 <= timeout <= 50
        if state.remote_error: raise RuntimeError('Native worker failed')
        if state.result is None: raise TimeoutError()
        return state.result
    def observe(call_id):
        state.observed.append(call_id)
        return SimpleNamespace(get=get)
    fn = SimpleNamespace(spawn=spawn, get_current_stats=lambda: SimpleNamespace(
        num_total_runners=0, num_running_inputs=int(state.busy), backlog=0))
    monkeypatch.setattr(modal.Function, 'from_name', lambda *a, **kw: fn)
    monkeypatch.setattr(modal.FunctionCall, 'from_id', observe)
    monkeypatch.setattr(modal.Dict, 'from_name', lambda *a, **kw: SimpleNamespace(get=lambda key: state.owner))
    monkeypatch.setattr(modal.Volume, 'from_name', lambda *a, **kw: state.volume)
    monkeypatch.setattr(bridge, 'verify_reference', lambda *a: None)
    return state


def invoke(transport, root):
    return bridge.cloud_run(payload(), root, 'reference', 'graph', completed_study=transport[0])


def test_timeout_reattaches_one_call_and_uses_saved_bundle(transport, cloud, tmp_path):
    root = tmp_path/'out'
    for _ in range(3): assert invoke(transport, root)['status'] == 'pending'
    assert len(cloud.calls) == len(cloud.volume.uploads) == 1
    assert cloud.observed == ['fc-selective-once']*3
    assert (root/'study-evidence.json').read_bytes() == transport[1].read_bytes()


def test_lost_submission_response_is_never_resubmitted(transport, cloud, tmp_path):
    cloud.lost = True; root = tmp_path/'out'
    with pytest.raises(ConnectionError): invoke(transport, root)
    with pytest.raises(RuntimeError, match='Uncertain submission'): invoke(transport, root)
    assert len(cloud.calls) == 1 and not cloud.observed


def test_uncertain_upload_can_resume_without_duplicating_paid_work(transport, cloud, tmp_path, monkeypatch):
    root = tmp_path/'out'; original = cloud.volume.put_file
    def upload_then_disconnect(local, remote):
        original(local, remote)
        raise ConnectionError('Upload succeeded but acknowledgement lost')
    monkeypatch.setattr(cloud.volume, 'put_file', upload_then_disconnect)
    with pytest.raises(ConnectionError): invoke(transport, root)
    assert not cloud.calls
    assert json.loads((root/'cloud-call.json').read_text())['status'] == 'prepared'
    monkeypatch.setattr(cloud.volume, 'put_file', original)
    assert invoke(transport, root)['status'] == 'pending'
    assert len(cloud.calls) == len(cloud.volume.uploads) == 1


@pytest.mark.parametrize('busy', ['owner', 'runners', 'remote_claim'])
def test_existing_worker_or_claim_prevents_submission(transport, cloud, tmp_path, busy):
    if busy == 'owner': cloud.owner = {'call_id': 'fc-live'}
    elif busy == 'runners': cloud.busy = True
    else: cloud.volume.files['/fly-debugger/'+bridge.CLAIM] = b'{"status":"claimed"}'
    with pytest.raises(RuntimeError): invoke(transport, tmp_path/'out')
    assert not cloud.calls and not cloud.volume.uploads and not (tmp_path/'out').exists()


def test_remote_failure_does_not_restart(transport, cloud, tmp_path):
    cloud.remote_error = True
    for _ in range(2):
        with pytest.raises(RuntimeError, match='Native worker failed'): invoke(transport, tmp_path/'out')
    assert len(cloud.calls) == 1 and cloud.observed == ['fc-selective-once']*2


def fixture_result(req, volume):
    p, _ = bridge.validate_request(req)
    report = {'status': 'selective_trace_captured_pending_audit', 'protocol': p,
              'code_sha256': req['selective_plan']['source_sha256'],
              'completed_study_sha256': req['selective_plan']['study_report_sha256']}
    base = '/state/fly-debugger/'+req['run_id']; hashes = {}
    for name in bridge.artifact_names(p):
        raw = json.dumps(report).encode() if name == 'summary.json' else ('fixture '+name).encode()
        hashes[name] = hashlib.sha256(raw).hexdigest()
        volume.files[base.removeprefix('/state')+'/'+name] = raw
    return {'status': 'selective_trace_completed', 'run_id': req['run_id'], 'remote_path': base,
            'report': report, 'artifact_sha256': hashes, 'request_sha256': signature(req),
            'call_id': 'fc-selective-once', 'input_id': 'in-selective-once',
            'budget': {'monthly_limit_usd': 25, 'estimated_compute_usd': 0.01, 'provider_bill': False}}


@pytest.mark.parametrize('change', ['identity', 'path', 'request', 'source', 'study', 'cap', 'extra_file', 'missing_file'])
def test_wrong_terminal_evidence_is_not_downloaded_or_restarted(transport, cloud, tmp_path, change):
    root = tmp_path/'out'; invoke(transport, root)
    req = json.loads((root/'request.json').read_text()); result = fixture_result(req, cloud.volume)
    if change == 'identity': result['call_id'] = 'fc-other'
    elif change == 'path': result['remote_path'] += '/elsewhere'
    elif change == 'request': result['request_sha256'] = '0'*64
    elif change == 'source': result['report']['code_sha256'] = {}
    elif change == 'study': result['report']['completed_study_sha256'] = '0'*64
    elif change == 'cap': result['budget']['monthly_limit_usd'] = 100
    elif change == 'extra_file': result['artifact_sha256']['../../escape'] = '0'*64
    else: del result['artifact_sha256']['circuit.npz']
    cloud.result = result
    with pytest.raises(ValueError): invoke(transport, root)
    assert len(cloud.calls) == 1 and not (root/'artifacts').exists()


def test_interrupted_download_resumes_then_audits_before_installing_views(transport, cloud, tmp_path, monkeypatch):
    root = tmp_path/'out'; invoke(transport, root)
    req = json.loads((root/'request.json').read_text()); cloud.result = fixture_result(req, cloud.volume)
    remote = cloud.result['remote_path'].removeprefix('/state')
    cloud.volume.fail_once = remote+'/trained-memory.npz'
    with pytest.raises(ConnectionError): invoke(transport, root)
    assert (root/'artifacts/initial-dynamics.npz').exists()
    assert not (root/'artifacts/trained-memory.npz').exists()
    assert not (root/'audit.json').exists()
    audits = []
    def fake_audit(report, p, artifacts, reference, data, *, completed_study):
        # The underlying full neural auditor is tested separately. Here test the
        # transport of every required file and the audit-before-install ordering.
        assert (completed_study/'report.json').exists()
        assert not (root/'audit.json').exists()
        for name, sha in cloud.result['artifact_sha256'].items(): assert digest(artifacts/name) == sha
        audits.append(True)
        return {'status': 'fixture_audit'}, {name: {'report': {'fixture': True}}
                                           for name in report['protocol']['arms']}
    monkeypatch.setattr(bridge, 'audit', fake_audit)
    receipt = invoke(transport, root)
    assert receipt['status'] == 'completed' and len(audits) == len(cloud.calls) == 1
    assert len(cloud.result['artifact_sha256']) == 175
    assert cloud.volume.reads.count(remote+'/initial-dynamics.npz') == 1
    assert cloud.volume.reads.count(remote+'/trained-memory.npz') == 2
    for arm in cloud.result['report']['protocol']['arms']:
        assert json.loads((root/arm/'remote.json').read_text())['call_id'] == 'fc-selective-once'
    assert invoke(transport, root) == receipt
    (root/'artifacts/circuit.npz').write_text('changed')
    with pytest.raises(ValueError, match='Completed local evidence differs'): invoke(transport, root)
    assert len(cloud.calls) == 1


def test_corrupt_download_never_becomes_an_installed_artifact(transport, cloud, tmp_path):
    root = tmp_path/'out'; invoke(transport, root)
    req = json.loads((root/'request.json').read_text()); cloud.result = fixture_result(req, cloud.volume)
    remote = cloud.result['remote_path'].removeprefix('/state')+'/initial-dynamics.npz'
    cloud.volume.files[remote] = b'corrupt'
    with pytest.raises(ValueError, match='Downloaded artifact hash differs'): invoke(transport, root)
    assert not (root/'artifacts/initial-dynamics.npz').exists() and not (root/'audit.json').exists()
    assert len(cloud.calls) == 1


def test_failed_independent_audit_installs_no_comparison_views(transport, cloud, tmp_path, monkeypatch):
    root = tmp_path/'out'; invoke(transport, root)
    req = json.loads((root/'request.json').read_text()); cloud.result = fixture_result(req, cloud.volume)
    def reject(*args, **kwargs): raise ValueError('Independent neural reconstruction failed')
    monkeypatch.setattr(bridge, 'audit', reject)
    with pytest.raises(ValueError, match='Independent neural reconstruction failed'): invoke(transport, root)
    assert json.loads((root/'cloud-call.json').read_text())['status'] == 'pending'
    assert not (root/'audit.json').exists()
    for arm in cloud.result['report']['protocol']['arms']: assert not (root/arm).exists()
    assert len(cloud.calls) == 1


@pytest.fixture
def worker(transport, tmp_path, monkeypatch):
    req = request(transport[2]); root = tmp_path/'state/fly-debugger'
    source = root.parent/bridge.INPUTS/(transport[2]['bundle_sha256']+'.json')
    source.parent.mkdir(parents=True); source.write_bytes(transport[1].read_bytes())
    events = []; monkeypatch.setattr(bridge, 'verify_reference', lambda *a: None)
    def capture(p, reference, data, output, *, completed_study):
        assert events == ['commit']
        claim = json.loads((root/bridge.CLAIM).read_text())
        assert claim['status'] == 'claimed' and claim['call_id'] == 'fc-test'
        assert (completed_study/'report.json').exists() and reference == root/bridge.REFERENCE_RUN
        events.append('capture')
        protocol, _ = bridge.validate_request(req)
        for name in bridge.artifact_names(protocol):
            file = output/name; file.parent.mkdir(parents=True, exist_ok=True); file.write_text('fixture')
        return {'fixture_capture': True}
    monkeypatch.setattr(bridge, 'capture', capture)
    return req, root, events, lambda: events.append('commit')


def test_worker_commits_claim_before_capture_and_refuses_a_second_run(worker):
    req, root, events, commit = worker
    result = bridge.run_request(req, root, 'data', call_id='fc-test', input_id='in-test', commit=commit)
    assert result['status'] == 'selective_trace_completed' and events == ['commit', 'capture', 'commit']
    assert len(result['artifact_sha256']) == 175
    req['run_id'] = 'assay-selective-different-output'
    with pytest.raises(ValueError, match='already claimed'):
        bridge.run_request(req, root, 'data', call_id='fc-new', input_id='in-new', commit=commit)
    assert events.count('capture') == 1


def test_failed_native_capture_retains_claim_and_never_restarts(worker, monkeypatch):
    req, root, events, commit = worker
    def failed(*a, **kw):
        events.append('failed_capture'); raise TimeoutError('Native deadline')
    monkeypatch.setattr(bridge, 'capture', failed)
    with pytest.raises(TimeoutError):
        bridge.run_request(req, root, 'data', call_id='fc-test', input_id='in-test', commit=commit)
    claim = json.loads((root/bridge.CLAIM).read_text())
    assert claim['status'] == 'failed' and claim['error_type'] == 'TimeoutError'
    with pytest.raises(ValueError, match='already claimed'):
        bridge.run_request(req, root, 'data', call_id='fc-new', input_id='in-new', commit=commit)
    assert events == ['commit', 'failed_capture', 'commit']


def test_worker_bad_release_does_not_claim_or_construct(worker):
    req, root, events, commit = worker
    req['selective_plan']['study_report_sha256'] = '0'*64
    with pytest.raises(ValueError, match='report differs'):
        bridge.run_request(req, root, 'data', call_id='fc-test', input_id='in-test', commit=commit)
    assert not events and not (root/bridge.CLAIM).exists()


def test_worker_production_release_gate_rejects_synthetic_evidence(worker, monkeypatch):
    from paperlab.fly_selective_trace import require_completed_study
    monkeypatch.setattr('paperlab.fly_study_evidence_bundle.require_completed_study', require_completed_study)
    req, root, events, commit = worker
    with pytest.raises(ValueError, match='actual study'):
        bridge.run_request(req, root, 'data', call_id='fc-test', input_id='in-test', commit=commit)
    assert not events and not (root/bridge.CLAIM).exists()
