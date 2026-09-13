"""Prepared bridge for the post-study selective assay; not deployed during study 11.

The unscheduled selective worker shares the existing worker lease and budget.
A saved local call and a persistent cloud claim each prevent retrying an
uncertain native execution.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import re
import tempfile
import time
import uuid

from .core import atomic_json, digest
from .fly_market_study import signature
from .fly_selective_trace import source_hashes, validate, verify_reference, run as capture
from .fly_selective_trace_audit import audit
from .fly_study_evidence_bundle import pack, unpack
from .fly_recording_download import transfer_files
from .budget import reserve, settle

REFERENCE_RUN = 'assay-credit-d4cef30fe9bc46be8d13da36a12135b9'
CLAIM = 'selective-trace-01-claim.json'
INPUTS = 'fly-selective-inputs'
ROOT_ARTIFACTS = ('initial-dynamics.npz', 'trained-memory.npz', 'pristine-memory.npz',
                  'full-weight-reference.npz', 'neuron-ids.npz', 'circuit.npz')
APP_NAME = 'fly-paper-selective-01'


def artifact_names(protocol):
    names = [*ROOT_ARTIFACTS, 'summary.json']
    for arm in protocol['arms']:
        names.extend(arm+'/'+name for name in ('view.json', 'report.json', 'initial-memory.npz',
                                             'end-01.npz', 'end-02.npz'))
        for i in range(1, 4):
            names.extend(f'{arm}/{name}' for name in
                         (f'boundary-{i:02}.npz', f'step-{i:02}.npz', f'input-{i:02}.png'))
    return tuple(names)


def validate_request(request):
    if (not isinstance(request, dict) or set(request) != {'run_id', 'selective_plan'}
            or not isinstance(request['run_id'], str)
            or not re.fullmatch(r'assay-selective-[a-z0-9-]{1,64}', request['run_id'])):
        raise ValueError('Expected a selective assay request and safe run ID')
    plan = request['selective_plan']
    if not isinstance(plan, dict) or set(plan) != {
            'payload', 'bundle_sha256', 'study_report_sha256', 'source_sha256'}:
        raise ValueError('Expected the exact selective request fields')
    protocol, parent, _, _ = validate(plan['payload'])
    for key in ('bundle_sha256', 'study_report_sha256'):
        if not isinstance(plan[key], str) or not re.fullmatch('[0-9a-f]{64}', plan[key]):
            raise ValueError('Invalid study evidence digest')
    if plan['source_sha256'] != source_hashes():
        raise ValueError('Prepared and executing assay sources differ')
    return protocol, parent


def check_bundle(path, expected_bundle, expected_report):
    with tempfile.TemporaryDirectory(prefix='selective-release-') as temporary:
        report = unpack(path, Path(temporary)/'study', expected_bundle)
    if report['report_sha256'] != expected_report:
        raise ValueError('Completed study report differs from request')
    return report


def exclusive_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def run_budgeted_request(request, state, data, *, call_id, input_id, commit):
    """Called only inside the shared lease by the non-preemptible worker."""
    validate_request(request)
    state = Path(state); root = state/'fly-debugger'; plan = request['selective_plan']
    if not call_id or not input_id:
        raise ValueError('The worker must supply its actual Modal call and input IDs')
    if (root/CLAIM).exists() or (root/request['run_id']).exists():
        raise ValueError('Selective assay already claimed; inspect its existing call')
    # Reject incomplete evidence before reserving compute or constructing a model.
    check_bundle(state/INPUTS/(plan['bundle_sha256']+'.json'),
                 plan['bundle_sha256'], plan['study_report_sha256'])
    reservation = reserve(state/'budget.json', True, seconds=1800,
                          startup_seconds=30, memory_gib=8, limit_override=25)
    if reservation is None:
        return {'status': 'budget_stopped'}
    # 610 seconds at 3x provider price, retaining the budget module's 2x margin.
    reservation['rate'] *= 3
    reservation['startup_seconds'] = 10
    commit()
    try:
        result = run_request(request, root, data, call_id=call_id, input_id=input_id, commit=commit)
        result['budget'] = {**settle(state/'budget.json', reservation,
            time.time()-reservation['started']), 'reserved_usd': reservation['reserve'],
            'nonpreemptible': True, 'price_multiplier': 3}
        atomic_json(root/request['run_id']/'cloud-result.json', result)
        commit()
        return result
    except BaseException:
        commit()  # Preserve the worst-case reservation and any claim after failure.
        raise


def run_request(request, root, data, *, call_id, input_id, commit):
    """Future worker adapter: release validation, durable claim, then one capture."""
    protocol, parent = validate_request(request)
    root = Path(root); plan = request['selective_plan']; output = root/request['run_id']
    claim_path = root/CLAIM
    if not call_id or not input_id:
        raise ValueError('The budgeted worker must supply its Modal call and input IDs')
    if claim_path.exists() or output.exists():
        raise ValueError('Selective assay already claimed; inspect the original call, never restart')
    with tempfile.TemporaryDirectory(prefix='selective-release-') as temporary:
        study = Path(temporary)/'study'
        release = unpack(root.parent/INPUTS/(plan['bundle_sha256']+'.json'), study, plan['bundle_sha256'])
        if release['report_sha256'] != plan['study_report_sha256']:
            raise ValueError('Completed study report differs from request')
        verify_reference(parent, root/REFERENCE_RUN)
        claim = {'status': 'claimed', 'call_id': call_id, 'input_id': input_id,
                 'run_id': request['run_id'], 'request_sha256': signature(request),
                 'claimed_at': time.time(), 'study_report_sha256': plan['study_report_sha256']}
        exclusive_json(claim_path, claim)
        commit()  # Persist before any native construction or propagation.
        try:
            report = capture(plan['payload'], root/REFERENCE_RUN, data, output, completed_study=study)
            hashes = {name: digest(output/name) for name in artifact_names(protocol)}
            result = {'status': 'selective_trace_completed', 'run_id': request['run_id'],
                      'remote_path': str(output), 'report': report, 'artifact_sha256': hashes,
                      'request_sha256': signature(request), 'call_id': call_id, 'input_id': input_id}
            claim.update(status='completed', completed_at=time.time(), artifact_sha256=hashes)
            atomic_json(claim_path, claim)
            commit()
            return result
        except Exception as exc:
            claim.update(status='failed', failed_at=time.time(), error_type=type(exc).__name__)
            atomic_json(claim_path, claim)
            commit()
            raise


def terminal_result(result, request, receipt):
    protocol, _ = validate_request(request)
    base = '/state/fly-debugger/'+request['run_id']
    report = result.get('report', {})
    if (result.get('status') != 'selective_trace_completed'
            or result.get('run_id') != request['run_id'] or result.get('remote_path') != base
            or result.get('call_id') != receipt['call_id'] or not result.get('input_id')
            or result.get('request_sha256') != signature(request)
            or report.get('status') != 'selective_trace_captured_pending_audit'
            or report.get('protocol') != protocol
            or report.get('code_sha256') != request['selective_plan']['source_sha256']
            or report.get('completed_study_sha256') != request['selective_plan']['study_report_sha256']
            or result.get('budget', {}).get('monthly_limit_usd') != 25
            or result['budget'].get('nonpreemptible') is not True
            or result['budget'].get('price_multiplier') != 3
            or not 0 < result['budget'].get('reserved_usd', 0) <= .1608936):
        raise ValueError('Terminal result, budget, ownership or executed evidence differs; do not resubmit')
    hashes = result.get('artifact_sha256')
    if not isinstance(hashes, dict) or set(hashes) != set(artifact_names(protocol)):
        raise ValueError('Incomplete or unexpected artifact manifest')
    if any(not isinstance(sha, str) or not re.fullmatch('[0-9a-f]{64}', sha) for sha in hashes.values()):
        raise ValueError('Invalid artifact digest')
    return base, report, hashes


def download(volume, base, root, hashes):
    """Bounded read retries follow terminal_result's exact ownership check."""
    return asyncio.run(transfer_files(volume, base, root, hashes, label=Path(base).name))


def idle_worker(modal):
    function = modal.Function.from_name(APP_NAME, 'worker', environment_name='main')
    main = modal.Function.from_name('fly-paper-lab', 'worker', environment_name='main')
    owner = modal.Dict.from_name('fly-paper-lab-writers', environment_name='main').get('worker')
    stats = [fn.get_current_stats() for fn in (function, main)]
    if owner is not None or any(s.num_total_runners or s.num_running_inputs or s.backlog for s in stats):
        raise RuntimeError('Worker has active work; no selective assay submitted')
    volume = modal.Volume.from_name('fly-paper-lab-state', environment_name='main')
    try:
        existing = b''.join(volume.read_file('/fly-debugger/'+CLAIM))
    except (FileNotFoundError, modal.exception.NotFoundError):
        existing = None
    if existing is not None:
        raise RuntimeError('Selective assay already claimed remotely; inspect the original call')
    return function, volume


def upload_bundle(modal, volume, path, expected):
    remote = '/'+INPUTS+'/'+expected+'.json'
    sha = hashlib.sha256()
    try:
        for block in volume.read_file(remote): sha.update(block)
    except (FileNotFoundError, modal.exception.NotFoundError):
        with volume.batch_upload() as batch:
            batch.put_file(path, remote)
        return
    if sha.hexdigest() != expected:
        raise ValueError('Existing remote evidence bundle differs')


def cloud_run(payload, output, reference, data, *, completed_study):
    """One submission after activation; later invocations observe the same handle."""
    import modal
    _, parent, _, _ = validate(payload)
    root = Path(output); receipt_path = root/'cloud-call.json'
    if not receipt_path.exists():
        # Even a completed synthetic fixture fails this pack's production gate.
        with tempfile.TemporaryDirectory(prefix='selective-prepare-') as temporary:
            bundle = Path(temporary)/'study-evidence.json'
            release = pack(completed_study, bundle)
            verify_reference(parent, reference)
            idle_worker(modal)
            # Refuse unidentified pre-existing work, including partially prepared output.
            root.mkdir(parents=True, exist_ok=False)
            (root/'study-evidence.json').write_bytes(bundle.read_bytes())
            request = {'run_id': 'assay-selective-'+uuid.uuid4().hex, 'selective_plan': {
                'payload': payload, 'bundle_sha256': release['bundle_sha256'],
                'study_report_sha256': release['report_sha256'], 'source_sha256': source_hashes()}}
            validate_request(request)
            atomic_json(root/'request.json', request)
            receipt = {'status': 'prepared', 'request_sha256': signature(request),
                       'run_id': request['run_id'], 'prepared_at': time.time()}
            exclusive_json(receipt_path, receipt)
    receipt = json.loads(receipt_path.read_text())
    request = json.loads((root/'request.json').read_text())
    validate_request(request)
    if (receipt['request_sha256'] != signature(request)
            or request['selective_plan']['payload'] != payload or receipt['run_id'] != request['run_id']):
        raise ValueError('Saved submission belongs to different evidence')
    if receipt['status'] not in ('prepared', 'submitting', 'pending', 'completed'):
        raise ValueError('Unknown saved call state')
    if receipt['status'] != 'prepared' and 'call_id' not in receipt:
        raise RuntimeError('Uncertain submission; inspect the original call, never resubmit')
    plan = request['selective_plan']
    check_bundle(root/'study-evidence.json', plan['bundle_sha256'], plan['study_report_sha256'])
    if receipt['status'] == 'prepared':
        if 'call_id' in receipt:
            raise ValueError('Prepared submission unexpectedly has a call ID')
        function, volume = idle_worker(modal)
        upload_bundle(modal, volume, root/'study-evidence.json', plan['bundle_sha256'])
        # Uploads can safely resume while prepared. Only the paid RPC creates an
        # uncertain outcome; persist that boundary before crossing it.
        receipt.update(status='submitting')
        atomic_json(receipt_path, receipt)
        call = function.spawn(debug=request)
        receipt.update(status='pending', call_id=call.object_id)
        atomic_json(receipt_path, receipt)
    if receipt['status'] == 'completed':
        for name, sha in receipt['local_artifact_sha256'].items():
            if digest(root/name) != sha:
                raise ValueError('Completed local evidence differs: '+name)
        return receipt
    print('Observing saved call '+receipt['call_id'], flush=True)
    try:
        result = modal.FunctionCall.from_id(receipt['call_id']).get(timeout=50)
    except TimeoutError:
        print('Same call remains pending; repeat this observer to reattach.', flush=True)
        return receipt
    atomic_json(root/'cloud-result.json', result)
    base, report, hashes = terminal_result(result, request, receipt)
    volume = modal.Volume.from_name('fly-paper-lab-state', environment_name='main')
    transfer = download(volume, base, root/'artifacts', hashes)
    atomic_json(root/'download.json', transfer)
    if json.loads((root/'artifacts/summary.json').read_text()) != report:
        raise ValueError('Stored and returned native summaries differ')
    with tempfile.TemporaryDirectory(prefix='selective-audit-') as temporary:
        study = Path(temporary)/'study'
        unpack(root/'study-evidence.json', study, plan['bundle_sha256'])
        verified, views = audit(report, payload, root/'artifacts', reference, data, completed_study=study)
    atomic_json(root/'audit.json', verified)
    atomic_json(root/'summary.json', report)
    local = {'artifacts/'+name: sha for name, sha in hashes.items()}
    for name, view in views.items():
        folder = root/name
        atomic_json(folder/'view.json', view)
        atomic_json(folder/'report.json', view['report'])
        atomic_json(folder/'remote.json', {'remote_path': base+'/'+name, 'call_id': receipt['call_id']})
        for file in ('view.json', 'report.json', 'remote.json'):
            local[name+'/'+file] = digest(folder/file)
    local.update({name: digest(root/name) for name in ('audit.json', 'summary.json', 'download.json')})
    receipt.update(status='completed', budget=result['budget'], local_artifact_sha256=local)
    atomic_json(receipt_path, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('payload', 'out', 'reference-recordings', 'fly-data', 'completed-study'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = cloud_run(json.loads(args.payload.read_text()), args.out, args.reference_recordings,
                       args.fly_data, completed_study=args.completed_study)
    print(json.dumps({k: result[k] for k in ('status', 'run_id', 'call_id')}, indent=2))


if __name__ == '__main__':
    main()
