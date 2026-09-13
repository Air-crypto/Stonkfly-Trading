"""Read existing study 12 calls, download exact recordings, and audit them offline.

This module has no model-submission or cloud-mutation path. Observation timeouts
leave the owning handle intact. Full audits reconstruct arrays, never a brain.
"""
import argparse
import asyncio
from contextlib import aclosing
import fcntl
import hashlib
import json
import math
from pathlib import Path
import time

from .core import atomic_json, digest
from .fly_market_study import signature
from .fly_rate_cloud import verify_arming
from .fly_rate_inputs import require_seed, validate
from .fly_rate_protocol import chunk_name, chunk_order
from .fly_rate_schedule import DIRECTORY, REGISTRATION, aggregate, source_hashes
from .fly_rate_study import SOURCE_FILES, select_development
from .fly_recording_download import safe_target, transfer_files


def pin_bytes(path, raw):
    path = Path(path)
    safe_target(path.parent, path.name)
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError('Preserve differing saved evidence: '+str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = safe_target(path.parent, path.name+'.partial')
    partial.write_bytes(raw); partial.replace(path)


def pin_json(path, value):
    pin_bytes(path, (json.dumps(value, indent=2, allow_nan=False)+'\n').encode())


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_receipt(receipt, name, envelope, sources):
    r = envelope['plan']['registration']
    if (receipt.get('chunk') != name or receipt.get('plan_sha256') != envelope['sha256']
            or receipt.get('source_sha256') != signature(sources)
            or receipt.get('remote_path') != f'/state/{DIRECTORY}/chunks/{name}/artifacts'
            or receipt.get('dispatch') != 'dedicated-rate-worker-shared-lease'
            or not receipt.get('call_id') or not receipt.get('input_id')
            or receipt.get('status') not in ('claimed', 'failed', 'completed')
            or not finite(receipt.get('claimed_at')) or receipt['claimed_at'] < r['end']
            or (receipt['status'] == 'completed' and (not finite(receipt.get('completed_at'))
                or receipt['completed_at'] < receipt['claimed_at']))):
        raise ValueError('Chunk ownership, source, clock or path differs: '+name)


def validate_selection(selection, receipt, receipts, raw):
    development = [r for name, r in receipts.items() if name.startswith('development-')]
    tests = [r for name, r in receipts.items() if name.startswith('test-')]
    if (len(development) != 8 or not finite(receipt.get('selected_at'))
            or not receipt.get('call_id') or not receipt.get('input_id')
            or receipt['development_chunk_sha256'] != selection['development_chunk_sha256']
            or receipt['selection_sha256'] != hashlib.sha256(raw).hexdigest()
            or receipt['selected_at'] < max(r['completed_at'] for r in development)
            or any(receipt['selected_at'] > r['claimed_at'] for r in tests)):
        raise ValueError('Selection must follow completed development and precede every test claim')


def validate_summary(summary, receipt, envelope, sources, condition):
    stage, pool, arm = condition; name = chunk_name(*condition)
    if (summary.get('status') != 'paper_rate_chunk_completed' or summary.get('chunk') != name
            or summary.get('registration') != envelope['plan']['registration']
            or summary.get('plan_sha256') != envelope['sha256'] or summary.get('stage') != stage
            or summary.get('pool_index') != pool or summary.get('arm') != arm
            or summary.get('pool') != envelope['plan']['registration']['cohort'][pool]
            or summary.get('code_sha256') != {f: sources['paperlab/'+f] for f in SOURCE_FILES}):
        raise ValueError('Completed chunk summary differs: '+name)


def validate_return(result, receipt):
    if (result.get('status') != 'paper_rate_chunk_completed'
            or result.get('chunk') != receipt['chunk'] or result.get('receipt') != receipt
            or result.get('call_id') != receipt['call_id'] or result.get('input_id') != receipt['input_id']):
        raise ValueError('Completed owning call differs from its committed receipt')


def validate_arming_return(result, armed, registration):
    if (result.get('status') != 'paper_rate_collecting'
            or result.get('call_id') != armed['call_id'] or result.get('input_id') != armed['input_id']
            or any(result.get(k) != registration[k] for k in ('development_start', 'test_start', 'end'))):
        raise ValueError('Arming call did not return the registered future window')


async def existing_call(call_id):
    import modal
    try:
        return await modal.FunctionCall.from_id(call_id).get.aio(timeout=0)
    except (modal.exception.FunctionTimeoutError, modal.exception.OutputExpiredError):
        # Execution timeout and expired evidence are not observation timeouts.
        raise
    except (TimeoutError, modal.exception.TimeoutError):
        return None


class Reader:
    def __init__(self, volume):
        self.volume = volume

    async def raw(self, name, optional=False):
        import modal
        safe_target(Path('/metadata'), name)
        try:
            async with asyncio.timeout(25):
                parts = []
                async with aclosing(self.volume.read_file.aio('/'+DIRECTORY+'/'+name)) as stream:
                    async for block in stream: parts.append(block)
                return b''.join(parts)
        except (FileNotFoundError, modal.exception.NotFoundError):
            if optional: return None
            raise


def _pin_receipt(root, name, receipt, raw):
    fixed = ('call_id', 'input_id', 'chunk', 'plan_sha256', 'source_sha256', 'claimed_at', 'remote_path')
    for earlier in (root/'receipts').glob(name+'-*.json'):
        old = json.loads(earlier.read_text())
        if any(old.get(k) != receipt.get(k) for k in fixed):
            raise ValueError('A previously observed chunk was replaced by another owner')
        if old['status'] == 'completed' and receipt['status'] != 'completed':
            raise ValueError('A completed receipt regressed')
    pin_bytes(root/'receipts'/f"{name}-{receipt['status']}.json", raw)


async def _observe(registration_path, root, download):
    import modal
    volume = modal.Volume.from_name('fly-paper-lab-state', environment_name='main')
    reader = Reader(volume); r = json.loads(Path(registration_path).read_text())
    progress = {'observed_at': time.time(), 'completed_chunks': 0, 'downloaded_chunks': 0,
                'total_chunks': 16, 'cloud_submissions': 0, 'new_neural_observations': 0}

    def finish(status, **extra):
        result = {**progress, 'status': status, **extra}
        atomic_json(root/'progress.json', result); return result

    armed_raw = await reader.raw('armed.json', optional=True)
    if armed_raw is None:
        missed = await reader.raw('missed.json', optional=True)
        return finish('registration_missed' if missed else 'waiting_for_cloud_arming')
    names = ('preregistration.json', 'source-hashes.json')
    raw_values = await asyncio.gather(*(reader.raw(n) for n in names))
    raw_files = dict(zip(names, raw_values)); raw_files['armed.json'] = armed_raw
    files = {k: json.loads(v) for k, v in raw_files.items()}
    hashes = {k: hashlib.sha256(v).hexdigest() for k, v in raw_files.items()}
    check = verify_arming(files, hashes, registration_path, source_hashes())
    if not check['verified']: raise ValueError('Cloud registration or executed sources differ')
    for name, raw in raw_files.items(): pin_bytes(root/name, raw)
    armed = files['armed.json']; sources = files['source-hashes.json']
    arming_result = await existing_call(armed['call_id'])
    if arming_result is None: return finish('arming_call_pending', call_id=armed['call_id'])
    validate_arming_return(arming_result, armed, r)
    pin_json(root/'arming-call-result.json', arming_result)
    now = time.time()
    phase = ('waiting_for_development' if now < r['development_start'] else
             'development_collection' if now < r['test_start'] else
             'test_collection' if now < r['end'] else 'evaluation')
    progress.update(development_start=r['development_start'], test_start=r['test_start'], end=r['end'])
    sealed_raw = await reader.raw('sealing.json', optional=True)
    halt_raw = await reader.raw('halt.json', optional=True)
    if sealed_raw is None:
        if halt_raw:
            halt = json.loads(halt_raw); result = await existing_call(halt['call_id'])
            return finish('halt_call_pending' if result is None else 'halt_requires_inspection',
                          call_id=halt['call_id'], halt=halt)
        return finish(phase)
    sealed = json.loads(sealed_raw)
    if (sealed.get('registration_sha256') != armed['registration_sha256']
            or not finite(sealed.get('claimed_at')) or sealed['claimed_at'] < r['end']
            or not sealed.get('call_id') or not sealed.get('input_id')
            or sealed.get('status') not in ('claimed', 'failed', 'completed')):
        raise ValueError('Sealing ownership or endpoint differs')
    if sealed['status'] != 'completed':
        result = await existing_call(sealed['call_id'])
        return finish('sealing_call_pending' if result is None else 'sealing_receipt_unresolved',
                      call_id=sealed['call_id'], receipt=sealed)
    pin_bytes(root/'sealing.json', sealed_raw)
    plan_raw = await reader.raw('plan.json'); envelope = json.loads(plan_raw); p = validate(envelope)
    if p['registration'] != r or sealed['plan_sha256'] != envelope['sha256']:
        raise ValueError('Sealed inputs differ from the witnessed registration')
    pin_bytes(root/'plan.json', plan_raw)
    conditions = chunk_order(); names = [chunk_name(*c) for c in conditions]
    raw_receipts = await asyncio.gather(*(reader.raw('chunks/'+n+'/receipt.json', optional=True) for n in names))
    receipts = {}; summaries = {}; selection = None; snapshots_checked = False
    for i, (condition, name, raw) in enumerate(zip(conditions, names, raw_receipts)):
        if raw is None:
            if any(v is not None for v in raw_receipts[i+1:]): raise ValueError('Chunks exist beyond a missing condition')
            if list((root/'receipts').glob(name+'-*.json')): raise ValueError('A saved receipt disappeared from cloud state')
            break
        receipt = json.loads(raw); validate_receipt(receipt, name, envelope, sources)
        _pin_receipt(root, name, receipt, raw)
        if receipt['call_id'] in {v['call_id'] for v in receipts.values()}:
            raise ValueError('Different chunks share an owning call')
        if receipt['status'] != 'completed' and any(v is not None for v in raw_receipts[i+1:]):
            raise ValueError('Chunks exist beyond an unresolved condition')
        result = await existing_call(receipt['call_id'])
        if result is None:
            return finish('chunk_worker_return_pending' if receipt['status'] == 'completed' else 'chunk_call_pending',
                          chunk=name, call_id=receipt['call_id'])
        if receipt['status'] != 'completed':
            return finish('chunk_receipt_unresolved', chunk=name, call_id=receipt['call_id'], receipt=receipt)
        validate_return(result, receipt); pin_json(root/'call-results'/(name+'.json'), result)
        remote = 'chunks/'+name+'/artifacts'
        raw_summary = await reader.raw(remote+'/summary.json')
        if hashlib.sha256(raw_summary).hexdigest() != receipt['summary_sha256']:
            raise ValueError('Completed summary bytes differ')
        summary = json.loads(raw_summary); validate_summary(summary, receipt, envelope, sources, condition)
        destination = root/'chunks'/name; pin_bytes(destination/'summary.json', raw_summary)
        receipts[name] = receipt; summaries[name] = summary; progress['completed_chunks'] = len(summaries)
        if condition[0] == 'test':
            raw_selection, raw_receipt = await asyncio.gather(reader.raw('selection.json'), reader.raw('selection-receipt.json'))
            selection, choice_receipt = json.loads(raw_selection), json.loads(raw_receipt)
            development = {k: v for k, v in summaries.items() if v['stage'] == 'development'}
            if select_development(envelope, development) != selection or summary['selection'] != selection:
                raise ValueError('Test selection differs from replayed development ledgers')
            validate_selection(selection, choice_receipt, receipts, raw_selection)
            pin_bytes(root/'selection.json', raw_selection); pin_bytes(root/'selection-receipt.json', raw_receipt)
        if download:
            if not snapshots_checked:
                from .fly_rate_price_audit import audit_prices
                transferred = await transfer_files(volume, '/state/'+DIRECTORY, root,
                    {'universe.db': p['snapshot_sha256'], 'news.db': p['news_snapshot_sha256']},
                    label='sealed-inputs', timeout=60, attempts=3, files_at_once=2)
                atomic_json(root/'downloads/sealed-inputs.json', transferred)
                pin_json(root/'price-audit.json', audit_prices(envelope, root/'universe.db'))
                snapshots_checked = True
            transferred = await transfer_files(volume, receipt['remote_path'], destination,
                summary['artifact_sha256'], label=name, timeout=60, attempts=3, files_at_once=4)
            atomic_json(root/'downloads'/(name+'.json'), {**transferred, 'call_id': receipt['call_id'],
                'summary_sha256': receipt['summary_sha256']})
            progress['downloaded_chunks'] += 1
    if len(summaries) == 16:
        expected = aggregate(envelope, summaries, selection)
        cloud_raw = await reader.raw('summary.json', optional=True)
        if cloud_raw is None: return finish('captured_waiting_for_cloud_summary')
        if json.loads(cloud_raw) != expected: raise ValueError('Cloud aggregate differs from its completed chunks')
        pin_bytes(root/'cloud-summary.json', cloud_raw)
        return finish('downloaded_pending_audit' if download else 'captured_pending_download',
                      selection=selection['selected'], total_equity=expected['total_equity'])
    return finish(phase)


def observe(registration, output, *, download=False):
    root = Path(output); safe_target(root, 'observe.lock'); root.mkdir(parents=True, exist_ok=True)
    with (root/'observe.lock').open('a') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc: raise ValueError('Another reader owns this output') from exc
        try: return asyncio.run(_observe(registration, root, download))
        except Exception as exc:
            atomic_json(root/'observation-error.json', {'observed_at': time.time(), 'type': type(exc).__name__,
                'message': str(exc)[:400], 'cloud_submissions': 0,
                'interpretation': 'Observation stopped. This does not authorize a replacement call or establish cloud termination.'})
            raise


def audit_saved(root, data, output, *, registration=None):
    from .fly_rate_audit import audit_chunk
    from .fly_rate_price_audit import audit_prices
    from .fly_view_projection import audit_view, load_graph
    require_seed(); root, output = Path(root), Path(output)
    registration = Path(registration) if registration is not None else Path(__file__).resolve().parents[1]/'reports'/REGISTRATION
    if output.exists(): raise ValueError('Preserve the earlier audit; use a new output directory')
    safe_target(output, 'report.json')
    read = lambda name: json.loads((root/name).read_text())
    sources = read('source-hashes.json')
    if sources != source_hashes(): raise ValueError('Executed study sources changed before the full audit')
    envelope = read('plan.json'); p = validate(envelope); r = p['registration']
    files = {name: read(name) for name in ('armed.json', 'preregistration.json', 'source-hashes.json')}
    hashes = {name: digest(root/name) for name in files}
    if not verify_arming(files, hashes, registration, sources)['verified']:
        raise ValueError('Saved cloud witness differs')
    if files['preregistration.json'] != r: raise ValueError('Saved plan registration differs')
    validate_arming_return(read('arming-call-result.json'), files['armed.json'], r)
    sealed = read('sealing.json')
    if (sealed.get('status') != 'completed' or sealed.get('plan_sha256') != envelope['sha256']
            or sealed.get('registration_sha256') != hashes['preregistration.json']
            or not finite(sealed.get('claimed_at')) or sealed['claimed_at'] < r['end']
            or not sealed.get('call_id') or not sealed.get('input_id')):
        raise ValueError('Saved sealing receipt differs')
    receipts = {}; summaries = {}
    for condition in chunk_order():
        name = chunk_name(*condition); receipt = read('receipts/'+name+'-completed.json')
        validate_receipt(receipt, name, envelope, sources)
        validate_return(read('call-results/'+name+'.json'), receipt)
        path = root/'chunks'/name/'summary.json'
        if digest(path) != receipt['summary_sha256']: raise ValueError('Saved completed summary differs')
        summary = json.loads(path.read_text()); validate_summary(summary, receipt, envelope, sources, condition)
        for file, sha in summary['artifact_sha256'].items():
            if digest(safe_target(path.parent, file)) != sha: raise ValueError('Saved full recording differs: '+file)
        receipts[name] = receipt; summaries[name] = summary
    if len({v['call_id'] for v in receipts.values()}) != 16: raise ValueError('Completed calls are not distinct')
    selection = select_development(envelope, {k: v for k, v in summaries.items() if v['stage'] == 'development'})
    if read('selection.json') != selection: raise ValueError('Saved selection differs')
    validate_selection(selection, read('selection-receipt.json'), receipts, (root/'selection.json').read_bytes())
    expected = aggregate(envelope, summaries, selection)
    if read('cloud-summary.json') != expected: raise ValueError('Saved aggregate differs')
    identities = {(v['native_build']['binary_sha256'], v['artifact_sha256']['initial-dynamics.npz'],
                   v['artifact_sha256']['pristine-memory.npz']) for v in summaries.values()}
    if len(identities) != 1: raise ValueError('Conditions differ in native build or fresh starting state')
    prices = audit_prices(envelope, root/'universe.db')
    if digest(root/'news.db') != p['news_snapshot_sha256']: raise ValueError('Saved news snapshot differs')
    output.mkdir(parents=True); atomic_json(output/'price-audit.json', prices)
    audits = {}; projections = {}; graph = None
    for condition in chunk_order():
        name = chunk_name(*condition); summary = summaries[name]; recording = root/'chunks'/name
        checked, view = audit_chunk(envelope, summary, recording, data)
        count = sum(row['event'] is not None for row in summary['outcome']['rows']); v = checked['verification']
        if (checked['status'] != 'paper_rate_chunk_audited' or checked['chunk'] != name
                or checked['plan_sha256'] != envelope['sha256']
                or checked['artifact_sha256'] != summary['artifact_sha256']
                or checked['executed_source_sha256'] != summary['code_sha256']
                or v['observations'] != count or v['native_bins'] != count*50 or v['decision_slots'] != 25
                or any(v.get(k) is not True for k in ('ledger_replayed', 'feedback_reconstructed', 'all_boundaries_verified', 'all_weights_exact'))):
            raise ValueError('Incomplete independent chunk audit')
        if count:
            if view is None: raise ValueError('Missing recorded view')
            if graph is None: graph = load_graph(data)
            projection = audit_view(view, recording/'trace', graph, observations=count)
            folder = output/'views'/name; pin_json(folder/'view.json', view); pin_json(folder/'report.json', view['report'])
            for file in (recording/'trace').glob('step-*.npz'):
                (folder/file.name).hardlink_to(file)
        else:
            if view is not None: raise ValueError('Unobserved phase has a fabricated view')
            projection = {'observations': 0, 'bins': 0, 'view_status': 'no_neural_observations'}
        pin_json(output/'audits'/(name+'.json'), checked); pin_json(output/'projections'/(name+'.json'), projection)
        audits[name] = digest(output/'audits'/(name+'.json')); projections[name] = digest(output/'projections'/(name+'.json'))
        print(json.dumps({'event': 'rate_recording_audited', 'chunk': name, 'observations': count}), flush=True)
    expected.update(status='paper_rate_study_audited', audited=True, cloud_registration=files['armed.json'],
        selection_receipt=read('selection-receipt.json'), price_audit=prices, audit_sha256=audits,
        projection_sha256=projections, executed_source_sha256=sources,
        audit_source_sha256={name: digest(Path(__file__).with_name(name)) for name in ('fly_rate_results.py', 'fly_view_projection.py')},
        phase_diagnostics={key: {arm: {stage: summaries[chunk_name(stage, pool, arm)]['outcome'] for stage in ('development', 'test')}
                                for arm in r['arms']} for pool, key in enumerate(r['cohort'])},
        validation_only=any('synthetic' in t['source'] for series in p['series'].values() for t in series),
        verification={'all_ledgers_and_feedback_reconstructed': True, 'all_full_bin_audits_passed': True,
                      'all_raw_prices_reconstructed': True, 'all_news_clocks_verified': True,
                      'all_view_projections_verified': True, 'source_and_native_builds_match': True,
                      'selection_precedes_every_test_chunk': True, 'distinct_completed_worker_calls': 16},
        new_neural_observations=0, cloud_submissions=0, policy_promoted=False,
        interpretation='Four-arm registered paper comparison with independently reconstructed inputs, execution, feedback, learning bins and viewer curves. Two pools and four hours do not establish monthly profitability. Insufficient test coverage is inconclusive; no policy promoted.')
    atomic_json(output/'report.json', expected); return expected


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('action', choices=('observe', 'audit'))
    p.add_argument('--registration', type=Path); p.add_argument('--root', type=Path)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--download', action='store_true')
    p.add_argument('--fly-data', type=Path); a = p.parse_args()
    if a.action == 'observe':
        if a.registration is None: p.error('observe requires --registration')
        result = observe(a.registration, a.out, download=a.download)
    else:
        if a.root is None or a.fly_data is None: p.error('audit requires --root and --fly-data')
        result = audit_saved(a.root, a.fly_data, a.out, registration=a.registration)
    print(json.dumps({k: result[k] for k in ('status', 'completed_chunks', 'downloaded_chunks', 'selection',
        'total_equity', 'validation_only', 'cloud_submissions') if k in result}, indent=2))


if __name__ == '__main__': main()
