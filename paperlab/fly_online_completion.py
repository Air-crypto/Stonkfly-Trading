"""Separate, non-preemptible completion attempt after recovery 02 was interrupted.

Retain six audited conditions. Preserve the interrupted seventh receipt and all
partial arrays; never resume or overwrite them. Ten new captures start fresh in
this namespace, at three times the CPU/RAM rate, under a $2 compute allowance.
"""
import json
from pathlib import Path
import time

from .budget import reserve, settle
from .core import atomic_json, digest
from .fly_market_study import signature
from .fly_online_protocol import chunk_name, chunk_order
from .fly_online_recovery import (DIRECTORY as PARENT, require, run_seeded_chunk,
                                  validate_protocol as validate_parent, verify_reference)
from .fly_online_schedule import DIRECTORY as ORIGINAL, aggregate
from .fly_online_study import SOURCE_FILES, select_development

DIRECTORY = 'registered-paper-11-completion-01'
PARENT_PROTOCOL_SHA256 = 'e72488706d138dea0f0c4e2b74e497dddd81bfd9606c83bd3b25d628a32fa0b4'
POLICY = ('Separate completion after Modal preemption interrupted recovery 02 condition seven. '
    'Preserve every original, recovery 01 and recovery 02 receipt, including the incomplete seventh recording. '
    'Reuse exactly six fully audited conditions; do not use or resume partial weights, accounts or dynamics. '
    'Capture the remaining ten original conditions once from their specified fresh starts in a new namespace. '
    'Keep the original 56 execution sources, sealed market/news inputs, full graph, decoder and learning protocol unchanged. '
    'Use fresh seed-zero Python subprocesses, non-preemptible two-CPU eight-GiB workers, 600-second limits and no retries. '
    'Reserve the full three-times CPU/RAM multiplier plus the existing two-times safety factor before capture. '
    'Share the original worker lease and $25 monthly budget; at most $2 additional conservative compute. '
    'Persist original-rule development selection before any held-out capture. Never repeat a claimed or failed completion chunk. '
    'Require all sixteen independent full-bin audits and disclose both amendments; no policy promotion.')


def validate_protocol(p):
    require(set(p) == {'version', 'created_at', 'policy', 'parent_protocol_sha256',
        'preemption_report_sha256', 'retained', 'code_sha256'}, 'Completion protocol fields differ')
    require(p['version'] == 1 and p['policy'] == POLICY and p['parent_protocol_sha256'] == PARENT_PROTOCOL_SHA256,
        'Completion amendment differs')
    require(list(p['retained']) == [chunk_name(*c) for c in chunk_order()[:6]],
        'Retain exactly six audited conditions')
    require(set(p['code_sha256']) == {'paperlab/fly_online_completion.py', 'completion_cloud.py'},
        'Completion source manifest differs')
    hashes = [p['preemption_report_sha256'], *p['code_sha256'].values()]
    for pins in p['retained'].values():
        require(set(pins) == {'receipt_sha256', 'summary_sha256', 'projection_sha256'}, 'Retained evidence fields differ')
        hashes.extend(pins.values())
    require(all(isinstance(h, str) and len(h) == 64 and all(c in '0123456789abcdef' for c in h) for h in hashes),
        'Invalid completion evidence hash')
    require(type(p['created_at']) in (int, float) and 1789268703 < p['created_at'] < 1800000000,
        'Completion must follow the preemption')
    return p


def reference(state, specifications, evidence, code_root):
    state, evidence, code_root = map(Path, (state, evidence, code_root))
    p = validate_protocol(json.loads((evidence/'completion-protocol.json').read_text()))
    require(digest(evidence/'protocol.json') == PARENT_PROTOCOL_SHA256, 'Parent recovery protocol changed')
    parent = validate_parent(json.loads((evidence/'protocol.json').read_text()))
    envelope, sources, news = verify_reference(parent, state/ORIGINAL, specifications, evidence, code_root)
    for file, sha in p['code_sha256'].items():
        require(digest(code_root/file) == sha, 'Completion execution source changed: ' + file)
    failure_path = evidence/'fly-online-recovery-preemption-11.json'
    require(digest(failure_path) == p['preemption_report_sha256'], 'Preemption evidence changed')
    failure = json.loads(failure_path.read_text()); old = state/PARENT
    interrupted = chunk_name(*chunk_order()[6])
    require(digest(old/'chunks'/interrupted/'receipt.json') == failure['receipt_sha256']
        and json.loads((old/'chunks'/interrupted/'receipt.json').read_text()) == failure['receipt']
        and failure['terminal_result'] == {'status': 'writer_busy', 'writer': 'worker'}
        and failure['provider_event'] == 'Container terminated due to preemption'
        and failure['recovery_app_stopped'] is True and failure['observed_at'] < p['created_at'],
        'Interrupted recovery ownership differs')
    require(not (old/'summary.json').exists() and not (old/'selection.json').exists()
        and not (old/'chunks'/interrupted/'artifacts/summary.json').exists()
        and not any((old/'chunks'/chunk_name(*c)).exists() for c in chunk_order()[7:]),
        'The interrupted recovery continued or completed unexpectedly')
    for i, (name, pins) in enumerate(p['retained'].items()):
        folder = state/(ORIGINAL if i < 2 else PARENT)/'chunks'/name
        projection = evidence/'retained'/f'{name}.json'
        require(digest(folder/'receipt.json') == pins['receipt_sha256']
            and digest(folder/'artifacts/summary.json') == pins['summary_sha256']
            and digest(projection) == pins['projection_sha256'], 'Retained recording changed: ' + name)
        audit = json.loads(projection.read_text())
        require(audit['status'] == 'online_view_projection_audited' and audit['validation_only'] is False
            and audit['summary_sha256'] == pins['summary_sha256']
            and audit['projection']['all_topology_verified'] and audit['projection']['all_plotted_series_verified'],
            'Retained recording is not independently audited: ' + name)
    return p, parent, envelope, sources, news


def retained(state, p, parent, envelope, sources):
    state = Path(state); completed = {}; receipts = {}
    for i, condition in enumerate(chunk_order()):
        name = chunk_name(*condition); namespace = ORIGINAL if i < 2 else PARENT if i < 6 else DIRECTORY
        folder = state/namespace/'chunks'/name
        if not (folder/'receipt.json').exists():
            require(i >= 6 and not folder.exists(), 'Missing retained receipt or unclaimed artifacts')
            require(not any((state/DIRECTORY/'chunks'/chunk_name(*c)).exists() for c in chunk_order()[i+1:]),
                'Completion skipped a condition')
            return completed, receipts, condition, None
        receipt = json.loads((folder/'receipt.json').read_text())
        require(receipt['chunk'] == name and receipt['plan_sha256'] == envelope['sha256']
            and receipt['source_sha256'] == signature(sources) and receipt.get('call_id') and receipt.get('input_id')
            and receipt['remote_path'] == str(folder/'artifacts'), 'Completion ownership differs')
        if i >= 2:
            require(receipt['amendment_sha256'] == signature(parent if i < 6 else p), 'Capture amendment differs')
        if receipt['status'] != 'completed':
            require(i >= 6 and receipt['status'] in ('claimed', 'failed'), 'Incomplete retained condition')
            return completed, receipts, None, receipt
        require(digest(folder/'artifacts/summary.json') == receipt['summary_sha256'], 'Completed summary changed')
        summary = json.loads((folder/'artifacts/summary.json').read_text())
        require(summary['status'] == 'paper_online_chunk_completed' and summary['chunk'] == name
            and summary['plan_sha256'] == envelope['sha256']
            and summary['code_sha256'] == {f: sources['paperlab/'+f] for f in SOURCE_FILES}, 'Capture execution differs')
        require(receipt['call_id'] not in {r['call_id'] for r in receipts.values()}, 'Reused owning call')
        completed[name] = summary; receipts[name] = receipt
    return completed, receipts, None, None


def reservation(path):
    # 1,830 base-rate seconds reserve 610 actual seconds at the 3x price.
    # Settlement then uses the same 3x rate and the actual duration + 10 s.
    value = reserve(path, True, seconds=1800, startup_seconds=30, memory_gib=8, limit_override=25)
    if value is not None:
        value['rate'] *= 3
        value['startup_seconds'] = 10
        value['nonpreemptible_price_multiplier'] = 3
    return value


def execute(state, specifications, evidence, code_root, *, call_id, input_id, commit, run=run_seeded_chunk):
    state = Path(state); root = state/DIRECTORY
    if (root/'halt.json').exists():
        return {'status': 'completion_halted', 'receipt': json.loads((root/'halt.json').read_text())}
    if (root/'summary.json').exists():
        return {'status': 'paper_online_completion_captured', 'summary_sha256': digest(root/'summary.json')}
    require(call_id and input_id, 'Completion requires an owning cloud input')
    p, parent, envelope, sources, news = reference(state, specifications, evidence, code_root)
    root.mkdir(parents=True, exist_ok=True)
    if (root/'protocol.json').exists():
        require(json.loads((root/'protocol.json').read_text()) == p, 'Saved completion protocol changed')
    else:
        atomic_json(root/'protocol.json', p)
        atomic_json(root/'input-check.json', {'news': news, 'seed': '0', 'interpreter_start': 'fresh-subprocess',
            'verified_hash_probe': 7584921261715552910, 'original_inputs_changed': False,
            'call_id': call_id, 'input_id': input_id, 'checked_at': time.time()})
        commit()
    completed, receipts, condition, unresolved = retained(state, p, parent, envelope, sources)
    if unresolved:
        return {'status': 'completion_chunk_unresolved', 'receipt': unresolved, 'completed_chunks': len(completed)}
    development = {k:v for k,v in completed.items() if v['stage'] == 'development'}; selection = None
    if len(development) == 8:
        from .fly_online_cloud import validate_selection_receipt
        selection = select_development(envelope, development)
        if (root/'selection.json').exists():
            require(json.loads((root/'selection.json').read_text()) == selection, 'Saved selection changed')
        else:
            require(len(completed) == 8, 'Test preceded development selection')
            atomic_json(root/'selection.json', selection)
            atomic_json(root/'selection-receipt.json', {'selected_at': time.time(), 'call_id': call_id,
                'input_id': input_id, 'selection_sha256': digest(root/'selection.json'),
                'development_chunk_sha256': selection['development_chunk_sha256']})
            commit()
        selected = json.loads((root/'selection-receipt.json').read_text())
        require(selected['selection_sha256'] == digest(root/'selection.json'), 'Selection bytes changed')
        validate_selection_receipt(selection, selected, receipts)
    if condition is None:
        result = {'status': 'paper_online_completion_captured', 'amendment': p, 'parent_amendment': parent,
            'original_attempt': 'failed', 'parent_recovery': 'preempted_incomplete',
            'reused_conditions': list(p['retained']), 'new_chunks': 10,
            'aggregate': aggregate(envelope, completed, selection), 'audited': False}
        atomic_json(root/'summary.json', result); commit(); return result
    charged = sum(r['reserved_usd'] for k,r in receipts.items() if k not in p['retained'])
    require(charged + .1608936 <= 2, 'Completion compute allowance exhausted')
    budget = reservation(state/'budget.json')
    if budget is None:
        return {'status': 'budget_stopped'}
    stage, pool, arm = condition; name = chunk_name(*condition); folder = root/'chunks'/name
    receipt = {'status': 'claimed', 'chunk': name, 'call_id': call_id, 'input_id': input_id, 'claimed_at': time.time(),
        'plan_sha256': envelope['sha256'], 'source_sha256': signature(sources), 'amendment_sha256': signature(p),
        'remote_path': str(folder/'artifacts'), 'reserved_usd': budget['reserve'],
        'dispatch': 'nonpreemptible-completion-worker-shared-lease', 'nonpreemptible': True, 'price_multiplier': 3}
    atomic_json(folder/'receipt.json', receipt); commit()
    print(json.dumps({'event': 'completion_chunk_started', **receipt}), flush=True)
    try:
        result = run(envelope, Path('/opt/paperlab/paper-memory-01'), state/ORIGINAL/'news.db', state/'fly-data',
            folder/'artifacts', stage=stage, pool_index=pool, arm=arm,
            development=development if stage == 'test' else None, seconds=480)
        require(result['chunk'] == name and result['plan_sha256'] == envelope['sha256']
            and result['code_sha256'] == {f: sources['paperlab/'+f] for f in SOURCE_FILES}, 'Capture result differs')
        receipt.update(status='completed', completed_at=time.time(), summary_sha256=digest(folder/'artifacts/summary.json'),
            budget=settle(state/'budget.json', budget, time.time()-budget['started']))
        atomic_json(folder/'receipt.json', receipt); commit()
        return {'status': 'completion_chunk_completed', 'completed_chunks': len(completed)+1, 'receipt': receipt}
    except BaseException as exc:
        receipt.update(status='failed', error_type=type(exc).__name__, error=str(exc)[:300])
        atomic_json(folder/'receipt.json', receipt); commit(); raise
