"""Amended capture of study 11 after its terminal, pre-neural news-audit failure.

Original receipts and arrays are read only. Two audited controls are reused;
fourteen fresh chunks have separate owners, with no automatic retry or resume.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .budget import reserve, settle
from .core import atomic_json, digest
from .fly_market_study import signature
from .fly_online_protocol import chunk_name, chunk_order
from .fly_online_schedule import DIRECTORY as ORIGINAL, aggregate, source_hashes
from .fly_online_study import SOURCE_FILES, run_chunk, select_development
from .fly_paper_inputs import audit_news, validate

DIRECTORY = 'registered-paper-11-recovery-02'
POLICY = ('Amended recovery after the original third chunk failed before neural construction. '
    'Preserve all original receipts, sealed inputs and the first two independently audited controls. '
    'Start verification and capture in fresh Python subprocesses with PYTHONHASHSEED=0 in the environment '
    'before interpreter startup; verify the seed fingerprint and exact reconstruction of every original news vector. '
    'Preserve the halted image-environment recovery in its original namespace. '
    'Run the remaining fourteen original conditions once, in their original order, in a separate namespace. '
    'Keep all 56 original execution sources, full graph, decoder, reward, learning settings and costs unchanged. '
    'Persist original-rule development selection before any test capture. Never retry a claimed or failed chunk. '
    'Use the shared worker lease and budget with two CPUs, 8 GiB, 600 seconds, no platform retries, '
    'a $25 worker monthly cap and at most $0.80 additional conservative reserved compute. '
    'Publish only with separate amendment provenance and independent full-recording audits; no policy promotion.')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def validate_protocol(p):
    require(set(p) == {'version', 'created_at', 'policy', 'seed', 'plan_sha256',
        'original_sources_sha256', 'failed_receipt_sha256', 'controls',
        'code_sha256', 'evidence_sha256'}, 'Recovery protocol fields differ')
    require(p['version'] == 2 and p['seed'] == '0' and p['policy'] == POLICY,
        'Recovery changes the fixed amendment')
    require(list(p['controls']) == [chunk_name(*c) for c in chunk_order()[:2]],
        'Recovery must reuse exactly the first two controls')
    require(set(p['code_sha256']) == {'paperlab/fly_online_recovery.py', 'recovery_cloud.py'},
        'Recovery source manifest differs')
    require(set(p['evidence_sha256']) == {'fly-online-failure-11.json',
        'fly-online-control-first-11.json', 'fly-online-control-second-11.json',
        'fly-news-process-probe-11.json', 'fly-online-recovery-preflight-failure-11.json'},
        'Recovery evidence manifest differs')
    hashes = [p[k] for k in ('plan_sha256', 'original_sources_sha256', 'failed_receipt_sha256')]
    hashes += list(p['code_sha256'].values()) + list(p['evidence_sha256'].values())
    for row in p['controls'].values():
        require(set(row) == {'receipt_sha256', 'summary_sha256', 'projection_sha256'},
            'Control provenance differs')
        hashes += list(row.values())
    require(all(isinstance(h, str) and len(h) == 64 and all(c in '0123456789abcdef' for c in h)
        for h in hashes), 'Invalid recovery evidence hash')
    require(type(p['created_at']) in (int, float) and 1789260773 < p['created_at'] < 1800000000,
        'Recovery must follow the confirmed failure')
    return p


def fresh_news_audit(envelope, archive):
    script = '''
import json,sys
from paperlab.fly_paper_inputs import audit_news
assert hash('fly-news-seed-probe') == 7584921261715552910, 'Seed was not applied before interpreter startup'
print(json.dumps(audit_news(json.load(sys.stdin), sys.argv[1])))
'''
    child = subprocess.run([sys.executable, '-c', script, str(archive)],
        input=json.dumps(envelope), capture_output=True, text=True, check=True, timeout=30,
        env={**os.environ, 'PYTHONHASHSEED':'0'})
    return json.loads(child.stdout)


def run_seeded_chunk(envelope, memory, news, data, output, *, stage, pool_index, arm,
                     development=None, seconds=480):
    require(seconds == 480, 'Use the original fixed chunk compute limit')
    output = Path(output)
    atomic_json(output.parent/'inputs.json', envelope)
    script = ("assert hash('fly-news-seed-probe') == 7584921261715552910, 'Seed initialization differs'; "
        'from paperlab.fly_online_study import main; main()')
    command = [sys.executable, '-c', script, '--plan', str(output.parent/'inputs.json'),
        '--memory', str(memory), '--news', str(news), '--fly-data', str(data), '--out', str(output),
        '--stage', stage, '--pool', str(pool_index), '--arm', arm]
    if development is not None:
        atomic_json(output.parent/'development-inputs.json', development)
        command += ['--development', str(output.parent/'development-inputs.json')]
    subprocess.run(command, check=True, timeout=510, env={**os.environ, 'PYTHONHASHSEED':'0'})
    return json.loads((output/'summary.json').read_text())


def verify_reference(p, original, specifications, evidence, code_root):
    validate_protocol(p)
    original, specifications, evidence, code_root = map(Path, (original, specifications, evidence, code_root))
    for name, sha in p['code_sha256'].items():
        require(digest(code_root/name) == sha, 'Recovery source changed: '+name)
    for name, sha in p['evidence_sha256'].items():
        require(digest(evidence/name) == sha, 'Recovery evidence changed: '+name)
    prior = json.loads((evidence/'fly-online-recovery-preflight-failure-11.json').read_text())
    prior_root = original.parent/'registered-paper-11-recovery-01'
    require(digest(prior_root/'halt.json') == prior['halt_sha256']
        and json.loads((prior_root/'halt.json').read_text()) == prior['halt']
        and prior['terminal_call']['status'] == 'terminal_exception'
        and not (prior_root/'protocol.json').exists() and not (prior_root/'chunks').exists(),
        'Preserve the previous preflight failure without neural claims')
    probe = json.loads((evidence/'fly-news-process-probe-11.json').read_text())
    require(probe['fresh_seed0_child']['hash_probe'] == 7584921261715552910
        and probe['fresh_seed0_child']['differing_vectors'] == 0
        and probe['parent']['differing_vectors'] > 0
        and probe['new_neural_observations'] == 0 and probe['archive_writes'] == 0,
        'Cloud seed-initialization evidence differs')
    envelope = json.loads((original/'plan.json').read_text()); validate(envelope)
    require(envelope['sha256'] == p['plan_sha256'], 'Original sealed plan changed')
    sources = json.loads((original/'source-hashes.json').read_text())
    require(signature(sources) == p['original_sources_sha256'] and source_hashes(specifications) == sources,
        'Original execution sources changed')
    failed = original/'chunks'/chunk_name(*chunk_order()[2])/'receipt.json'
    require(digest(failed) == p['failed_receipt_sha256'], 'Original failure receipt changed')
    failure = json.loads((evidence/'fly-online-failure-11.json').read_text())
    receipt = json.loads(failed.read_text())
    require(receipt == failure['failure']['receipt'] and receipt['status'] == 'failed'
        and receipt['error'] == 'Sealed news differs from timestamp-eligible archived revisions'
        and failure['failure']['call_state'] == 'failed', 'Original terminal failure differs')
    require(not (failed.parent/'artifacts').exists(), 'Failed original chunk unexpectedly has neural artifacts')
    require(not any((original/'chunks'/chunk_name(*c)).exists() for c in chunk_order()[3:]),
        'Original work continued beyond its failed chunk')
    for i, (name, pins) in enumerate(p['controls'].items()):
        folder = original/'chunks'/name
        require(digest(folder/'receipt.json') == pins['receipt_sha256']
            and digest(folder/'artifacts/summary.json') == pins['summary_sha256'],
            'Original completed control changed')
        projection_file = evidence/('fly-online-control-'+('first' if i == 0 else 'second')+'-11.json')
        projection = json.loads(projection_file.read_text())
        require(digest(projection_file) == pins['projection_sha256']
            and projection['summary_sha256'] == pins['summary_sha256']
            and projection['status'] == 'online_view_projection_audited'
            and projection['validation_only'] is False
            and projection['projection']['observations'] == 24
            and projection['projection']['all_topology_verified']
            and projection['projection']['all_plotted_series_verified'], 'Control audit differs')
    # This exact audit happens before a new chunk or native brain can be created.
    news = fresh_news_audit(envelope, original/'news.db')
    return envelope, sources, news


def retained_chunks(p, envelope, sources, original, root):
    completed, receipts = {}, {}
    amendment_sha = signature(p)
    for index, condition in enumerate(chunk_order()):
        name = chunk_name(*condition)
        folder = (original if index < 2 else root)/'chunks'/name
        receipt_path = folder/'receipt.json'
        if not receipt_path.exists():
            require(not folder.exists(), 'Unclaimed recovery artifacts exist')
            require(not any((root/'chunks'/chunk_name(*c)).exists() for c in chunk_order()[index+1:]),
                'Recovery skipped a condition')
            return completed, receipts, condition, None
        receipt = json.loads(receipt_path.read_text())
        require(receipt.get('chunk') == name and receipt.get('plan_sha256') == envelope['sha256']
            and receipt.get('source_sha256') == signature(sources)
            and receipt.get('call_id') and receipt.get('input_id'), 'Chunk identity differs')
        require(receipt.get('remote_path') == str(folder/'artifacts'), 'Chunk artifact path differs')
        if index >= 2:
            require(receipt.get('amendment_sha256') == amendment_sha
                and receipt.get('claimed_at', 0) >= p['created_at'], 'Recovery ownership differs')
        if receipt.get('status') != 'completed':
            require(receipt.get('status') in ('claimed', 'failed'), 'Unknown chunk status')
            return completed, receipts, None, receipt
        require(digest(folder/'artifacts/summary.json') == receipt['summary_sha256'], 'Chunk summary changed')
        result = json.loads((folder/'artifacts/summary.json').read_text())
        require(result.get('status') == 'paper_online_chunk_completed' and result.get('chunk') == name
            and result.get('plan_sha256') == envelope['sha256']
            and result.get('code_sha256') == {f:sources['paperlab/'+f] for f in SOURCE_FILES},
            'Chunk execution differs')
        require(receipt['call_id'] not in {r['call_id'] for r in receipts.values()}, 'Reused owning call')
        completed[name], receipts[name] = result, receipt
    return completed, receipts, None, None


def execute(state, specifications, evidence, code_root, *, call_id, input_id, commit,
            run=run_seeded_chunk):
    state = Path(state); root = state/DIRECTORY; original = state/ORIGINAL
    if (root/'halt.json').exists():
        return {'status':'recovery_halted', 'receipt':json.loads((root/'halt.json').read_text())}
    if (root/'summary.json').exists():
        return {'status':'paper_online_recovery_captured', 'summary_sha256':digest(root/'summary.json')}
    require(call_id and input_id, 'Recovery requires a cloud owner')
    p = json.loads((Path(evidence)/'protocol.json').read_text())
    envelope, sources, news = verify_reference(p, original, specifications, evidence, code_root)
    root.mkdir(parents=True, exist_ok=True)
    saved = root/'protocol.json'
    if saved.exists():require(json.loads(saved.read_text()) == p, 'Saved recovery protocol changed')
    else:
        atomic_json(saved, p)
        atomic_json(root/'input-check.json', {'news':news, 'seed':'0', 'call_id':call_id,
            'input_id':input_id, 'checked_at':time.time(), 'original_inputs_changed':False,
            'interpreter_start':'fresh-subprocess', 'verified_hash_probe':7584921261715552910})
        commit()
    completed, receipts, condition, unresolved = retained_chunks(p, envelope, sources, original, root)
    if unresolved:
        return {'status':'recovery_chunk_unresolved', 'receipt':unresolved, 'completed_chunks':len(completed)}
    development = {k:v for k,v in completed.items() if v['stage'] == 'development'}
    selection = None
    if len(development) == 8:
        selection = select_development(envelope, development); selected = root/'selection.json'
        if selected.exists():
            require(json.loads(selected.read_text()) == selection, 'Development selection changed')
        else:
            require(not any(k.startswith('test-') for k in completed), 'Test preceded selection')
            atomic_json(selected, selection)
            atomic_json(root/'selection-receipt.json', {'selected_at':time.time(), 'call_id':call_id,
                'input_id':input_id, 'selection_sha256':digest(selected),
                'development_chunk_sha256':selection['development_chunk_sha256']})
            commit()
        selection_receipt = json.loads((root/'selection-receipt.json').read_text())
        require(selection_receipt['selection_sha256'] == digest(selected)
            and selection_receipt['selected_at'] >= max(receipts[k]['completed_at'] for k in development)
            and all(r['claimed_at'] >= selection_receipt['selected_at'] for k,r in receipts.items() if k.startswith('test-')),
            'Selection timing differs')
    if condition is None:
        report = {'status':'paper_online_recovery_captured', 'amendment':p,
            'original_attempt':'failed', 'reused_controls':list(p['controls']), 'new_chunks':14,
            'aggregate':aggregate(envelope, completed, selection), 'audited':False,
            'interpretation':'Amended capture only. Independent full-bin audits are still required; no policy promotion.'}
        atomic_json(root/'summary.json', report); commit(); return report
    # At most fourteen worst-case reservations fit inside the separate $0.80 ceiling.
    charged = sum(r.get('budget', {}).get('estimated_compute_usd', r.get('reserved_usd', 0))
        for k,r in receipts.items() if k not in p['controls'])
    require(charged + 610 * 2 * (.0000131 * 2 + .00000222 * 8) <= .80,
        'Recovery compute allowance exhausted')
    reservation = reserve(state/'budget.json', True, limit_override=25, startup_seconds=10, memory_gib=8)
    if reservation is None:return {'status':'budget_stopped'}
    stage, pool, arm = condition; name = chunk_name(*condition); folder = root/'chunks'/name
    receipt = {'status':'claimed', 'chunk':name, 'call_id':call_id, 'input_id':input_id,
        'claimed_at':time.time(), 'plan_sha256':envelope['sha256'], 'source_sha256':signature(sources),
        'amendment_sha256':signature(p), 'remote_path':str(folder/'artifacts'),
        'reserved_usd':reservation['reserve'], 'dispatch':'separate-recovery-worker-shared-lease'}
    atomic_json(folder/'receipt.json', receipt); commit()
    print(json.dumps({'event':'recovery_chunk_started', **receipt}), flush=True)
    try:
        result = run(envelope, Path('/opt/paperlab/paper-memory-01'), original/'news.db', state/'fly-data',
            folder/'artifacts', stage=stage, pool_index=pool, arm=arm,
            development=development if stage == 'test' else None, seconds=480)
        require(result['chunk'] == name and result['plan_sha256'] == envelope['sha256']
            and result['code_sha256'] == {f:sources['paperlab/'+f] for f in SOURCE_FILES},
            'Recovery result source differs')
        receipt.update(status='completed', completed_at=time.time(), summary_sha256=digest(folder/'artifacts/summary.json'),
            budget=settle(state/'budget.json', reservation, time.time()-reservation['started']))
        atomic_json(folder/'receipt.json', receipt); commit()
        return {'status':'recovery_chunk_completed', 'completed_chunks':len(completed)+1, 'receipt':receipt}
    except Exception as exc:
        receipt.update(status='failed', error_type=type(exc).__name__, error=str(exc)[:300])
        atomic_json(folder/'receipt.json', receipt); commit(); raise
