"""Finalize study 11's amended capture from existing audits, without running models.

Exact receipt and proof bytes travel inside the report so the existing evidence
bundle can verify the amendment without rewriting the failed original study.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from .core import atomic_json, digest
from .fly_market_study import signature
from .fly_online_cloud import validate_selection_receipt
from .fly_online_protocol import ARMS, chunk_name, chunk_order
from .fly_online_recovery import DIRECTORY, validate_protocol
from .fly_online_schedule import DIRECTORY as ORIGINAL, aggregate
from .fly_online_study import SOURCE_FILES, select_development
from .fly_paper_inputs import validate
from .fly_paper_price_audit import audit_prices

STATUS = 'paper_online_recovery_audited'
COMPLETION_STATUS = 'paper_online_completion_audited'
COMPLETION_PROTOCOL_SHA256 = '24e12606eaba9e3112dc820ef0ce9f815ef3b4836046cbf6c6f76be08b4aeb4f'
PROTOCOL_SHA256 = 'e72488706d138dea0f0c4e2b74e497dddd81bfd9606c83bd3b25d628a32fa0b4'
ROOT = Path(__file__).resolve().parents[1]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def text_sha(raw):
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


def provenance(envelope, proof):
    """Validate immutable historical inputs, including both unsuccessful attempts."""
    require(text_sha(proof['protocol']) == PROTOCOL_SHA256, 'Recovery protocol is not the published amendment')
    p = validate_protocol(json.loads(proof['protocol']))
    plan = validate(envelope); registration = plan['registration']
    require(p['plan_sha256'] == envelope['sha256'], 'Recovery plan differs')
    sources = json.loads(proof['sources'])
    require(len(sources) == 56 and signature(sources) == p['original_sources_sha256'],
        'Original execution source manifest differs')
    require(set(proof['evidence']) == set(p['evidence_sha256']), 'Missing historical recovery evidence')
    for name, sha in p['evidence_sha256'].items():
        require(text_sha(proof['evidence'][name]) == sha, 'Recovery evidence changed: ' + name)
    failure = json.loads(proof['evidence']['fly-online-failure-11.json'])
    failed = json.loads(proof['failed_receipt'])
    require(text_sha(proof['failed_receipt']) == p['failed_receipt_sha256']
        and failed == failure['failure']['receipt'] and failed['status'] == 'failed'
        and failure['failure']['call_state'] == 'failed'
        and failure['failure']['observed_at'] < p['created_at'], 'Original failure was not preserved')
    prior = json.loads(proof['evidence']['fly-online-recovery-preflight-failure-11.json'])
    require(text_sha(proof['prior_halt']) == prior['halt_sha256']
        and json.loads(proof['prior_halt']) == prior['halt']
        and prior['terminal_call']['status'] == 'terminal_exception'
        and prior['halt']['at'] < p['created_at'], 'Previous recovery halt was not preserved')
    armed = json.loads(proof['armed']); sealed = json.loads(proof['sealing'])
    require(json.loads(proof['preregistration']) == registration
        and text_sha(proof['preregistration']) == armed['registration_sha256']
        and armed['registration_signature'] == signature(registration)
        and armed['source_sha256'] == signature(sources)
        and registration['recorded_at'] <= armed['armed_at'] < registration['development_start']
        and armed.get('call_id') and armed.get('input_id'), 'Original cloud registration differs')
    require(sealed['status'] == 'completed' and sealed['plan_sha256'] == envelope['sha256']
        and sealed['registration_sha256'] == armed['registration_sha256']
        and registration['end'] <= sealed['claimed_at'] < failed['claimed_at']
        and sealed.get('call_id') and sealed.get('input_id'), 'Sealed endpoint provenance differs')
    news = json.loads(proof['input_check'])
    require(news['seed'] == '0' and news['interpreter_start'] == 'fresh-subprocess'
        and news['verified_hash_probe'] == 7584921261715552910
        and news['original_inputs_changed'] is False
        and news['news'] == {'news_snapshot_sha256': plan['news_snapshot_sha256'],
                            'vectors_reconstructed': len(plan['news_features'])}
        and news['checked_at'] >= p['created_at'], 'Fresh-process input verification differs')
    return p, sources


def verify(root, report, envelope, summaries, receipts):
    """Additional gate for amended reports; the figure gate still checks all audits."""
    root = Path(root); proof = report['recovery']; p, sources = provenance(envelope, proof)
    names = [chunk_name(*c) for c in chunk_order()]
    require(report['status'] in (STATUS, COMPLETION_STATUS) and report['original_attempt'] == 'failed'
        and report['reused_controls'] == names[:2] and report['new_chunks'] == 14
        and set(summaries) == set(receipts) == set(proof['call_results']) == set(proof['projections']) == set(names)
        and proof['all_recorded_artifacts_rehashed'] is True, 'Amended completion is incomplete')
    armed = json.loads(proof['armed']); sealed = json.loads(proof['sealing'])
    require(report['cloud_registration'] == armed, 'Reported registration differs')
    news = json.loads(proof['input_check']); identities = set(); graph_data = set()
    completion = proof.get('completion'); cp = None; completion_news = None
    if report['status'] == COMPLETION_STATUS:
        from .fly_online_completion import DIRECTORY as FINAL_DIRECTORY, validate_protocol as validate_completion
        require(completion is not None and text_sha(completion['protocol']) == COMPLETION_PROTOCOL_SHA256,
            'Missing or changed non-preemptible completion amendment')
        cp = validate_completion(json.loads(completion['protocol']))
        failure = json.loads(completion['preemption_report'])
        require(text_sha(completion['preemption_report']) == cp['preemption_report_sha256']
            and failure['receipt']['chunk'] == names[6] and failure['receipt']['status'] == 'claimed'
            and failure['receipt']['amendment_sha256'] == signature(p)
            and failure['terminal_result'] == {'status': 'writer_busy', 'writer': 'worker'}
            and failure['provider_event'] == 'Container terminated due to preemption'
            and failure['recovery_app_stopped'] is True and failure['observed_at'] < cp['created_at']
            and failure['complete_summary_present'] is False and failure['development_selection_present'] is False
            and report['parent_recovery'] == 'preempted_incomplete'
            and report['completion_reused_conditions'] == names[:6] and report['completion_new_chunks'] == 10,
            'Preemption or completion provenance differs')
        completion_news = json.loads(completion['input_check'])
        require(all(completion_news[k] == news[k] for k in ('news', 'seed', 'interpreter_start',
            'verified_hash_probe', 'original_inputs_changed')) and completion_news['checked_at'] >= cp['created_at'],
            'Completion changed the original seeded inputs')
        require(failure['receipt']['call_id'] not in {r['call_id'] for r in receipts.values()},
            'Interrupted partial capture was used as a completed condition')
    else:
        require(completion is None, 'Completion cannot be relabeled as its preempted parent')
    last_completed = sealed['claimed_at']; charge = 0; completion_charge = 0
    for i, name in enumerate(names):
        receipt = receipts[name]; summary = summaries[name]
        final_capture = cp is not None and i >= 6
        namespace = ORIGINAL if i < 2 else FINAL_DIRECTORY if final_capture else DIRECTORY
        dispatch = ('scheduled-worker-after-paper-cycle' if i < 2 else
            'nonpreemptible-completion-worker-shared-lease' if final_capture else 'separate-recovery-worker-shared-lease')
        require(receipt['status'] == 'completed' and receipt['chunk'] == name
            and receipt['plan_sha256'] == envelope['sha256'] and receipt['source_sha256'] == signature(sources)
            and receipt['remote_path'] == f'/state/{namespace}/chunks/{name}/artifacts'
            and receipt['dispatch'] == dispatch and receipt.get('call_id') and receipt.get('input_id')
            and receipt['claimed_at'] >= last_completed and receipt['completed_at'] >= receipt['claimed_at'],
            'Completed receipt ownership or chronology differs: ' + name)
        last_completed = receipt['completed_at']
        require(summary['code_sha256'] == {f: sources['paperlab/' + f] for f in SOURCE_FILES}
            and summary['graph'] == {'neurons': 166700, 'edges': 25582938, 'plastic_edges': 7835}
            and summary['news_audit'] == news['news'], 'Executed model or news inputs differ: ' + name)
        identities.add((summary['native_build']['binary_sha256'],
            summary['artifact_sha256']['initial-dynamics.npz'], summary['artifact_sha256']['pristine-memory.npz']))
        projection_raw = proof['projections'][name]; projection = json.loads(projection_raw)
        require(projection['status'] == 'online_view_projection_audited' and projection['validation_only'] is False
            and projection['chunk'] == name and projection['plan_sha256'] == envelope['sha256']
            and projection['summary_sha256'] == receipt['summary_sha256']
            and projection['native_audit_sha256'] == report['audit_sha256'][name]
            and projection['artifact_sha256'] == summary['artifact_sha256']
            and projection['projection']['all_topology_verified'] is True
            and projection['projection']['all_plotted_series_verified'] is True,
            'Full recording projection differs: ' + name)
        graph_data.add(signature(projection['graph_data_sha256']))
        if cp is not None and i < 6:
            pins = cp['retained'][name]
            require(digest(root/f'receipts/{name}-completed.json') == pins['receipt_sha256']
                and receipt['summary_sha256'] == pins['summary_sha256']
                and text_sha(projection_raw) == pins['projection_sha256'], 'Retained completion evidence changed')
        call = json.loads(proof['call_results'][name])
        if i < 2:
            pins = p['controls'][name]
            require(digest(root/f'receipts/{name}-completed.json') == pins['receipt_sha256']
                and receipt['summary_sha256'] == pins['summary_sha256']
                and text_sha(projection_raw) == pins['projection_sha256'], 'Reused control bytes changed')
            result = call.get('study_11') or {}
            require(result.get('status') == 'paper_online_chunk_completed' and result.get('chunk') == name
                and result.get('receipt') == receipt, 'Original control call did not complete')
        else:
            owner_protocol = cp if final_capture else p
            require(receipt['amendment_sha256'] == signature(owner_protocol)
                and receipt['claimed_at'] >= owner_protocol['created_at']
                and call.get('status') == ('completion_chunk_completed' if final_capture else 'recovery_chunk_completed')
                and call.get('receipt') == receipt
                and call.get('completed_chunks') == i + 1, 'Recovery call did not complete its own condition')
            require(0 < receipt['reserved_usd'] <= (.1608936 if final_capture else .0536312)
                and receipt['budget']['monthly_limit_usd'] == 25
                and receipt['budget']['provider_bill'] is False, 'Recovery compute policy differs')
            if final_capture:
                require(receipt['nonpreemptible'] is True and receipt['price_multiplier'] == 3,
                    'Completion capture was not priced as non-preemptible')
                completion_charge += receipt['reserved_usd']
            else:
                charge += receipt['budget']['estimated_compute_usd']
        if i == 2:
            require(news['call_id'] == receipt['call_id'] and news['input_id'] == receipt['input_id']
                and news['checked_at'] <= receipt['claimed_at'], 'Input check did not precede first recovery capture')
        if final_capture and i == 6:
            require(completion_news['call_id'] == receipt['call_id'] and completion_news['input_id'] == receipt['input_id']
                and completion_news['checked_at'] <= receipt['claimed_at'], 'Completion input check did not precede capture')
    require(len(identities) == len(graph_data) == 1 and len({r['call_id'] for r in receipts.values()}) == 16
        and 0 <= charge <= .8 and 0 <= completion_charge <= 2, 'Model identity, distinct calls or recovery budget differs')
    selection = select_development(envelope, {k: v for k, v in summaries.items() if v['stage'] == 'development'})
    require(json.loads(proof['selection']) == selection
        and text_sha(proof['selection']) == report['selection_receipt']['selection_sha256'],
        'Saved selection bytes differ')
    validate_selection_receipt(selection, report['selection_receipt'], receipts)
    first_test = receipts[names[8]]
    require(all(report['selection_receipt'][k] == first_test[k] for k in ('call_id', 'input_id')),
        'Selection belongs to a different worker')
    require(all(s['selection'] == (selection if s['stage'] == 'test' else None) for s in summaries.values()),
        'Capture selection differs from its phase')
    cloud = json.loads(proof['cloud_summary'])
    require(cloud['audited'] is False and cloud['original_attempt'] == 'failed'
        and cloud['aggregate'] == aggregate(envelope, summaries, selection), 'Cloud recovery aggregate differs')
    if cp is None:
        require(cloud['status'] == 'paper_online_recovery_captured' and cloud['amendment'] == p
            and cloud['reused_controls'] == names[:2] and cloud['new_chunks'] == 14, 'Cloud recovery provenance differs')
    else:
        require(cloud['status'] == 'paper_online_completion_captured' and cloud['amendment'] == cp
            and cloud['parent_amendment'] == p and cloud['parent_recovery'] == 'preempted_incomplete'
            and cloud['reused_conditions'] == names[:6] and cloud['new_chunks'] == 10, 'Cloud completion provenance differs')


def build(original, recovery, controls, evidence, price_archive, output, *, completion=None):
    """Rehash complete audited recordings, then install the validated report last."""
    from .fly_online_figure import CHECKS, evidence as verify_figure
    original, recovery, evidence, price_archive, output = map(Path, (original, recovery, evidence, price_archive, output))
    require(not output.exists(), 'Preserve earlier final reports; choose a new output')
    final_root = Path(completion) if completion is not None else recovery
    # These reads fail early while collection or selection remains incomplete.
    cloud_raw = (final_root/'summary.json').read_text()
    selection_raw = (final_root/'selection.json').read_text()
    envelope = json.loads((original/'plan.json').read_text()); registration = validate(envelope)['registration']
    protocol_raw = (recovery/'protocol.json').read_text(); protocol = json.loads(protocol_raw)
    proof = {'protocol': protocol_raw, 'sources': (original/'source-hashes.json').read_text(),
        'failed_receipt': (recovery/'original-failed-receipt.json').read_text(),
        'prior_halt': (recovery/'previous-recovery-halt.json').read_text(),
        'evidence': {name: (evidence/name).read_text() for name in protocol['evidence_sha256']},
        'preregistration': (original/'preregistration.json').read_text(), 'armed': (original/'armed.json').read_text(),
        'sealing': (original/'sealing.json').read_text(), 'input_check': (recovery/'input-check.json').read_text(),
        'selection': selection_raw, 'cloud_summary': cloud_raw, 'call_results': {}, 'projections': {},
        'all_recorded_artifacts_rehashed': True}
    provenance(envelope, proof)
    if completion is not None:
        proof['completion'] = {'protocol': (final_root/'protocol.json').read_text(),
            'preemption_report': (evidence/'fly-online-recovery-preemption-11.json').read_text(),
            'input_check': (final_root/'input-check.json').read_text()}
    names = [chunk_name(*c) for c in chunk_order()]
    summaries = {}; receipts = {}; projections = {}
    for i, name in enumerate(names):
        base = original if i < 2 else final_root if completion is not None and i >= 6 else recovery
        chunk = base/'chunks'/name
        projection_dir = Path(controls[i]) if i < 2 else base/'projections'/name
        s = json.loads((chunk/'summary.json').read_text()); summaries[name] = s
        receipts[name] = json.loads((base/f'receipts/{name}-completed.json').read_text())
        projection_raw = (projection_dir/'report.json').read_text(); projection = json.loads(projection_raw)
        require(digest(chunk/'summary.json') == projection['summary_sha256']
            and digest(projection_dir/'audit.json') == projection['native_audit_sha256']
            and digest(projection_dir/'view.json') == projection['audited_view_sha256']
            and s['artifact_sha256'] == projection['artifact_sha256'], 'Cached audit inputs changed: ' + name)
        for file, sha in s['artifact_sha256'].items():
            path = Path(file)
            require(not path.is_absolute() and '..' not in path.parts, 'Unsafe artifact path')
            require(digest(chunk/path) == sha, 'Recorded artifact changed: ' + name + '/' + file)
        proof['call_results'][name] = (base/f'call-results/{name}.json').read_text()
        proof['projections'][name] = projection_raw
        projections[name] = projection_dir
        print(json.dumps({'event': 'cached_audit_bytes_verified', 'chunk': name}), flush=True)
    selection = json.loads(selection_raw); prices = audit_prices(envelope, price_archive)
    report = aggregate(envelope, summaries, selection)
    report.update(status=COMPLETION_STATUS if completion is not None else STATUS,
        audited=True, original_attempt='failed', reused_controls=names[:2], new_chunks=14,
        recovery=proof, cloud_registration=json.loads(proof['armed']),
        selection_receipt=json.loads((final_root/'selection-receipt.json').read_text()),
        verification={**dict.fromkeys(CHECKS, True), 'distinct_completed_worker_calls': 16}, price_audit=prices,
        phase_diagnostics={key: {arm: {stage: summaries[chunk_name(stage, pool, arm)]['outcome']
            for stage in ('development', 'test')} for arm in ARMS} for pool, key in enumerate(registration['cohort'])},
        audit_sha256={name: digest(projections[name]/'audit.json') for name in names},
        interpretation='Amended recovery of study 11: two preserved controls and fourteen separately owned captures. '
            'All original inputs and execution sources retained; fresh subprocesses fix seed initialization. '
            'Full-bin and view audits reused after rehashing recordings. No policy promotion or profitability claim.')
    if completion is not None:
        report.update(parent_recovery='preempted_incomplete', completion_reused_conditions=names[:6], completion_new_chunks=10,
            interpretation='Study 11 with two disclosed amendments. Recovery 02 was preempted during condition seven; '
                'its partial recording is excluded. Six complete conditions are retained and ten fresh conditions '
                'are captured on non-preemptible workers. Original sealed inputs and model sources remain unchanged. '
                'Independent full-bin audits and source receipts cover all sixteen conditions; no policy promotion.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='fly-recovery-report-', dir=output.parent) as temporary:
        staging = Path(temporary)/'complete'; staging.mkdir()
        shutil.copyfile(original/'plan.json', staging/'plan.json')
        atomic_json(staging/'price-audit.json', prices); atomic_json(staging/'report.json', report)
        for i, name in enumerate(names):
            base = original if i < 2 else final_root if completion is not None and i >= 6 else recovery
            for source, relative in ((base/f'chunks/{name}/summary.json', f'chunks/{name}/summary.json'),
                    (base/f'receipts/{name}-completed.json', f'receipts/{name}-completed.json'),
                    (projections[name]/'audit.json', f'audits/{name}.json')):
                target = staging/relative; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, target)
            view_root = staging/'views'/name; view_root.mkdir(parents=True)
            shutil.copyfile(projections[name]/'view.json', view_root/'view.json')
            projection = json.loads(proof['projections'][name])
            require(digest(view_root/'view.json') == projection['audited_view_sha256'],
                'Copied debugger view differs from its audit')
            view = json.loads((view_root/'view.json').read_text())
            atomic_json(view_root/'report.json', view['report'])
            atomic_json(view_root/'remote.json', {'remote_path': receipts[name]['remote_path']+'/trace',
                'call_id': receipts[name]['call_id']})
            for source in (base/'chunks'/name/'trace').glob('step-*.npz'):
                (view_root/source.name).hardlink_to(source)
        checked, _, _, fixture = verify_figure(staging)
        require(not fixture and checked == report, 'A fixture cannot finalize the recovery')
        # Create exclusively, including an empty existing destination. No earlier
        # report or failed receipt is replaced if publication is interrupted.
        output.mkdir(exist_ok=False)
        for source in sorted(staging.iterdir(), key=lambda p: p.name == 'report.json'):
            shutil.move(str(source), output/source.name)
    return {'status': report['status'], 'report_sha256': digest(output/'report.json'),
        'total_equity': report['total_equity'], 'selection': selection['selected'],
        'new_neural_observations': 0, 'cloud_submissions': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('original', 'recovery', 'evidence', 'price-archive', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--control-projections', type=Path, nargs=2, required=True)
    parser.add_argument('--completion', type=Path, help='Separately captured non-preemptible completion directory')
    args = parser.parse_args()
    print(json.dumps(build(args.original, args.recovery, args.control_projections,
        args.evidence, args.price_archive, args.out, completion=args.completion), indent=2))


if __name__ == '__main__':
    main()
