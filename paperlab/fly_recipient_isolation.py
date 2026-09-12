"""Isolate each stimulated recipient while preserving frozen paper-trained memory."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import shutil
import time

import numpy as np

from .core import atomic_json, digest
from .fly import UPSTREAM_COMMIT
from .fly_market_activity import apply_boundary, attach_boundary_state, dynamic_state
from .fly_market_restoration import matches
from .fly_market_study import learned_state, memory_signature, restore_learned
from .fly_paper_inputs import audit_news
from .fly_paper_stimulation import AppliedCurrent, FILES as PARENT_FILES, INFERENCE, SOURCE, TARGETS, sequences, validate as validate_parent
from .fly_paper_study import imported_memory
from .fly_trace import TraceLab

FILES = tuple(dict.fromkeys((*PARENT_FILES, 'fly_recipient_isolation.py', 'fly_recipient_isolation_audit.py',
                            'fly_recipient_isolation_cloud.py')))
SETS = {'both': ['10704', '11402'], 'only_10704': ['10704'], 'only_11402': ['11402']}
INTERPRETATION = ('Post-hoc isolation on the same six previously inspected images. '
    'Current 10 to both recipients must reproduce the prior native count bins before either '
    'single-recipient intervention. Memory stays frozen. No account, trading returns, training, '
    'or automatic policy promotion. This tests the declared intervention, not anatomical causality.')


def protocol(parent_payload, reference_raw, audit_raw, reference_run):
    _, _, plan = validate_parent(parent_payload)
    return deepcopy({'schema': 1, 'kind': 'paper_memory_recipient_isolation',
        'parent_report_sha256': hashlib.sha256(reference_raw.encode()).hexdigest(),
        'parent_audit_sha256': hashlib.sha256(audit_raw.encode()).hexdigest(),
        'plan_sha256': parent_payload['paper_plan']['sha256'], 'reference_run_id': reference_run,
        'current': 10, 'target_sets': SETS, 'inference': INFERENCE,
        'arms': {f'pool{i}-{memory}-{label}': {'pool': key, 'memory': memory, 'current': 10,
                    'target_ids': ids, 'target_set': label}
                 for label, ids in SETS.items() for i, key in enumerate(plan['registration']['cohort'])
                 for memory in ('pristine', 'trained')}, 'interpretation': INTERPRETATION})


def validate(payload):
    if not isinstance(payload, dict) or set(payload) != {'protocol', 'paper_payload', 'reference_json', 'audit_json'}:
        raise ValueError('Expected fixed isolation protocol and its parent evidence')
    p = payload['protocol']; raw = payload['reference_json']; audit_raw = payload['audit_json']
    if not all(isinstance(x, str) and len(x.encode()) < 3_000_000 for x in (raw, audit_raw)):
        raise ValueError('Invalid bounded parent evidence')
    if not isinstance(p.get('reference_run_id'), str) or not re.fullmatch(r'assay-recipient-[a-f0-9]{32}', p['reference_run_id']):
        raise ValueError('Invalid original recording directory')
    if p != protocol(payload['paper_payload'], raw, audit_raw, p['reference_run_id']):
        raise ValueError('Isolation protocol differs from fixed target sets')
    old_protocol, original, plan = validate_parent(payload['paper_payload'])
    reference = json.loads(raw); audit = json.loads(audit_raw)
    if (reference.get('status') != 'paper_stimulation_completed' or reference['protocol'] != old_protocol
            or audit.get('status') != 'verified' or audit['protocol'] != old_protocol
            or audit['full_observations_verified'] != 36 or audit['native_bins_verified'] != 1800
            or reference['initial_dynamics_sha256'] != original['initial_dynamics_sha256']):
        raise ValueError('Incomplete or inconsistent parent stimulation evidence')
    for i, key in enumerate(plan['registration']['cohort']):
        for memory in ('pristine', 'trained'):
            name = f'pool{i}-{memory}-current10'; report = reference['reports'][name]
            if report['native_build'] != original['native_build'] or len(report['events']) != 3:
                raise ValueError('Parent native build or observations differ')
            for j, event in enumerate(report['events'], 1):
                row = audit['rows'][name][j-1]
                if any(event[k] != row[k] for k in ('side', 'difference_hz', 'gate_spikes', 'total_spikes', 'spike_sha256', 'input_sha256')):
                    raise ValueError('Parent report and audited decisions differ')
                sha = audit['artifact_sha256'].get(f'{name}/step-{j:02}.npz', '')
                if len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
                    raise ValueError('Missing audited native control artifact')
    return p, reference, audit, plan


class SelectiveCurrent(AppliedCurrent):
    """Use the existing native-argument guard with one or two declared targets."""
    def save(self, path):
        durations = np.array([x[0] for x in self.rows]); currents = np.array([x[1] for x in self.rows])
        if durations.shape != (50,) or not np.all(durations == 10) or currents.shape != (50, len(self.indices)):
            raise ValueError('Expected exactly 500 ms of selective native stimulation')
        np.savez_compressed(path, durations_ms=durations, currents=currents, target_indices=self.indices)


def verify_reference_artifacts(audit, root):
    root = Path(root); hashes = {}
    for pool in (0, 1):
        for memory in ('pristine', 'trained'):
            for observation in (1, 2, 3):
                name = f'pool{pool}-{memory}-current10/step-{observation:02}.npz'
                sha = audit['artifact_sha256'][name]
                if digest(root/name) != sha:raise ValueError('Original native control artifact differs: '+name)
                hashes[name] = sha
    return hashes


def run(payload, memory_root, news_archive, data, output):
    p, reference, parent_audit, plan = validate(payload)
    root = Path(output); reference_root = root.parent/p['reference_run_id']
    if root.exists():raise ValueError('Refuse to overwrite recipient isolation')
    # Existing control files are checked before graph loading or any new inference.
    verify_reference_artifacts(parent_audit, reference_root)
    seq = sequences(payload['paper_payload']); news_check = audit_news(payload['paper_payload']['paper_plan'], news_archive)
    root.mkdir(parents=True); atomic_json(root/'payload.json', payload); shutil.copyfile(news_archive, root/'news.db')
    deadline = time.monotonic()+480; lab = TraceLab(data); b = lab.brain
    first = next(iter(reference['reports'].values()))
    if b.build != first['native_build']:raise ValueError('Native build differs from recorded controls')
    if any(str(lab.types[lab.id_index[x]]) != 'MBON11' for x in TARGETS):raise ValueError('Recipient type differs')
    imported = imported_memory(b, plan, memory_root); pristine = learned_state(b); initial = dynamic_state(b)
    original = json.loads(payload['paper_payload']['reference_json'])
    if memory_signature(pristine) != original['pristine_memory_sha256']:raise ValueError('Pristine memory differs from parent')
    for name, arrays in (('pristine-memory', pristine), ('initial-dynamics', initial)):
        np.savez_compressed(root/(name+'.npz'), **arrays)
    np.savez_compressed(root/'neuron-ids.npz', neuron_ids=b.ids)
    if digest(root/'initial-dynamics.npz') != reference['initial_dynamics_sha256']:
        raise ValueError('Fresh dynamics differ from original stimulation')
    (root/'imported').mkdir(); shutil.copyfile(Path(memory_root)/'audit.json', root/'imported/audit.json')
    for i in range(2):shutil.copyfile(Path(memory_root)/f'pool{i}-memory.npz', root/f'imported/pool{i}-memory.npz')
    reports = {}; controls = []; started = time.monotonic()
    for name, arm in p['arms'].items():
        if arm['target_set'] != 'both' and len(controls) != 4:raise ValueError('Matched controls have not passed')
        memory = pristine if arm['memory'] == 'pristine' else imported[arm['pool']]
        restore_learned(b, memory); b.eta = .001; b.weights_frozen = True
        lab.fly.controller.s = replace(lab.fly.controller.s, learning=False)
        if any(not np.array_equal(v, getattr(b, k)) for k, v in initial.items()):raise ValueError('Arm did not start fresh')
        folder = root/name; folder.mkdir(); np.savez_compressed(folder/'initial-memory.npz', **memory)
        events = []; wall = time.monotonic(); whole_weights = hashlib.sha256(b.weight.tobytes()).hexdigest()
        targets = [lab.id_index[x] for x in arm['target_ids']]
        pool_index = plan['registration']['cohort'].index(arm['pool']); parent_name = f'pool{pool_index}-{arm["memory"]}-current10'
        if whole_weights != reference['reports'][parent_name]['all_weight_sha256']:
            raise ValueError('Full graph weights differ from original stimulation')
        for j, (rgb, row) in enumerate(seq[arm['pool']], 1):
            if time.monotonic() > deadline:raise TimeoutError('Bounded isolation experiment expired')
            boundary = apply_boundary(b, 'carry', j, folder/f'boundary-{j:02}.npz')
            with SelectiveCurrent(b, targets, 10) as applied:event = lab.capture(rgb, 'none', folder, j, targets, 10)
            applied.save(folder/f'current-{j:02}.npz')
            if memory_signature(learned_state(b)) != memory_signature(memory) or hashlib.sha256(b.weight.tobytes()).hexdigest() != whole_weights:
                raise ValueError('Frozen inference changed graph weights or memory')
            if event['input_sha256'] != row['neural']['input_sha256']:raise ValueError('Replayed image differs')
            if arm['target_set'] == 'both':
                expected = reference['reports'][parent_name]['events'][j-1]
                if not matches(event, expected):raise ValueError('Original decoder or observation counts differ')
                with np.load(reference_root/parent_name/f'step-{j:02}.npz', allow_pickle=False) as old, np.load(folder/f'step-{j:02}.npz', allow_pickle=False) as new:
                    if not np.array_equal(old['counts'], new['counts']):raise ValueError('Original full-neuron count bins differ')
            event.update(phase='replay', phase_step=j, market_decision_ts=row['decision_ts'], quote_ts=row['quote_ts'],
                input_preset='recorded_market', input_news='sealed_timestamped', news_features=row['neural']['news_features'],
                activity_boundary=boundary, stimulation={'target_ids':arm['target_ids'], 'current':10, 'duration_ms':500,
                    'artifact_sha256':digest(folder/f'current-{j:02}.npz')})
            events.append(event)
        if arm['target_set'] == 'both':controls.append(name)
        report = {'schema':1, 'source':SOURCE, 'config':{'preset':'recorded_market', 'view':'original', 'news':'sealed_timestamped',
            'pulses':'none', **INFERENCE, **arm, 'neurons':TARGETS}, 'native_build':b.build, 'upstream_commit':UPSTREAM_COMMIT,
            'graph':{'neurons':b.n, 'edges':len(b.post), 'plastic_edges':len(b.circuit['edges'])}, 'events':events,
            'seconds':time.monotonic()-wall, 'initial_memory_sha256':memory_signature(memory), 'final_memory_sha256':memory_signature(learned_state(b)),
            'all_weight_sha256':whole_weights, 'plan_sha256':p['plan_sha256'], 'reference_report_sha256':p['parent_report_sha256'],
            'pool':arm['pool'], 'interpretation':INTERPRETATION}
        atomic_json(folder/'report.json', report)
        requested = [lab.id_index[x] for x in (*TARGETS, '18540', '21778', '44069', '520206')]
        view = lab.export_view(folder, report, requested); attach_boundary_state(view, folder); atomic_json(folder/'view.json', view)
        reports[name] = report; atomic_json(root/'progress.json', {'completed':list(reports), 'controls_verified':controls})
        print('recipient_isolation '+json.dumps({'arm':name, 'actions':[e['side'] for e in events]}), flush=True)
    result = {'status':'paper_recipient_isolation_completed', 'protocol':p, 'reports':reports, 'controls_verified':controls,
        'seconds':time.monotonic()-started, 'news_audit':news_check, 'initial_dynamics_sha256':digest(root/'initial-dynamics.npz'),
        'neuron_ids_sha256':digest(root/'neuron-ids.npz'), 'code_sha256':{n:digest(Path(__file__).with_name(n)) for n in FILES}}
    atomic_json(root/'summary.json', result);return result
