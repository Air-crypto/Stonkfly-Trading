"""Bounded recipient stimulation using audited paper-trained memories and inputs."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

from .core import Tick, atomic_json, digest
from .fly import UPSTREAM_COMMIT, frame
from .fly_market_activity import apply_boundary, attach_boundary_state, dynamic_state
from .fly_market_restoration import matches
from .fly_market_study import learned_state, memory_signature, quote_at, restore_learned, signature
from .fly_paper_inputs import SealedNews, audit_news, validate as validate_inputs
from .fly_paper_study import SOURCE_FILES, imported_memory
from .fly_trace import TraceLab

TARGETS = ['10704', '11402']
CURRENTS = [0, 5, 10]
MEMORIES = ['pristine', 'trained']
SOURCE = 'matched_market_stimulus_assay'
FILES = tuple(dict.fromkeys((*SOURCE_FILES, 'fly_paper_stimulation.py',
    'fly_paper_stimulation_audit.py', 'fly_paper_stimulation_cloud.py', 'cloud_debug.py')))
INFERENCE = {'learning': False, 'reinforcement': 'none', 'steps': 3,
             'duration_ms': 500, 'state_reset': 'carry', 'eta': .001}
INTERPRETATION = ('Post-hoc mechanism test on the already examined study 09 test inputs. '
    'No account, fills, returns, training, or policy promotion. Applied current is an '
    'explicit diagnostic intervention, not a biologically calibrated input.')


def conditions(cohort):
    # Every zero-current control finishes before any stimulated condition starts.
    return {f'pool{i}-{memory}-current{current}': {'pool': key, 'memory': memory, 'current': current}
            for current in CURRENTS for i, key in enumerate(cohort) for memory in MEMORIES}


def protocol(reference_raw, envelope):
    plan = validate_inputs(envelope)
    return deepcopy({'schema': 1, 'kind': 'paper_memory_recipient_stimulation',
        'reference_report_sha256': hashlib.sha256(reference_raw.encode()).hexdigest(),
        'plan_sha256': envelope['sha256'], 'target_ids': TARGETS,
        'currents': CURRENTS, 'inference': INFERENCE,
        'arms': conditions(plan['registration']['cohort']),
        'timing': 'Same current to both targets throughout each observation; fresh dynamics per arm; carry between observations.',
        'control_gate': 'All four zero-current controls must reproduce the original full-neuron counts and decoder before stimulation.',
        'interpretation': INTERPRETATION})


def validate(payload):
    if not isinstance(payload, dict) or set(payload) != {'protocol', 'reference_json', 'paper_plan'}:
        raise ValueError('Expected protocol, reference JSON and sealed paper plan')
    raw = payload['reference_json']
    if not isinstance(raw, str) or len(raw.encode()) > 2_000_000:
        raise ValueError('Invalid bounded reference')
    plan = validate_inputs(payload['paper_plan']); ref = json.loads(raw)
    if payload['protocol'] != protocol(raw, payload['paper_plan']):
        raise ValueError('Stimulation protocol differs from the fixed conditions')
    r = plan['registration']
    if len(r['cohort']) != 2 or ref['plan_sha256'] != payload['paper_plan']['sha256'] or ref['registration'] != r:
        raise ValueError('Reference cohort or plan differs')
    if not ref.get('verification') or any(v is not True for v in ref['verification'].values()):
        raise ValueError('Reference must have completed every evidence audit')
    for key in r['cohort']:
        for memory in MEMORIES:
            outcome = ref['phase_diagnostics'][key][memory+'_frozen']['test']
            rows = outcome['decisions']
            if len(rows) != 4 or not rows[-1]['terminal']:
                raise ValueError('Reference must contain three decisions and a terminal mark')
            expected = ref['pristine_memory_sha256'] if memory == 'pristine' else r['source_memories'][key]['memory_sha256']
            if outcome['initial_memory_sha256'] != expected or outcome['final_memory_sha256'] != expected:
                raise ValueError('Reference memory origin differs')
            for j, row in enumerate(rows[:3]):
                event = row['neural']
                if row['terminal'] or not row['available'] or event is None or row['decision_ts'] != r['test_start']+j*300:
                    raise ValueError('Reference has a missing or shifted observation')
                if event['stimulus'] != 'none' or event['plasticity_enabled'] or event['weight_delta_l2'] != 0:
                    raise ValueError('Reference is not frozen')
                if event['activity_boundary']['requested_mode'] != 'carry':
                    raise ValueError('Reference must carry activity between observations')
                other = ref['phase_diagnostics'][key]['pristine_frozen']['test']['decisions'][j]
                if row['quote_ts'] != other['quote_ts'] or event['input_sha256'] != other['neural']['input_sha256']:
                    raise ValueError('Reference controls saw different inputs')
    return payload['protocol'], ref, plan


def sequences(payload):
    _, ref, p = validate(payload)
    vendor = str(Path(__file__).resolve().parents[1]/'vendor/stonkfly')
    if vendor not in sys.path: sys.path.insert(0, vendor)
    news = SealedNews(p['news_features']); result = {}
    for key in p['registration']['cohort']:
        ticks = [Tick(**t) for t in p['series'][key]]; result[key] = []
        for row in ref['phase_diagnostics'][key]['pristine_frozen']['test']['decisions'][:3]:
            index, quote = quote_at(ticks, row['decision_ts'])
            rgb = frame(ticks, index, news)
            if not quote.available or quote.ts != row['quote_ts'] or hashlib.sha256(rgb.tobytes()).hexdigest() != row['neural']['input_sha256']:
                raise ValueError('Reconstructed input differs from original observation')
            if news.features(quote.ts).tolist() != row['neural']['news_features']:
                raise ValueError('Reconstructed news differs from original observation')
            result[key].append((rgb, row))
    return result


class AppliedCurrent:
    """Observe the exact extra-current arguments forwarded to the native RGB step."""
    def __init__(self, brain, indices, current):
        self.brain = brain; self.indices = np.asarray(indices, dtype=np.int32)
        self.current = current; self.rows = []; self.original = brain.rgb_step
        self.had_override = 'rgb_step' in vars(brain)

    def __enter__(self):
        self.brain.rgb_step = self.step
        return self

    def __exit__(self, *exc):
        if self.had_override: self.brain.rgb_step = self.original
        else: del self.brain.rgb_step

    def step(self, rgb, duration_ms, **kwargs):
        pulses = kwargs.get('stimulation')
        pulses = [] if pulses is None else pulses if isinstance(pulses, list) else [pulses]
        if self.current == 0:
            if pulses: raise ValueError('Zero-current control received stimulation')
            actual = np.zeros(len(self.indices))
        else:
            if len(pulses) != 1 or not np.array_equal(pulses[0][0], self.indices) or pulses[0][1] != self.current:
                raise ValueError('Native stimulation targets or current differ')
            actual = np.full(len(self.indices), pulses[0][1], dtype=float)
        self.rows.append((float(duration_ms), actual))
        return self.original(rgb, duration_ms, **kwargs)

    def save(self, path):
        durations = np.array([x[0] for x in self.rows]); currents = np.array([x[1] for x in self.rows])
        if durations.shape != (50,) or not np.all(durations == 10) or currents.shape != (50, 2):
            raise ValueError('Expected exactly 500 ms of recorded native stimulation')
        np.savez_compressed(path, durations_ms=durations, currents=currents, target_indices=self.indices)


def run(payload, memory_root, news_archive, data, output):
    p, ref, plan = validate(payload); seq = sequences(payload); root = Path(output)
    if root.exists(): raise ValueError('Refuse to overwrite a stimulation experiment')
    news_check = audit_news(payload['paper_plan'], news_archive)
    root.mkdir(parents=True); atomic_json(root/'payload.json', payload)
    shutil.copyfile(news_archive, root/'news.db')
    deadline = time.monotonic()+480; lab = TraceLab(data); b = lab.brain
    if b.build != ref['native_build']: raise ValueError('Native build differs from original control')
    for identity in TARGETS:
        if identity not in lab.id_index or str(lab.types[lab.id_index[identity]]) != 'MBON11':
            raise ValueError('Stimulation target is not its annotated retained MBON11 cell')
    indices = [lab.id_index[x] for x in TARGETS]
    imported = imported_memory(b, plan, memory_root); pristine = learned_state(b)
    if memory_signature(pristine) != ref['pristine_memory_sha256']: raise ValueError('Pristine memory differs')
    initial = dynamic_state(b)
    np.savez_compressed(root/'initial-dynamics.npz', **initial)
    np.savez_compressed(root/'neuron-ids.npz', neuron_ids=b.ids)
    np.savez_compressed(root/'pristine-memory.npz', **pristine)
    (root/'imported').mkdir(); shutil.copyfile(Path(memory_root)/'audit.json', root/'imported/audit.json')
    for i in range(2): shutil.copyfile(Path(memory_root)/f'pool{i}-memory.npz', root/f'imported/pool{i}-memory.npz')
    reports = {}; controls = []; start = time.monotonic()
    for name, arm in p['arms'].items():
        if arm['current'] and len(controls) != 4: raise ValueError('Unstimulated controls have not passed')
        memory = pristine if arm['memory'] == 'pristine' else imported[arm['pool']]
        restore_learned(b, memory); b.eta = .001; b.weights_frozen = True
        lab.fly.controller.s = replace(lab.fly.controller.s, learning=False)
        if any(not np.array_equal(v, getattr(b, k)) for k, v in initial.items()): raise ValueError('Arm did not start fresh')
        folder = root/name; folder.mkdir(); np.savez_compressed(folder/'initial-memory.npz', **memory)
        events = []; wall = time.monotonic(); whole_weights = hashlib.sha256(b.weight.tobytes()).hexdigest()
        references = ref['phase_diagnostics'][arm['pool']][arm['memory']+'_frozen']['test']['decisions']
        for j, (rgb, row) in enumerate(seq[arm['pool']], 1):
            if time.monotonic() > deadline: raise TimeoutError('Bounded stimulation experiment expired')
            boundary = apply_boundary(b, 'carry', j, folder/f'boundary-{j:02}.npz')
            with AppliedCurrent(b, indices, arm['current']) as applied:
                event = lab.capture(rgb, 'none', folder, j, indices, arm['current'])
            applied.save(folder/f'current-{j:02}.npz')
            if memory_signature(learned_state(b)) != memory_signature(memory) or hashlib.sha256(b.weight.tobytes()).hexdigest() != whole_weights:
                raise ValueError('Frozen inference changed memory or graph weights')
            if event['input_sha256'] != row['neural']['input_sha256']: raise ValueError('Input hash differs')
            if not arm['current'] and not matches(event, references[j-1]['neural']):
                raise ValueError('Original unstimulated control did not reproduce: '+name)
            event.update(phase='replay', phase_step=j, market_decision_ts=row['decision_ts'],
                quote_ts=row['quote_ts'], input_preset='recorded_market', input_news='sealed_timestamped',
                news_features=row['neural']['news_features'], activity_boundary=boundary,
                stimulation={'target_ids': TARGETS, 'current': arm['current'], 'duration_ms': 500,
                             'artifact_sha256': digest(folder/f'current-{j:02}.npz')})
            events.append(event)
        if not arm['current']: controls.append(name)
        report = {'schema': 1, 'source': SOURCE, 'config': {'preset': 'recorded_market', 'view': 'original',
            'news': 'sealed_timestamped', 'pulses': 'none', **INFERENCE, **arm, 'neurons': TARGETS},
            'graph': {'neurons': b.n, 'edges': len(b.post), 'plastic_edges': len(b.circuit['edges'])},
            'native_build': b.build, 'upstream_commit': UPSTREAM_COMMIT, 'events': events,
            'seconds': time.monotonic()-wall, 'initial_memory_sha256': memory_signature(memory),
            'final_memory_sha256': memory_signature(learned_state(b)), 'all_weight_sha256': whole_weights,
            'reference_report_sha256': p['reference_report_sha256'], 'plan_sha256': p['plan_sha256'],
            'pool': arm['pool'], 'interpretation': INTERPRETATION}
        atomic_json(folder/'report.json', report)
        # Include active plastic sources and both intervention targets in the paired viewer.
        requested = indices+[lab.id_index[x] for x in ('18540','21778','44069','520206')]
        view = lab.export_view(folder, report, requested); attach_boundary_state(view, folder)
        atomic_json(folder/'view.json', view); reports[name] = report
        atomic_json(root/'progress.json', {'completed': list(reports), 'controls_verified': controls})
        print('recipient_stimulation '+json.dumps({'arm': name, 'actions': [e['side'] for e in events],
              'gate': [e['gate_spikes'] for e in events]}), flush=True)
    result = {'status': 'paper_stimulation_completed', 'protocol': p, 'reports': reports,
        'controls_verified': controls, 'seconds': time.monotonic()-start, 'news_audit': news_check,
        'initial_dynamics_sha256': digest(root/'initial-dynamics.npz'), 'neuron_ids_sha256': digest(root/'neuron-ids.npz'),
        'code_sha256': {n: digest(Path(__file__).with_name(n)) for n in FILES}}
    atomic_json(root/'summary.json', result)
    return result
