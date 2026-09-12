"""Read-only output-count sensitivity audit using the retained native decoder."""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .core import atomic_json

ROOT = Path(__file__).resolve().parents[1]
CONTROLLER = ROOT / 'vendor/stonkfly/stonkfly/neural/controller.py'
CONFIG = ROOT / 'vendor/stonkfly/stonkfly/config.py'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native_decoder():
    # Compile only the exact decoder method. No graph load, network, or simulation.
    tree = ast.parse(CONTROLLER.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Decoder')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'decode')
    namespace = {'np': np}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(CONTROLLER), 'exec'), namespace)
    settings = next(n for n in ast.parse(CONFIG.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'Settings')
    defaults = {n.target.id: ast.literal_eval(n.value) for n in settings.body
                if isinstance(n, ast.AnnAssign) and n.target.id in ('neural_ms', 'decoder_threshold_hz')}
    return namespace['decode'], defaults


def output_counts(view, frame, seconds):
    event = frame['event']; ids = [str(n['id']) for n in view['nodes']]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate displayed neuron identity')
    groups = event['cell_ids']; all_ids = [str(i) for key in ('left', 'right', 'gate') for i in groups[key]]
    if len(set(all_ids)) != len(all_ids) or any(not groups[key] for key in ('left', 'right', 'gate')):
        raise ValueError('Decoder populations must be distinct and nonempty')
    if any(i not in ids for i in all_ids):
        raise ValueError('Missing decoder neuron in displayed recording')
    times = np.asarray(frame['times_ms'], dtype=float)
    counts = np.asarray(frame['counts'])
    if counts.shape != (len(times), len(ids)) or not np.isfinite(counts).all() or np.any(counts < 0) or np.any(counts != np.floor(counts)):
        raise ValueError('Invalid recorded spike counts')
    # These recordings use fixed 10 ms bins. Do not infer duration from last-first.
    expected = event['brain_ms'] - seconds*1000 + np.arange(1, len(times)+1)*10
    if len(times)*10 != seconds*1000 or not np.allclose(times, expected, rtol=0, atol=1e-6):
        raise ValueError('Recorded bins do not match the decoder observation duration')
    return groups, all_ids, counts[:, [ids.index(i) for i in all_ids]].sum(axis=0, dtype=np.int64)


def audit_view(view):
    decode, defaults = native_decoder(); seconds = defaults['neural_ms']/1000
    if not view['frames'] or [f['event'] for f in view['frames']] != view['report']['events']:
        raise ValueError('Frame events differ from the recorded report')
    results = []
    for observation, frame in enumerate(view['frames'], 1):
        groups, identities, counts = output_counts(view, frame, seconds)
        decoder = SimpleNamespace(threshold=defaults['decoder_threshold_hz'], identities=groups,
            **{k: np.array([identities.index(str(i)) for i in groups[k]]) for k in ('left','right','gate')})
        baseline = decode(decoder, counts, seconds); event = frame['event']
        if any(baseline[k] != event[k] for k in baseline):
            raise ValueError('Native default decoder does not reproduce the recorded event')
        edits = []
        for i, identity in enumerate(identities):
            for delta in (-1, 1):
                if counts[i] + delta < 0:
                    continue
                changed = counts.copy(); changed[i] += delta; output = decode(decoder, changed, seconds)
                edits.append({'body_id': identity, 'delta_spikes': delta, 'side': output['side'],
                    'difference_hz': output['difference_hz'], 'gate_spikes': output['gate_spikes'],
                    'action_changed': output['side'] != baseline['side']})
        results.append({'observation': observation, 'market_decision_ts': event.get('market_decision_ts'),
            'input_sha256': event.get('input_sha256'), 'spike_sha256': event.get('spike_sha256'),
            'baseline': baseline, 'counts_by_body': dict(zip(identities, map(int, counts))),
            'single_spike_changes_action': any(e['action_changed'] for e in edits), 'edits': edits})
    return {'settings': defaults, 'observations': results}


def audit(paths):
    runs = {}
    for path in paths:
        path = Path(path); name = path.parent.name
        if name in runs:
            raise ValueError('Duplicate recording name')
        runs[name] = {'view_sha256': digest(path), **audit_view(json.loads(path.read_text()))}
    totals = Counter()
    for run in runs.values():
        for row in run['observations']:
            totals['observations'] += 1
            totals['one_spike_sensitive'] += row['single_spike_changes_action']
            totals['side_'+row['baseline']['side']] += 1
    return {'schema':1, 'kind':'post_hoc_output_count_sensitivity',
        'controller_sha256':digest(CONTROLLER), 'settings_sha256':digest(CONFIG),
        'settings_basis':'Retained native defaults, checked against every recorded rate, gate and side; historical reports do not independently record the threshold.',
        'interpretation':'One spike is added to or removed from one output count after the observation. No neural propagation, plasticity, input, execution, or profit is recomputed. These edits are mathematical counterfactuals, not evidence of actual noise, a causal neural intervention, or profitable threshold tuning. Repeated controls are not independent samples.',
        'totals':dict(totals), 'runs':runs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--views', nargs='+', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); result = audit(args.views); atomic_json(args.out, result)
    print(json.dumps(result['totals'], indent=2))


if __name__ == '__main__':
    main()
