"""Reconstruct recorded plasticity and split its drive by trace origin.

No neural propagation, intervention, account replay or cloud submission. The
split is algebra on observed rates; it is not causal attribution to an image.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .core import atomic_json, digest
from .fly_pulse_audit import audit as pulse_audit
from .fly_trace_memory import enrich_frame

ROOT = Path(__file__).resolve().parents[1]
NEURAL = ROOT / 'vendor/stonkfly/stonkfly/neural'
ARMS = ('trained_online_recorded', 'trained_online_none', 'trained_frozen_recorded')
INTERPRETATION = (
    'Earlier-image and current-image trace contributions sum to the observed '
    'learning drive before memory filtering and clipping. These are algebraic '
    'components with full recorded activity held fixed, not causal effects of '
    'images, dopamine pulses, trades, or removing traces. No gradients, neural '
    'simulation, trading returns, or policy promotion are computed.'
)


def rule_module():
    # Load only the pure rate rule, without constructing a brain or native kernel.
    spec = importlib.util.spec_from_file_location('recorded_fly_rule', NEURAL / 'rule.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def close(actual, recorded, name):
    if (actual.shape != recorded.shape or not np.isfinite(recorded).all()
            or not np.allclose(actual, recorded, rtol=1e-11, atol=1e-12)):
        raise ValueError('Recorded rule state does not reproduce: ' + name)
    return float(np.max(np.abs(actual - recorded), initial=0))


def split_drive(kc, dan, boundary_kc, boundary_dan, rates_kc, rates_dan,
                gain, elapsed_seconds, eta, parameters):
    """Midpoint trace split, before frozen/learning gating, in u units/second."""
    h = .01
    ak = math.exp(-h / parameters['trace_kc_seconds'])
    ad = math.exp(-h / parameters['trace_dan_seconds'])
    kmid = kc * math.sqrt(ak) + rates_kc * (1 - math.sqrt(ak))
    dmid = dan * math.sqrt(ad) + rates_dan * (1 - math.sqrt(ad))
    old_k = boundary_kc * math.exp(-(elapsed_seconds + h/2) / parameters['trace_kc_seconds'])
    old_d = boundary_dan * math.exp(-(elapsed_seconds + h/2) / parameters['trace_dan_seconds'])
    previous = eta * (rates_kc * (gain.T @ old_d) - (gain.T @ rates_dan) * old_k)
    current = eta * (rates_kc * (gain.T @ (dmid-old_d)) - (gain.T @ rates_dan) * (kmid-old_k))
    total = eta * (rates_kc * (gain.T @ dmid) - (gain.T @ rates_dan) * kmid)
    close(previous + current, total, 'drive decomposition')
    return previous, current, total


def reconstruct(arrays, state, circuit, baseline, rule, learning, selection):
    """Verify every bin/edge and return a bounded display plus full-edge totals.

    state is the independently verified initial memory or preceding observation's
    reconstructed end state. All rate traces begin at zero in this scoped assay.
    """
    a = arrays; edges = circuit['edges']; pre = circuit['pre']; dan = circuit['dan']
    count = len(edges); n = len(a['neuron_ids'])
    for key, expected in (('plastic_edges', edges), ('plastic_pre', pre), ('plastic_post', circuit['post'])):
        if not np.array_equal(a[key], expected):
            raise ValueError('Recorded circuit identity differs: ' + key)
    counts = a['counts']
    if counts.shape != (50, n) or counts.dtype.kind not in 'iu' or np.any(counts < 0):
        raise ValueError('Invalid whole-network counts')
    for key, width in (('kc', count), ('dan', len(dan)), ('u', count), ('w', count), ('weights', count)):
        if a[key].shape != (50, width) or not np.isfinite(a[key]).all():
            raise ValueError('Invalid full recorded array: ' + key)
    if not np.array_equal(a['initial_weights'], state['weights']):
        raise ValueError('Initial weights differ from preceding state')
    boundary_kc = state['kc'].copy(); boundary_dan = state['dan'].copy()
    values = {k: [] for k in ('earlier', 'current', 'total', 'weight_step')}
    full = {k: [] for k in ('earlier_l1', 'current_l1', 'total_l1', 'opposed_edges')}
    errors = {k: 0. for k in ('kc', 'dan', 'u', 'w')}
    clipped = 0
    for i, row in enumerate(counts):
        K = row[pre] / .01; D = row[dan] / .01  # This assay used the default zero DAN baseline.
        earlier, current, total = split_drive(state['kc'], state['dan'], boundary_kc,
            boundary_dan, K, D, circuit['gain'], i*.01, .001, rule.PARAMETERS)
        if not learning:
            earlier = np.zeros(count); current = np.zeros(count); total = np.zeros(count)
        before = state['weights'].copy()
        rule.advance(state['kc'], state['dan'], state['u'], state['w'], K, D,
                     circuit['gain'], .01, .001, learning=learning, frozen=not learning)
        if learning:
            state['weights'] = (baseline*(1+state['w'])).astype(before.dtype)
        for key in errors:
            errors[key] = max(errors[key], close(state[key], a[key][i], key))
        if not np.array_equal(state['weights'], a['weights'][i]):
            raise ValueError('Recorded float32 weights do not reproduce exactly')
        lo = rule.PARAMETERS['minimum_fraction'] - 1; hi = rule.PARAMETERS['maximum_fraction'] - 1
        clipped += int(np.count_nonzero((state['u'] <= lo) | (state['u'] >= hi)
                                        | (state['w'] <= lo) | (state['w'] >= hi)))
        for key, value in (('earlier', earlier), ('current', current), ('total', total),
                           ('weight_step', state['weights'].astype(float)-before)):
            values[key].append(value[selection].tolist())
        for key, value in (('earlier_l1', earlier), ('current_l1', current), ('total_l1', total)):
            full[key].append(float(np.abs(value).sum()))
        full['opposed_edges'].append(int(np.count_nonzero(earlier*current < -1e-20)))
    values.update(schema=1, plastic_selection=selection.tolist(), times_ms=a['ms'].tolist(),
                  learning=learning, units='u per second', weight_step_units='synaptic weight per bin',
                  boundary='start of this 500 ms image', interpretation=INTERPRETATION)
    summary = {'integrated_absolute_drive': {k.removesuffix('_l1'): .01*sum(v)
                for k, v in full.items() if k.endswith('_l1')},
               'opposed_edge_bins': sum(full['opposed_edges']), 'saturated_edge_bins': clipped,
               'maximum_reconstruction_error': errors, 'all_weights_exact': True,
               'bins': full}
    return values, summary


def graph_arrays(data):
    from .fly import configure
    configure(data)
    from stonkfly.data import verify
    from stonkfly.neural.circuit import identify
    verify()  # Check prepared arrays and annotations against upstream locks.
    with np.load(Path(data)/'graph.npz', allow_pickle=False) as a:
        graph = SimpleNamespace(**{k: a[k] for k in ('ids', 'ptr', 'post', 'weight')})
    graph.n = len(graph.ids)
    circuit = identify(graph)
    circuit['post'] = graph.post[circuit['edges']]
    return graph.ids, circuit, graph.weight[circuit['edges']].copy()


def run(root, reference, data, output):
    root = Path(root); output = Path(output)
    if output.exists():
        raise ValueError('Write to a new output directory; preserve original recordings')
    study = json.loads((root/'summary.json').read_text())
    pulse_audit(study, Path(reference).read_text())
    ids, circuit, baseline = graph_arrays(data); rule = rule_module()
    results = {}; views = {}; files = {'summary.json': digest(root/'summary.json')}
    for name in ARMS:
        folder = root/name; view = json.loads((folder/'view.json').read_text())
        report = study['reports'][name]
        if view['report'] != report or len(view['frames']) != 3 or report['config']['eta'] != .001:
            raise ValueError('Original view or learning rate differs from study')
        if report['graph'] != {'neurons': len(ids), 'edges': 25582938, 'plastic_edges': len(baseline)}:
            raise ValueError('Recorded graph dimensions differ')
        with np.load(folder/'initial-memory.npz', allow_pickle=False) as a:
            state = {key: a[key].copy() for key in ('weights', 'u', 'w')}
        if any(v.shape != baseline.shape or not np.isfinite(v).all() for v in state.values()):
            raise ValueError('Invalid initial memory')
        memory_hash = hashlib.sha256(b''.join(state[k].tobytes() for k in ('weights', 'u', 'w'))).hexdigest()
        if memory_hash != report['initial_memory_sha256']:
            raise ValueError('Initial memory hash differs')
        if not np.array_equal(state['weights'], (baseline*(1+state['w'])).astype(np.float32)):
            raise ValueError('Initial memory does not reproduce prepared weights')
        state.update(kc=np.zeros(len(baseline)), dan=np.zeros(len(circuit['dan'])))
        expanded = copy.deepcopy(view); results[name] = []
        for filename in ('view.json', 'initial-memory.npz'):
            files[f'{name}/{filename}'] = digest(folder/filename)
        for i, frame in enumerate(view['frames']):
            event = frame['event']; path = folder/f'step-{i+1:02}.npz'
            if event != report['events'][i] or event['brain_ms'] != (i+1)*500:
                raise ValueError('Missing observation or changed activity boundary')
            if event['stimulus_ms'] != (0 if event['stimulus'] == 'none' else 200):
                raise ValueError('Recorded reinforcement duration differs')
            memory = enrich_frame(view, frame, path)
            with np.load(path, allow_pickle=False) as a:
                arrays = {k: a[k] for k in ('counts','neuron_ids','plastic_edges','plastic_pre',
                    'plastic_post','kc','dan','u','w','weights','initial_weights','ms')}
            if not np.array_equal(ids, arrays['neuron_ids']):
                raise ValueError('Native neuron order differs')
            values, summary = reconstruct(arrays, state, circuit, baseline, rule,
                report['config']['learning'], np.asarray(view['plastic_selection'], dtype=np.int64))
            summary.update(observation=i+1, stimulus=event['stimulus'], side=event['side'],
                           gate_spikes=event['gate_spikes'], market_decision_ts=event['market_decision_ts'])
            summary['pulse_segments'] = {}
            for label, sl in (('pulse_on', slice(0,20)), ('pulse_off', slice(20,50))):
                if event['stimulus'] == 'none':
                    continue
                summary['pulse_segments'][label] = {k.removesuffix('_l1'): .01*sum(v[sl])
                    for k,v in summary['bins'].items() if k.endswith('_l1')}
            expanded['frames'][i].update(**memory, plastic_credit=values)
            results[name].append(summary); files[f'{name}/{path.name}'] = digest(path)
        views[name] = expanded
    sources = {'audit': digest(__file__), **{k: digest(NEURAL/k) for k in
               ('rule.py','circuit.py','brain.py','controller.py','arrays.lock.json','sources.lock.json')}}
    result = {'schema': 1, 'source': 'recorded_plasticity_trace_origin', 'arms': results,
              'source_sha256': sources, 'artifact_sha256': files,
              'reference_sha256': digest(reference), 'rule_parameters': rule.PARAMETERS,
              'verification': {'observations': 9, 'bins': 450, 'plastic_edges_per_bin': len(baseline),
                  'prepared_graph_verified': True, 'full_counts_match_original_events': True,
                  'all_weights_exact': True, 'memory_tolerance': {'rtol':1e-11,'atol':1e-12},
                  'neural_observations': 0}, 'interpretation': INTERPRETATION}
    output.mkdir(parents=True)
    for name, view in views.items():
        view['credit_audit'] = {'schema':1, 'source_sha256':sources, 'interpretation':INTERPRETATION,
            'artifact_sha256':{k:v for k,v in files.items() if k.startswith(name+'/')},
            'study_sha256':files['summary.json'], 'reference_sha256':result['reference_sha256'],
            'verification':result['verification']}
        atomic_json(output/name/'view.json', view)
        atomic_json(output/name/'report.json', view['report'])
    result['view_sha256'] = {name:digest(output/name/'view.json') for name in ARMS}
    atomic_json(output/'audit.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('recordings','reference','fly-data','out'):
        p.add_argument('--'+key, type=Path, required=True)
    a = p.parse_args(); result = run(a.recordings, a.reference, a.fly_data, a.out)
    print(json.dumps({k:[{field:v for field,v in row.items() if field != 'bins'} for row in rows]
                      for k,rows in result['arms'].items()}, indent=2))


if __name__ == '__main__':
    main()
