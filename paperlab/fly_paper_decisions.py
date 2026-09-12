"""Compare audited decisions and lagged fills; no model execution or selection."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from urllib.parse import urlencode

from .core import atomic_json, digest
from .fly_paper_figure import PHASES, evidence
from .fly_paper_study import trace_name

PAIRS = {
    'memory_carry': ('pristine_frozen', 'trained_frozen'),
    'memory_reset': ('pristine_input_reset', 'trained_input_reset'),
    'reset_pristine': ('pristine_frozen', 'pristine_input_reset'),
    'reset_trained': ('trained_frozen', 'trained_input_reset'),
}
ACTIVATION_PAIRS = {
    'memory_current_0': ('pristine_frozen', 'trained_frozen'),
    'memory_current_10': ('pristine_stimulated', 'trained_stimulated'),
    'activation_pristine': ('pristine_frozen', 'pristine_stimulated'),
    'activation_trained': ('trained_frozen', 'trained_stimulated'),
}


def neural_summary(event):
    if event is None:
        return None
    left, right, gate = (event[k] for k in ('left_hz', 'right_hz', 'gate_spikes'))
    if any(not isinstance(x, (int, float)) or x < 0 or x % 1 for x in (left, right, gate)):
        raise ValueError('Invalid recorded decoder counts')
    if left % 2 or right % 2 or event['difference_hz'] != right-left:
        raise ValueError('Decoder rates differ from 500 ms spike counts')
    side = 'HOLD' if gate == 0 or abs(right-left) < 2 else 'BUY' if right > left else 'SELL'
    if event['side'] != side or event['plasticity_enabled'] or event['weight_delta_l2'] != 0:
        raise ValueError('Expected frozen, fixed-decoder inference')
    result = {k: event[k] for k in ('side', 'left_hz', 'right_hz', 'difference_hz', 'gate_spikes',
                                   'total_spikes', 'spike_sha256', 'input_sha256')}
    if 'stimulation' in event:
        result['stimulation'] = dict(event['stimulation'])
    return result


def compare(report):
    evidence(report)
    r = report['registration']; comparisons = []
    activation = r.get('study') == '10'
    pairs = ACTIVATION_PAIRS if activation else PAIRS
    for i, key in enumerate(r['cohort']):
        for phase in PHASES:
            for label, (baseline, variant) in pairs.items():
                a = report['phase_diagnostics'][key][baseline][phase]['decisions']
                b = report['phase_diagnostics'][key][variant][phase]['decisions']
                if len(a) != len(b):
                    raise ValueError('Unaligned comparison timelines')
                rows = []; observed = 0
                for x, y in zip(a, b):
                    if any(x[k] != y[k] for k in ('decision_ts', 'quote_ts', 'available', 'terminal')):
                        raise ValueError('Comparison does not use identical quote slots')
                    ex, ey = x['neural'], y['neural']
                    if (ex is None) != (ey is None):
                        raise ValueError('Comparison observation coverage differs')
                    nx, ny = neural_summary(ex), neural_summary(ey)
                    row = {'decision_ts': x['decision_ts'], 'quote_ts': x['quote_ts'],
                           'available': x['available'], 'terminal': x['terminal'],
                           'baseline': nx, 'variant': ny,
                           'baseline_fill': x['fill'], 'variant_fill': y['fill'],
                           'fills_equal': x['fill'] == y['fill'],
                           'baseline_equity': x['equity'], 'variant_equity': y['equity'],
                           'equity_difference': y['equity'] - x['equity'],
                           'observation_index': None, 'viewer_query': None}
                    if ex is not None:
                        if (x['terminal'] or not x['available']
                                or ex['input_sha256'] != ey['input_sha256']
                                or ex['news_features'] != ey['news_features']):
                            raise ValueError('Cannot compare memory, activity or stimulation with different market inputs')
                        row.update(observation_index=observed, input_match=True,
                                   full_counts_equal=ex['spike_sha256'] == ey['spike_sha256'],
                                   decoder_equal=all(ex[k] == ey[k] for k in ('left_hz', 'right_hz', 'gate_spikes')),
                                   side_equal=ex['side'] == ey['side'],
                                   direction_difference_hz=ey['difference_hz'] - ex['difference_hz'],
                                   gate_difference=ey['gate_spikes'] - ex['gate_spikes'])
                        row['viewer_query'] = '?' + urlencode({'run': trace_name(i, variant, phase),
                                                               'step': observed, 'compare': trace_name(i, baseline, phase)})
                        observed += 1
                    rows.append(row)
                first = lambda field: next((x['decision_ts'] for x in rows if x.get(field) is False), None)
                comparisons.append({'pool_index': i, 'pool': key, 'phase': phase, 'comparison': label,
                                    'baseline_arm': baseline, 'variant_arm': variant,
                                    'first_full_count_difference': first('full_counts_equal'),
                                    'first_decoder_difference': first('decoder_equal'),
                                    'first_side_difference': first('side_equal'),
                                    'first_fill_difference': first('fills_equal'), 'rows': rows})
    result = {'kind': 'paper_checkpoint_decision_comparison', 'plan_sha256': report['plan_sha256'],
            'selection': report['selection'], 'comparisons': comparisons,
            'interpretation': 'Derived from an audited report. Full-count differences refer to whole observations, not the first spike in time. Current fills can execute an earlier decision. Identical sides can coexist with different neural counts. This comparison neither reruns nor selects a policy.'}
    if activation:
        effects = {}
        for phase in PHASES:
            e = {a: report['total_equity'][a][phase] for a in r['arms']}
            pristine = e['pristine_stimulated'] - e['pristine_frozen']
            trained = e['trained_stimulated'] - e['trained_frozen']
            effects[phase] = {'activation_pristine_usd': pristine,
                              'activation_trained_usd': trained,
                              'memory_current_0_usd': e['trained_frozen'] - e['pristine_frozen'],
                              'memory_current_10_usd': e['trained_stimulated'] - e['pristine_stimulated'],
                              'memory_activation_interaction_usd': trained - pristine}
        result['activation_effects'] = effects
        result['interpretation'] += (' Applied current is explicitly recorded for both conditions. '
            'Activation effects compare current 10 with 0 within the same memory. The interaction subtracts '
            'the pristine activation effect from the trained activation effect; a gain shared by both brains '
            'is not a learned-memory advantage. These are descriptive interval effects, not statistical significance.')
    return result


def markdown(result):
    stamp = lambda t: datetime.fromtimestamp(t, timezone.utc).strftime('%H:%M:%S')
    lines = ['# Recorded decision comparisons', '', result['interpretation'], '',
             'Times are UTC. Equity differences are variant minus baseline, per $250 pool sleeve. '
             'Open viewer links after serving the downloaded study directory on port 8765.', '']
    def decision(e):
        if e is None:return '—'
        current = f" · current {e['stimulation']['current']}" if 'stimulation' in e else ''
        return f"{e['side']} · Δ {e['difference_hz']:+g} Hz · gates {e['gate_spikes']}" + current
    def fill(f):
        if f['status'] != 'filled':return f['status']
        return f"{f['side']} from {stamp(f['decision_ts'])} · fee ${float(f['fee']):.3f}"
    if 'activation_effects' in result:
        lines += ['Current is applied to cells 10704 and 11402 for each 500 ms observation. All weights remain frozen.', '',
                  '| Phase | Activation, pristine | Activation, trained | Memory at current 0 | Memory at current 10 | Interaction |',
                  '|---|---:|---:|---:|---:|---:|']
        for phase, effects in result['activation_effects'].items():
            values = ' | '.join(f'${value:+.4f}' for value in effects.values())
            lines.append(f'| {phase} | {values} |')
        lines.append('')
    for c in result['comparisons']:
        lines += [f"## Pool {c['pool_index']} · {c['phase']} · {c['comparison']}", '',
                  f"Baseline: `{c['baseline_arm']}`. Variant: `{c['variant_arm']}`.", '',
                  '| Time | Baseline decision | Variant decision | Full counts | Baseline fill now | Variant fill now | Equity difference | Trace |',
                  '|---|---|---|---|---|---|---:|---|']
        for row in c['rows']:
            counts = ('same' if row['full_counts_equal'] else 'different') if row['baseline'] else ('end' if row['terminal'] else 'unobserved')
            time = stamp(row['decision_ts']) + (' (end)' if row['terminal'] else '')
            link = f"[compare](http://127.0.0.1:8765/{row['viewer_query']})" if row['viewer_query'] else '—'
            lines.append(f"| {time} | {decision(row['baseline'])} | {decision(row['variant'])} | {counts} | "
                         f"{fill(row['baseline_fill'])} | {fill(row['variant_fill'])} | ${row['equity_difference']:+.4f} | {link} |")
        lines.append('')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--out', type=Path, required=True, help='New output directory')
    args = parser.parse_args()
    if args.out.exists():raise ValueError('Refuse to overwrite a decision comparison')
    result = compare(json.loads(args.report.read_text()))
    result.update(report_sha256=digest(args.report), source_sha256=digest(__file__))
    args.out.mkdir(parents=True)
    atomic_json(args.out/'comparison.json', result)
    (args.out/'comparison.md').write_text(markdown(result))
    print(args.out/'comparison.md')


if __name__ == '__main__':main()
