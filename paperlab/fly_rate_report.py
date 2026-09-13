"""Explain completed study 12 audits without rerunning a neural model."""
import argparse
import json
from pathlib import Path

from .core import Costs, Tick, atomic_json, digest
from .fly_loss_attribution import COMPONENTS, decompose
from .fly_market_study import signature
from .fly_rate_inputs import validate
from .fly_rate_protocol import ARMS, chunk_name, chunk_order
from .fly_rate_schedule import aggregate
from .fly_rate_study import select_development
from .fly_trade_trace import trace

LABELS = dict(zip(ARMS, ('Pristine frozen', 'Trained frozen', 'Online eta 0.001', 'Online eta 0.0001')))
COLORS = dict(zip(ARMS, ('#3166b0', '#13866d', '#bf7017', '#9d409f')))
CHECKS = ('all_ledgers_and_feedback_reconstructed', 'all_full_bin_audits_passed',
          'all_raw_prices_reconstructed', 'all_news_clocks_verified', 'all_view_projections_verified',
          'source_and_native_builds_match', 'selection_precedes_every_test_chunk')


def require(value, message):
    if not value:
        raise ValueError(message)


def validate_chunk(report, summary, audited, receipt, name):
    require(signature(summary) == report['chunk_sha256'][name]
            and summary['chunk'] == name and audited['chunk'] == name
            and receipt['chunk'] == name and receipt['status'] == 'completed'
            and summary['plan_sha256'] == audited['plan_sha256'] == report['plan_sha256']
            and audited['status'] == 'paper_rate_chunk_audited'
            and audited['artifact_sha256'] == summary['artifact_sha256']
            and audited['executed_source_sha256'] == summary['code_sha256'],
            'Chunk evidence differs: ' + name)
    rows = summary['outcome']['rows']
    events = [r['event'] for r in rows if r['event'] is not None]
    v = audited['verification']
    require(v['decision_slots'] == len(rows) == 25 and v['observations'] == len(events)
            and v['native_bins'] == 50 * len(events)
            and all(v.get(k) is True for k in ('ledger_replayed', 'feedback_reconstructed',
                                             'all_boundaries_verified', 'all_weights_exact')),
            'Incomplete chunk audit: ' + name)
    require(len(audited['observations']) == len(events), 'Missing audited observations')
    for event, checked in zip(events, audited['observations']):
        require(event['side'] == checked['side'] and event['gate_spikes'] == checked['gate_spikes']
                and event['weight_delta_l2'] == checked['weight_norm_audit']['reported']
                and event['equity_reward_usd'] == checked['equity_reward_usd'],
                'Neural diagnostic differs from audit')


def build(root, audit, output, *, baselines=None):
    root, audit, output = map(Path, (root, audit, output))
    require(not output.exists(), 'Preserve an existing report; choose a new output directory')
    read = lambda path: json.loads(path.read_text())
    report = read(audit/'report.json'); envelope = read(root/'plan.json')
    p = validate(envelope); r = p['registration']; names = [chunk_name(*c) for c in chunk_order()]
    require(report.get('status') == 'paper_rate_study_audited' and report.get('audited') is True
            and report['registration'] == r and r['study'] == '12'
            and report['plan_sha256'] == envelope['sha256']
            and all(report.get('verification', {}).get(k) is True for k in CHECKS)
            and report['verification'].get('distinct_completed_worker_calls') == 16
            and set(report['audit_sha256']) == set(report['projection_sha256']) == set(names),
            'Require the complete independent study 12 audit')
    require(read(audit/'price-audit.json') == report['price_audit'], 'Price audit differs')
    summaries = {}; chunks = {}; calls = set(); costs = Costs(**r['costs'])
    for stage, pool, arm in chunk_order():
        name = chunk_name(stage, pool, arm); summary_path = root/'chunks'/name/'summary.json'
        summary = read(summary_path); checked = read(audit/'audits'/(name+'.json'))
        receipt = read(root/'receipts'/(name+'-completed.json'))
        require(digest(audit/'audits'/(name+'.json')) == report['audit_sha256'][name]
                and digest(audit/'projections'/(name+'.json')) == report['projection_sha256'][name]
                and digest(summary_path) == receipt['summary_sha256'], 'Evidence hash differs: ' + name)
        validate_chunk(report, summary, checked, receipt, name)
        require(summary['registration'] == r and summary['stage'] == stage
                and summary['pool_index'] == pool and summary['arm'] == arm
                and summary['outcome'] == report['phase_diagnostics'][r['cohort'][pool]][arm][stage],
                'Reported outcome differs from full audit')
        require(receipt['call_id'] not in calls, 'Repeated cloud owner'); calls.add(receipt['call_id'])
        ticks = [Tick(**t) for t in p['series'][r['cohort'][pool]]]
        rows = summary['outcome']['rows']; accounting = decompose(rows, ticks, costs)
        decisions = trace(rows, ticks, costs)
        chunks[name] = {'stage': stage, 'pool': pool, 'arm': arm, 'accounting': accounting,
                        'trace': decisions, 'summary_sha256': digest(summary_path),
                        'weight_update_l2': [0 if row['event'] is None else row['event']['weight_delta_l2'] for row in rows],
                        'observed': [row['event'] is not None for row in rows]}
        summaries[name] = summary
    selection = select_development(envelope, {k:v for k,v in summaries.items() if v['stage'] == 'development'})
    expected = aggregate(envelope, summaries, selection)
    require(all(report[k] == expected[k] for k in expected if k not in ('status', 'audited', 'interpretation')),
            'Aggregate or registered selection differs')
    panels = {}
    for stage in ('development', 'test'):
        panels[stage] = {}
        for arm in ARMS:
            pair = [chunks[chunk_name(stage, pool, arm)] for pool in range(2)]
            terminal = {k:sum(v['accounting']['terminal'][k] for v in pair) for k in COMPONENTS}
            equity = [500 + sum(v['accounting']['rows'][i]['recorded_equity'] for v in pair) for i in range(25)]
            require(abs(1000 + terminal['net_pnl_usd'] - report['total_equity'][arm][stage]) < 1e-8,
                    'Attribution does not reconcile')
            panels[stage][arm] = {'equity': equity, 'terminal': terminal,
                'fills': sum(v['trace']['fills'] for v in pair),
                'single_gate_fills': sum(v['trace']['single_gate_fills'] for v in pair),
                'turnover_usd': sum(v['accounting']['terminal']['turnover_usd'] for v in pair),
                'weight_update_l2_sum_by_slot': [sum(v['weight_update_l2'][i] for v in pair) for i in range(25)],
                'observations': [sum(v['observed']) for v in pair],
                'unavailable_marks': sum(v['accounting']['missing_marks'] for v in pair),
                'terminal_quotes_usable': all(v['accounting']['terminal']['available'] for v in pair)}
    reference = None
    if baselines is not None:
        reference = read(Path(baselines))
        require(reference['status'] == 'descriptive_market_baselines' and reference['study'] == '12'
                and reference['plan_sha256'] == envelope['sha256'] and reference['costs'] == r['costs']
                and reference['provenance']['price_audit'] == report['price_audit'], 'Baseline inputs differ')
    fixture = any('synthetic' in t['source'] for series in p['series'].values() for t in series)
    require(report['validation_only'] == fixture, 'Synthetic-data label differs')
    result = {'status': 'audited_rate_diagnostics', 'study': '12', 'validation_only': fixture,
        'source_report_sha256': digest(audit/'report.json'), 'plan_sha256': envelope['sha256'],
        'registration': r, 'selection': selection, 'panels': panels, 'chunks': chunks,
        'baselines': None if reference is None else reference['aggregate'],
        'baselines_sha256': None if baselines is None else digest(baselines),
        'source_sha256': {name:digest(Path(__file__).with_name(name)) for name in (
            'fly_rate_report.py', 'fly_loss_attribution.py', 'fly_trade_trace.py')},
        'verification': {'audited_conditions': len(chunks),
            'account_marks': sum(len(v['accounting']['rows']) for v in chunks.values()),
            'observations': sum(len(v['trace']['decisions']) for v in chunks.values()),
            'fills': sum(v['trace']['fills'] for v in chunks.values()),
            'maximum_identity_error_usd': max(v['accounting']['maximum_identity_error_usd'] for v in chunks.values())},
        'new_neural_observations': 0, 'cloud_submissions': 0, 'policy_promoted': False,
        'interpretation': 'Fixed-fill accounting and recorded decision traces, not a zero-cost policy replay. '
            'Weight-update norms are not gradient loss or trading success. Missing inventory is stress-marked, not sold. '
            'Baselines are descriptive and have different exposure. Hosting is excluded. No test-based reselection.'}
    output.mkdir(parents=True); atomic_json(output/'report.json', result); return result


def export_case(root, report, output):
    """First action divergence in development pool 0; full-array checks, no model."""
    import numpy as np
    root, output = Path(root), Path(output)
    arms = ('trained_online_carry', 'trained_online_low_eta')
    names = [chunk_name('development', 0, arm) for arm in arms]
    decisions = [report['chunks'][name]['trace']['decisions'] for name in names]
    require([v['decision_ts'] for v in decisions[0]] == [v['decision_ts'] for v in decisions[1]],
            'Paired decision times differ')
    index = next((i for i, (a,b) in enumerate(zip(*decisions)) if a['decision'] != b['decision']), None)
    require(index is not None, 'No development action divergence to inspect')
    case = {'status':'audited_rate_first_action_difference', 'study':'12', 'validation_only':report['validation_only'],
            'selection_rule':'First differing action in development pool 0, standard versus lower eta.',
            'source_report_sha256':digest(output/'report.json'), 'observation':index, 'runs':{},
            'interpretation':'Recorded decoder consequence of differing learning rates. The selected edge is an independent example, not a demonstrated causal path to the gate. Decisions are decoded after 500 ms; bin times mark interval ends.'}
    prefixes = []
    for arm, name, actions in zip(arms, names, decisions):
        folder = root/'chunks'/name; summary = json.loads((folder/'summary.json').read_text())
        require(digest(folder/'summary.json') == report['chunks'][name]['summary_sha256'], 'Case summary changed')
        events = [row['event'] for row in summary['outcome']['rows'] if row['event'] is not None]
        prefixes.append([{k:event[k] for k in ('market_decision_ts','input_sha256','stimulus','stimulus_ms','equity_reward_usd')}
                         for event in events[:index+1]])
        files = ('neuron-ids.npz', 'circuit.npz', f'trace/step-{index+1:02d}.npz')
        require(all(digest(folder/f) == summary['artifact_sha256'][f] for f in files), 'Case recording changed')
        with np.load(folder/files[0], allow_pickle=False) as a: ids = a['neuron_ids'].astype(str)
        with np.load(folder/files[1], allow_pickle=False) as a: edge = int(np.flatnonzero(a['edges'] == 8022240)[0])
        with np.load(folder/files[2], allow_pickle=False) as a:
            counts = {node:a['counts'][:,int(np.flatnonzero(ids == node)[0])].astype(int).tolist()
                      for node in ('10162','10059','10527','555871')}
            values = {field:a[field][:,edge].astype(float).tolist() for field in ('weights','u','w')}
        action = actions[index]
        require(2*sum(counts['10162']) == action['left_hz'] and 2*sum(counts['10059']) == action['right_hz']
                and sum(counts['10527'])+sum(counts['555871']) == action['gate_spikes'], 'Raw decoder differs')
        case['runs'][arm] = {'name':name, 'action':action, 'counts':counts, 'edge':8022240, **values,
                            'artifact_sha256':{f:summary['artifact_sha256'][f] for f in files}}
    require(prefixes[0] == prefixes[1], 'Input or feedback history differs before the first action divergence')
    case['matched_input_and_feedback_prefix'] = prefixes[0]
    atomic_json(output/'case.json', case)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with plt.rc_context({'font.size':10, 'svg.fonttype':'none', 'text.parse_math':False}):
        fig, axes = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True)
        fig.suptitle(('SYNTHETIC VALIDATION — ' if case['validation_only'] else '')
                     +f'First changed action: observation {index+1}, development RAY', fontsize=18)
        x = np.arange(10, 501, 10)
        weights = [v for item in case['runs'].values() for v in item['weights']]
        margin = max((max(weights)-min(weights))*.07, .001)
        gate_counts = np.array(case['runs'][arms[1]]['counts']['10527']) + np.array(case['runs'][arms[1]]['counts']['555871'])
        gate_bins = np.flatnonzero(gate_counts)
        bin_text = ', '.join(f'{10*b}–{10*(b+1)}' for b in gate_bins) or 'none'
        directions = []
        for item in case['runs'].values():
            left = 2*np.cumsum(item['counts']['10162']); right = 2*np.cumsum(item['counts']['10059'])
            directions.extend(np.concatenate((left,right,right-left)).tolist())
        direction_margin = max(1, (max(directions)-min(directions))*.08)
        gate_max = max(sum(v['counts'][node]) for v in case['runs'].values() for node in ('10527','555871'))
        for column, arm in enumerate(arms):
            item = case['runs'][arm]; counts = item['counts']; action = item['action']
            left = 2*np.cumsum(counts['10162']); right = 2*np.cumsum(counts['10059'])
            axes[0,column].plot(x, left, label='Left contribution'); axes[0,column].plot(x, right, label='Right contribution')
            axes[0,column].plot(x, right-left, label='Right minus left', color='#31835a')
            axes[0,column].set(title=LABELS[arm]+' → '+action['decision'], ylabel='Accumulated contribution (Hz)',
                               ylim=(min(directions)-direction_margin,max(directions)+direction_margin))
            axes[0,column].legend(fontsize=8)
            axes[1,column].step(x, np.cumsum(counts['10527']), where='post', label='Gate 10527')
            axes[1,column].step(x, np.cumsum(counts['555871']), where='post', label='Gate 555871')
            axes[1,column].set(ylabel='Accumulated gate spikes', ylim=(-.05,max(1,gate_max)*1.2)); axes[1,column].legend(fontsize=8)
            axes[2,column].plot(x, item['weights'], color=COLORS[arm])
            axes[2,column].set(ylabel='Edge 8022240 weight', xlabel='Time within image (ms)',
                               ylim=(min(weights)-margin,max(weights)+margin))
            for row in range(3):
                axes[row,column].grid(alpha=.2)
                for b in gate_bins: axes[row,column].axvspan(10*b,10*(b+1),color='#777777',alpha=.12)
        fig.supxlabel(f'Same preceding actions, price images and feedback. Full decision at 500 ms; lower-rate gate spike bins: {bin_text} ms.\n'
                      'The edge is a separately inspected connection, not a proven path causing this spike. No new neural run or policy promotion.', fontsize=10)
        for ext in ('png','svg'): fig.savefig(output/('case.'+ext), dpi=160)
        plt.close(fig)
    return case


def render(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    output = Path(output)
    with plt.rc_context({'font.size': 9, 'svg.fonttype': 'none', 'text.parse_math': False}):
        fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
        title = ('Study 12: lower learning rate did not pass selection' if report['selection']['selected'] is None
                 else 'Study 12: development candidate selected; inspect fresh test outcomes')
        if report['validation_only']: title = 'SYNTHETIC VALIDATION — study 12 diagnostics'
        fig.suptitle(title, fontsize=19)
        for column, stage in enumerate(('development', 'test')):
            panels = report['panels'][stage]
            for arm, panel in panels.items():
                axes[0,column].plot(range(25), panel['equity'], label=LABELS[arm], color=COLORS[arm])
                axes[2,column].plot(range(24), panel['weight_update_l2_sum_by_slot'][:24],
                                    label=LABELS[arm], color=COLORS[arm])
            axes[0,column].axhline(1000, color='#666666', linestyle='--', linewidth=1, label='Cash')
            axes[0,column].set(title=stage.capitalize()+' — full account marks', ylabel='Paper equity ($)', xlabel='Five-minute slot')
            axes[0,column].legend(fontsize=8, loc='lower left')
            names = list(ARMS); pos = list(range(4)); width = .36
            axes[1,column].bar([v-width/2 for v in pos], [panels[a]['terminal']['gross_reference_pnl_usd'] for a in names],
                               width, label='Fixed-fill midpoint component', color='#3b80af')
            axes[1,column].bar([v+width/2 for v in pos], [panels[a]['terminal']['net_pnl_usd'] for a in names],
                               width, label='Net after execution / exit costs', color='#b34d58')
            axes[1,column].axhline(0, color='#888888', linewidth=.8)
            axes[1,column].set(xticks=pos, xticklabels=['Pristine', 'Trained frozen', 'Eta .001', 'Eta .0001'], ylabel='Terminal P&L ($)')
            axes[1,column].legend(fontsize=8, loc='lower center', bbox_to_anchor=(.5,1.01))
            axes[2,column].set(yscale='symlog', ylabel='Sum of two sleeve update L2 norms', xlabel='Five-minute slot',
                               title='Recorded weight changes; zero also includes unobserved slots')
            axes[2,column].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
            for row in range(3): axes[row,column].grid(axis='y', alpha=.2)
        fig.supxlabel('Separate $1,000 accounts per arm and phase. Quote gaps remain visible; they are not realized crashes.\n'
                      'Fixed-fill midpoint component is not a zero-cost strategy. Hosting excluded. No policy promoted.', fontsize=10)
        for ext in ('png', 'svg'): fig.savefig(output/('figure.'+ext), dpi=160)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True); p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--baselines', type=Path)
    p.add_argument('--plot', action='store_true'); p.add_argument('--case', action='store_true'); a = p.parse_args()
    result = build(a.root, a.audit, a.out, baselines=a.baselines)
    if a.plot: render(result, a.out)
    if a.case: export_case(a.root, result, a.out)
    print(json.dumps({'status':result['status'], 'verification':result['verification'],
                      'selected':result['selection']['selected']}, indent=2))


if __name__ == '__main__': main()
