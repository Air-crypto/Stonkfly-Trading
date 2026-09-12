"""Plot an audited paper-memory comparison without running the neural model."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from .fly_market_figure import phase_series
from .fly_paper_protocol import ARMS, development_choice
from .fly_activation_protocol import ARMS as ACTIVATION_ARMS

PHASES = ('development', 'test')
LABELS = {
    'pristine_frozen': 'Pristine / carry input state',
    'trained_frozen': 'Paper-trained / carry input state',
    'pristine_input_reset': 'Pristine / reset input state',
    'trained_input_reset': 'Paper-trained / reset input state',
}
ACTIVATION_LABELS = {
    'pristine_frozen': 'Pristine / current 0',
    'trained_frozen': 'Paper-trained / current 0',
    'pristine_stimulated': 'Pristine / current 10',
    'trained_stimulated': 'Paper-trained / current 10',
}
CHECKS = ('news_reconstructed', 'imported_memory_verified',
          'all_paper_ledgers_replayed', 'all_full_count_decoders_verified',
          'activity_boundaries_audited', 'selection_matches_development_gate')


def presentation(registration):
    """Only the two registered experiments have supported matched comparisons."""
    if registration['arms'] == ACTIVATION_ARMS and registration.get('study') == '10':
        return ACTIVATION_LABELS, {'current_0': ('pristine_frozen', 'trained_frozen'),
                                   'current_10': ('pristine_stimulated', 'trained_stimulated')}
    if registration['arms'] == ARMS and registration.get('study', '09') == '09':
        return LABELS, {'carry': ('pristine_frozen', 'trained_frozen'),
                        'reset': ('pristine_input_reset', 'trained_input_reset')}
    raise ValueError('Unknown registered paper comparison conditions')


def evidence(report):
    """Require reconciled ledgers and the saved development choice before plotting."""
    r = report['registration']
    labels, pairs = presentation(r)
    arms = tuple(labels)
    checks = CHECKS + (('prospective_cloud_registration_verified', 'native_recipient_current_verified')
                      if r.get('study') == '10' else ())
    if (report.get('status') != 'paper_checkpoint_study_completed'
            or r.get('kind') != 'paper_checkpoint_comparison'
            or set(report['total_equity']) != set(arms)
            or report['initial_capital'] != 1000
            or report['costs'] != r['costs']
            or any(report.get('verification', {}).get(k) is not True for k in checks)):
        raise ValueError('Use a completed, audited paper-checkpoint report')
    if set(report['phase_diagnostics']) != set(r['cohort']):
        raise ValueError('Figure cohort differs from registration')
    panels = {}
    for arm in arms:
        if set(report['total_equity'][arm]) != set(PHASES):
            raise ValueError('Expected separate development and test; no training account')
        for phase in PHASES:
            if not math.isfinite(report['total_equity'][arm][phase]):
                raise ValueError('Nonfinite reported equity')
            ts, equity, unavailable = phase_series(report, arm, phase)
            expected = [r[phase + '_start'] + i * r['decision_seconds']
                        for i in range(r['phase_steps'] + 1)]
            if ts != expected or not all(math.isfinite(x) for x in equity):
                raise ValueError('Figure timeline or equity differs from registration')
            lanes = [report['phase_diagnostics'][key][arm][phase] for key in r['cohort']]
            fills = fees = observed = decisions = 0
            for lane in lanes:
                executed = [d['fill'] for d in lane['decisions'] if d['fill']['status'] == 'filled']
                actual_fees = sum(float(f['fee']) for f in executed)
                if (not math.isfinite(actual_fees) or not math.isfinite(lane['fees'])
                        or actual_fees < 0 or len(executed) != lane['fills']
                        or abs(actual_fees - lane['fees']) > 1e-8):
                    raise ValueError('Figure fills or fees do not reconcile to ledger')
                fills += len(executed)
                fees += actual_fees
                eligible_slots = [d for d in lane['decisions'] if not d['terminal']]
                if r.get('study') == '10':
                    declared = r['arms'][arm]
                    for row in lane['decisions']:
                        if row['neural'] is None:
                            continue
                        stimulus = row['neural'].get('stimulation', {})
                        sha = stimulus.get('artifact_sha256', '')
                        if (row['terminal'] or not row['available']
                                or stimulus != {'target_ids': declared['recipient_ids'],
                                                'current': declared['recipient_current'],
                                                'duration_ms': 500, 'artifact_sha256': sha}
                                or not isinstance(sha, str) or len(sha) != 64
                                or any(c not in '0123456789abcdef' for c in sha)):
                            raise ValueError('Figure recipient stimulation differs from audited conditions')
                observed += sum(d['neural'] is not None for d in eligible_slots)
                decisions += len(eligible_slots)
            panels[arm, phase] = {'times': ts, 'equity': equity, 'unavailable': unavailable,
                                  'fills': fills, 'fees': fees, 'observed': observed,
                                  'decisions': decisions}
    development = {a: report['total_equity'][a]['development'] for a in arms}
    selection = report['selection']
    if (selection['development_equity'] != development
            or selection['selected'] != development_choice(development)
            or selection['test_simulated_before_selection'] is not False
            or selection['plan_sha256'] != report['plan_sha256']):
        raise ValueError('Figure selection differs from the saved development gate')
    # Memory effects hold the activity reset or applied current equal on both sides.
    effects = {phase: {label: report['total_equity'][trained][phase] - report['total_equity'][pristine][phase]
                       for label, (pristine, trained) in pairs.items()} for phase in PHASES}
    return panels, effects


def render(report, output, *, fixture=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    panels, effects = evidence(report)
    r = report['registration']
    labels, pairs = presentation(r)
    activation = r.get('study') == '10'
    arms = tuple(labels)
    effect_labels = {'carry': 'Carry input state', 'reset': 'Reset input state',
                     'current_0': 'Current 0', 'current_10': 'Current 10'}
    colors = dict(zip(arms, ('#8fbcff', '#67e8cf', '#ffbf69', '#e6a8e8')))
    background, foreground, muted = '#0e1728', '#e8edf6', '#b9c6da'
    with plt.rc_context({'font.size': 11, 'font.family': 'DejaVu Sans',
                         'text.color': foreground, 'axes.labelcolor': foreground,
                         'xtick.color': muted, 'ytick.color': muted,
                         'axes.edgecolor': '#52617a', 'svg.fonttype': 'none'}):
        fig = plt.figure(figsize=(14, 12.8), facecolor=background, layout=None)
        grid = fig.add_gridspec(3, 2, left=.085, right=.97, top=.755, bottom=.17,
                               height_ratios=[2, 1.35, 1.1], hspace=.45, wspace=.16)
        title = ('Does activating these neurons improve paper trading?' if activation
                 else 'Paper-trained connection memory: does it help?')
        if fixture:
            title = 'SYNTHETIC FIXTURE — layout and audit check'
        fig.text(.035, .965, title, fontsize=22, weight='bold')
        fig.text(.035, .94, 'Full retained fly graph · frozen inference · identical timestamp-eligible prices and news', fontsize=12)
        fig.text(.035, .918, ('Each phase starts with $1,000 and fresh activity. Current 0 or 10 to MBON11 cells 10704/11402 for 500 ms/image.'
                            if activation else 'Each phase starts with $1,000 and fresh activity. Only connection weights/u/w are imported.'), color=muted, fontsize=10.5)
        fig.text(.035, .886, 'Pinned paper-training exposure (paper rewards are not counts of winning/losing trades):', fontsize=11)
        for i, key in enumerate(r['cohort']):
            x = r['source_memories'][key]
            fig.text(.035, .865 - i * .022,
                     f"Pool {i}: {x['observations']} observations; {x['positive_rewards']} positive and "
                     f"{x['negative_rewards']} negative rewards. {key[:30]}…", color=muted, fontsize=10)
        all_values = [v - 1000 for p in panels.values() for v in p['equity']] + [0]
        low, high = min(all_values), max(all_values)
        pad = max(high - low, 1) * .18
        stamp = lambda t: datetime.fromtimestamp(t, timezone.utc).strftime('%H:%M')
        legend_handles = []
        for j, phase in enumerate(PHASES):
            ax = fig.add_subplot(grid[0, j], facecolor='#172237')
            for index, arm in enumerate(arms):
                p = panels[arm, phase]
                xs = [(t - p['times'][0]) / 60 for t in p['times']]
                ys = [v - 1000 for v in p['equity']]
                line, = ax.plot(xs, ys, color=colors[arm], lw=2,
                                linestyle='--' if index >= 2 else '-',
                                marker='s' if index >= 2 else 'o', markersize=4,
                                label=labels[arm])
                if not j:
                    legend_handles.append(line)
                for x, y, missing in zip(xs, ys, p['unavailable']):
                    if missing:
                        ax.scatter(x, y, color='#ff727d', marker='x', s=75, zorder=10)
            ax.axhline(0, color='#75859c', lw=1, ls=':')
            ax.set_ylim(low - pad, high + pad)
            ax.set_xticks([0, 5, 10, 15], [stamp(r[phase + '_start'] + i * 300) for i in range(4)])
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'${v:+.2f}'))
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.set_title(phase.title(), loc='left', fontsize=15, color=foreground)
            ax.set_xlabel('Decision time (UTC)')
            ax.grid(alpha=.12)
            if not j:
                ax.set_ylabel('Simulated P&L from $1,000')
            else:
                ax.tick_params(labelleft=False)
            table_ax = fig.add_subplot(grid[1, j]);table_ax.axis('off')
            rows = []
            for arm in arms:
                p = panels[arm, phase]
                rows.append([labels[arm].replace(' input state', ''), f"${p['equity'][-1]:,.2f}",
                             str(p['fills']), f"${p['fees']:.3f}", f"{p['observed']}/{p['decisions']}"])
            table = table_ax.table(cellText=rows, colLabels=['Condition', 'End equity', 'Fills', 'Fees', 'Observed'],
                                   colWidths=[.43, .21, .10, .14, .12], bbox=[0, 0, 1, 1], cellLoc='left')
            table.auto_set_font_size(False);table.set_fontsize(8.8)
            for (row, col), cell in table.get_celld().items():
                cell.set_facecolor('#172237' if row else '#25334b')
                cell.set_edgecolor('#35445c');cell.set_text_props(color=foreground)
                if row and col == 0:
                    cell.set_text_props(color=colors[arms[row - 1]])
        ax = fig.add_subplot(grid[2, :], facecolor='#172237')
        ax.set_title('Matched memory effect: paper-trained minus pristine', loc='left', color=foreground, fontsize=13)
        effect_values = [v for values in effects.values() for v in values.values()]
        bound = max(.1, *(abs(v) for v in effect_values)) * 1.45
        for j, phase in enumerate(PHASES):
            for k, (reset, color) in enumerate(zip(pairs, ('#67e8cf', '#e6a8e8'))):
                value = effects[phase][reset];x = j + (k - .5) * .3
                ax.bar(x, value, width=.25, color=color,
                       label=effect_labels[reset] if j == 0 else None)
                ax.annotate(f'${value:+.3f}', (x, value), xytext=(0, 5 if value >= 0 else -12),
                            textcoords='offset points', ha='center', fontsize=10)
        ax.axhline(0, color=muted, lw=1);ax.set_ylim(-bound, bound)
        ax.set_xticks([0, 1], ['Development', 'Test'])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'${v:+.2f}'))
        ax.yaxis.set_major_locator(MaxNLocator(4));ax.set_ylabel('Equity difference')
        ax.legend(loc='upper center', ncols=2, frameon=False, labelcolor=foreground, fontsize=9)
        fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(.53, .817),
                   ncols=2, frameon=False, labelcolor=foreground, fontsize=10)
        selected = report['selection']['selected']
        gate = ('No trained condition beat cash and both pristine controls.' if selected is None
                else labels[selected] + ' passed the development gate; no automatic promotion.')
        fig.text(.035, .126, 'Development selection: ' + gate, fontsize=12)
        fig.text(.035, .102, 'Red ×: at least one pool quote unavailable. Missing inventory uses stress valuation; curves can overlap.', fontsize=10, color=muted)
        fig.text(.035, .08, 'Simulated DEX fees/slippage included; hosting excluded. Short two-pool comparison, not a monthly return forecast.', fontsize=10, color=muted)
        fig.text(.035, .058, 'Positive memory effect describes this interval only. Test results cannot change the saved development choice.', fontsize=10, color=muted)
        fig.text(.035, .035, ('SYNTHETIC INPUTS AND MEMORY — not market results. ' if fixture else '')
                 + 'Plan SHA-256: ' + report['plan_sha256'][:24] + '…', fontsize=9, color=muted)
        path = Path(output);path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fig.savefig(path, dpi=140, facecolor=background)
        finally:
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--fixture', action='store_true', help='Prominently mark synthetic regression inputs')
    args = parser.parse_args()
    render(json.loads(args.report.read_text()), args.out, fixture=args.fixture)


if __name__ == '__main__':
    main()
