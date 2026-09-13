"""Plot the six retained, audited conditions; exclude the preempted recording."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
COLORS = ['#8fbcff', '#67e8cf', '#ffbf69', '#e6a8e8']
LABELS = ['Pristine frozen', 'Trained frozen', 'Online carry', 'Online rate reset']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('original', 'recovery', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists(): raise ValueError('Preserve previous plots')
    protocol = json.loads((ROOT/'reports/fly-online-completion-protocol-11.json').read_text())
    series = []
    for i, (name, pin) in enumerate(protocol['retained'].items()):
        base = args.original if i < 2 else args.recovery
        summary = base/'chunks'/name/'summary.json'
        projection = ROOT/'reports/online-completion-retained'/f'{name}.json'
        if sha(summary) != pin['summary_sha256'] or sha(projection) != pin['projection_sha256']:
            raise ValueError('Retained evidence differs: ' + name)
        s = json.loads(summary.read_text()); a = json.loads(projection.read_text()); rows = s['outcome']['rows']
        if (a['validation_only'] or not a['projection']['all_plotted_series_verified']
                or len(rows) != 25 or s['stage'] != 'development' or s['chunk'] != name):
            raise ValueError('Expected a complete audited development recording')
        event_images = [r['event']['input_sha256'] if r['event'] else None for r in rows]
        series.append({'chunk': name, 'pool': s['pool_index'], 'arm': s['arm'],
            'summary_sha256': sha(summary), 'minutes': [(r['decision_ts']-rows[0]['decision_ts'])/60 for r in rows],
            'equity': [r['equity'] for r in rows], 'available': [r['available'] for r in rows],
            'exposure': [100*(r['equity']-float(r['broker']['cash']))/r['equity'] for r in rows],
            'event_images': event_images, 'fees': s['outcome']['fees'], 'fills': s['outcome']['fills']})
    for pool in range(2):
        pair = [s for s in series if s['pool'] == pool]
        if any(s['minutes'] != pair[0]['minutes'] or s['event_images'] != pair[0]['event_images'] for s in pair):
            raise ValueError('Inputs differ within a pool')
    plt.rcParams.update({'figure.facecolor': '#0d1523', 'axes.facecolor': '#152034', 'savefig.facecolor': '#0d1523',
        'text.color': '#edf1fc', 'axes.labelcolor': '#c5cfe0', 'xtick.color': '#c5cfe0', 'ytick.color': '#c5cfe0',
        'axes.edgecolor': '#435269', 'font.size': 10, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey='row',
                            gridspec_kw={'height_ratios': [2, 1]})
    for pool in range(2):
        group = [s for s in series if s['pool'] == pool]
        for i, s in enumerate(group):
            y = [v-250 for v in s['equity']]; x = s['minutes']
            axes[0, pool].plot(x, y, color=COLORS[i], lw=2, label=f'{LABELS[i]}: ${s["equity"][-1]:.2f}')
            gaps = [j for j, present in enumerate(s['available']) if not present]
            axes[0, pool].scatter([x[j] for j in gaps], [y[j] for j in gaps], color='#ff727d', marker='x', s=40)
            axes[1, pool].plot(x, s['exposure'], color=COLORS[i], lw=1.8)
        axes[0, pool].axhline(0, color='#bac4d2', ls=':', lw=1)
        axes[0, pool].set_title(('ALL: four completed conditions', 'baton: two completed controls')[pool], loc='left')
        axes[0, pool].legend(loc='lower left', frameon=False, fontsize=9)
        axes[1, pool].set_ylim(0, 65); axes[1, pool].set_xticks(range(0, 121, 20))
        axes[1, pool].set_xlabel('Minutes since development start')
        for ax in axes[:, pool]: ax.grid(alpha=.13); ax.spines[['top', 'right']].set_visible(False)
    axes[0, 0].set_ylabel('Paper P&L from $250 ($)'); axes[1, 0].set_ylabel('Marked inventory / equity (%)')
    axes[0, 1].text(.03, .58, 'Carry excluded: preempted\nRate reset not captured',
        transform=axes[0, 1].transAxes, color='#c5cfe0', fontsize=10)
    axes[0, 1].annotate('Missing quote;\ninventory marked at $0', xy=(65, -87), xytext=(78, -62),
        color='#c5cfe0', fontsize=9, arrowprops={'arrowstyle': '->', 'color': '#c5cfe0'})
    fig.suptitle('Six audited development conditions: all lost money', x=.075, y=.975,
                 ha='left', fontsize=19, weight='bold')
    fig.text(.075, .925, 'September 12, 2026 · 20:35–22:35 UTC · incomplete comparison; no held-out result', color='#bac4d2')
    fig.text(.075, .06, 'Simulated fees, spread and slippage included; hosting excluded. Red ×: unavailable quote with conservative marking.', fontsize=9, color='#bac4d2')
    fig.text(.075, .033, 'Audited frozen aggregates: pristine $946.73, trained $942.86 from $1,000. Online aggregates are not yet available.', fontsize=9, color='#bac4d2')
    fig.subplots_adjust(left=.075, right=.97, top=.855, bottom=.15, hspace=.19, wspace=.12)
    args.out.mkdir(parents=True)
    for extension in ('png', 'svg'): fig.savefig(args.out/('development-six.'+extension), dpi=150)
    plt.close(fig)
    result = {'status': 'six_audited_development_conditions_plotted', 'partial_development_only': True,
        'selection': None, 'held_out_results': False, 'series': series, 'source_sha256': sha(__file__),
        'png_sha256': sha(args.out/'development-six.png'), 'svg_sha256': sha(args.out/'development-six.svg'),
        'new_neural_observations': 0, 'cloud_submissions': 0}
    (args.out/'report.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__': main()
