"""Plot every audited selective trace condition without running a model."""
import argparse
import json
from pathlib import Path

from .core import atomic_json, digest


MODES = ('carry', 'reset_rates', 'reset_kc', 'reset_dan')
LABELS = ('Carry both', 'Reset both', 'Reset KC only', 'Reset DAN only')
COLORS = ('#ffbf69', '#67e8cf', '#8fbcff', '#e6a8e8')
GROUPS = ('trained_online_recorded', 'trained_online_none', 'trained_frozen_recorded')
GROUP_LABELS = ('Online + recorded pulses', 'Online + no injected pulses', 'Frozen + recorded pulses')
CHECKS = ('all_weights_exact', 'all_six_original_controls_reproduced',
          'all_boundaries_verified', 'frozen_selective_counts_and_voltage_unchanged',
          'all_decoders_reconstructed', 'all_displayed_topology_and_series_verified')


def load_audit(path):
    a = json.loads(Path(path).read_text())
    names = {f'{group}_{mode}' for group in GROUPS for mode in MODES}
    v = a['verification']
    if (a['status'] != 'selective_trace_audited' or set(a['arms']) != names
            or set(a['protocol']['arms']) != names
            or v['observations'] != 36 or v['bins'] != 1800
            or any(v.get(k) is not True for k in CHECKS)
            or any(len(rows) != 3 for rows in a['arms'].values())):
        raise ValueError('The complete independently audited 12-condition assay is required')
    expected = {(f'{g}_{c}', f'{g}_{m}') for g in GROUPS
                for c in ('carry', 'reset_rates') for m in ('reset_kc', 'reset_dan')}
    actual = [(p['control'], p['intervention']) for p in a['comparisons']]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('All selective comparisons against both original controls are required')
    for p in a['comparisons']:
        if [o['observation'] for o in p['observations']] != [1, 2, 3]:
            raise ValueError('Incomplete comparison observations')
        for o in p['observations']:
            for field in ('u', 'w', 'weights', 'voltage', 'counts'):
                f = o['fields'][field]
                if len(f['different_entities_per_bin']) != 50 or len(f['difference_l2_per_bin']) != 50:
                    raise ValueError('Incomplete comparison series')
    return a


def render(audit_path, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    a = load_audit(audit_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    bg, fg, muted = '#0e1728', '#e8edf6', '#b9c6da'
    assets = {}

    def save(fig, stem):
        for extension in ('.png', '.svg'):
            path = output / (stem + extension)
            fig.savefig(path, dpi=150, facecolor=bg, metadata={'Creator': 'Stonkfly paper lab'})
            if extension == '.svg':
                path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
            assets[path.name] = digest(path)
        plt.close(fig)

    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10,
                        'text.color': fg, 'axes.labelcolor': fg, 'axes.titlecolor': fg,
                        'xtick.color': muted, 'ytick.color': muted,
                        'axes.edgecolor': '#52617a', 'svg.fonttype': 'none',
                        'text.parse_math': False}):
        fig = plt.figure(figsize=(13, 8), facecolor=bg)
        ax = fig.add_axes([.035, .19, .93, .64]); ax.axis('off')
        fig.text(.035, .95, 'Separate trace resets change actions, but amplify later weight movement',
                 fontsize=18, weight='bold')
        fig.text(.035, .90, 'Same trained memory, three historical images and full native graph. All twelve conditions are shown.', color=muted)
        rows = []
        for group, group_label in zip(GROUPS, GROUP_LABELS):
            for mode, label in zip(MODES, LABELS):
                events = a['arms'][group+'_'+mode]
                rows.append([group_label, label, *[e['side'] for e in events],
                             ' / '.join(str(e['gate_spikes']) for e in events),
                             f"{events[-1]['weight_update_l2']:.4f}"])
        table = ax.table(cellText=rows,
                         colLabels=['Learning / pulses', 'Between images', 'Image 1', 'Image 2', 'Image 3', 'Gate spikes 1/2/3', 'Image 3 Δ L2'],
                         colWidths=[.24, .17, .09, .09, .09, .14, .12], cellLoc='left', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False); table.set_fontsize(9)
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor('#35445c'); cell.set_facecolor('#25334b' if r == 0 else '#172237' if ((r-1)//4)%2 == 0 else '#1d2b42')
            value = cell.get_text().get_text()
            cell.set_text_props(color={'BUY': '#67e8cf', 'SELL': '#ff9a9e', 'HOLD': muted}.get(value, fg))
            if c == 1 and r: cell.set_text_props(color=COLORS[(r-1)%4])
        fig.text(.035, .135, 'Δ L2 measures weight movement during image 3 across 7,835 plastic connections; it is not a loss or gradient.', color=muted)
        fig.text(.035, .095, 'Six original controls reproduce. Frozen resets preserve recorded activity and memory. 36 observations / 1,800 bins audited.', color=muted)
        fig.text(.035, .055, 'No fills, accounts or returns are recomputed here. Matching HOLD decisions do not mean matching neural trajectories.', color=muted)
        fig.text(.035, .02, 'Audit SHA-256: '+digest(audit_path)[:32]+'…', fontsize=9, color=muted)
        save(fig, 'matrix')

        fig, axes = plt.subplots(3, 2, figsize=(13, 10.5), facecolor=bg)
        fig.subplots_adjust(left=.08, right=.975, top=.865, bottom=.18, hspace=.50, wspace=.30)
        fig.text(.035, .955, 'A one-bin match does not predict the later trajectory', fontsize=21, weight='bold')
        fig.text(.035, .92, 'Selective resets compared with resetting both traces. Each point uses the same saved input and native-time bin.', color=muted)
        for row, (group, label) in enumerate(zip(GROUPS, GROUP_LABELS)):
            for mode, color, mode_label in zip(MODES[2:], COLORS[2:], LABELS[2:]):
                pair = next(p for p in a['comparisons'] if p['control'] == group+'_reset_rates'
                            and p['intervention'] == group+'_'+mode)
                xs = [x for o in pair['observations'] for x in o['times_ms']]
                axes[row, 0].plot(xs, [n for o in pair['observations'] for n in o['fields']['weights']['difference_l2_per_bin']],
                                  color=color, lw=1.8, label=mode_label)
                axes[row, 1].plot(xs, [n for o in pair['observations'] for n in o['fields']['counts']['different_entities_per_bin']],
                                  color=color, lw=1.6, label=mode_label)
            axes[row, 0].set_title(label, loc='left', fontsize=11)
            axes[row, 0].set_ylabel('Weight difference L2\nversus reset both')
            axes[row, 1].set_ylabel('Neurons with different counts\nper 10 ms bin')
            axes[row, 1].set_yscale('symlog', linthresh=1)
            axes[row, 1].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:,.0f}'))
            if row == 2:
                for ax in axes[row]:
                    ax.set_yscale('linear'); ax.set_ylim(-.1, .1); ax.set_yticks([0])
            for ax in axes[row]:
                ax.set_facecolor('#172237'); ax.grid(alpha=.12)
                ax.set_xlim(0, 1500); ax.set_xticks([0, 500, 1000, 1500])
                for x in (500, 1000): ax.axvline(x, color=muted, lw=.9, ls=':')
                if row == 2: ax.set_xlabel('Native elapsed time (ms)')
            if row == 0: axes[row, 0].legend(frameon=False, labelcolor=fg, fontsize=9)
        fig.text(.035, .09, 'Boundary resets occur before images 2 and 3, at 500 and 1,000 ms. All first observations match their controls.', color=muted)
        fig.text(.035, .055, 'Counts cover all 166,700 neurons. These are full propagated recordings, not a causal path inferred from sampled order.', color=muted)
        fig.text(.035, .02, 'No profitable policy has been selected. The experiment tests a learning mechanism on three historical images.', color=muted)
        save(fig, 'trajectories')

    comparisons = []
    for pair in a['comparisons']:
        comparisons.append({'control': pair['control'], 'intervention': pair['intervention'],
            'observations': [{'observation': o['observation'], 'decoder': o['decoder'],
                'fields': {name: {'first': f['first'], 'final_different_entities': f['different_entities_per_bin'][-1],
                                 'final_difference_l2': f['difference_l2_per_bin'][-1]}
                           for name, f in o['fields'].items()}} for o in pair['observations']]})
    result = {'status': 'audited_selective_figures_rendered', 'audit_sha256': digest(audit_path),
              'source_sha256': digest(__file__), 'assets_sha256': assets,
              'arms': {name: [{k: e[k] for k in ('observation', 'side', 'gate_spikes', 'difference_hz', 'weight_update_l2')}
                              for e in rows] for name, rows in a['arms'].items()},
              'comparisons': comparisons, 'model_submissions': 0,
              'interpretation': 'All 12 conditions from the independently audited historical mechanism assay. No account, profit, new training or policy selection.'}
    atomic_json(output/'figures.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    r = render(args.audit, args.out)
    print(json.dumps({'status': r['status'], 'assets_sha256': r['assets_sha256']}, indent=2))


if __name__ == '__main__':
    main()
