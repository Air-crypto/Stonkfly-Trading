"""Plot audited recipient isolation without training, trading, or selecting a policy."""
import argparse
import json
from pathlib import Path

import numpy as np

from .core import atomic_json, digest
from .fly_recipient_isolation import INFERENCE, SETS

TARGETS = ('10704', '11402')
POOL_LABELS = {'solana:uzAK8txfJAvqS9VHtDaYAWfBiYbzwq4BzxdiFJTQEnJ':'ALL',
               'solana:Cb7ZRgPLhji3Th7htXyKEqbXvNjPpeTvckXuWakBUhXu':'baton'}


def evidence(a):
    p = a['protocol']; arms = p['arms']
    names = [f'pool{i}-{memory}-{target}' for target in SETS for i in (0, 1) for memory in ('pristine', 'trained')]
    controls = names[:4]
    if (a.get('status') != 'verified' or a.get('full_observations_verified') != 36
            or a.get('native_bins_verified') != 1800 or p.get('kind') != 'paper_memory_recipient_isolation'
            or p['target_sets'] != SETS or p['current'] != 10 or p['inference'] != INFERENCE or list(arms) != names
            or list(a['rows']) != names or a['controls_verified'] != controls
            or len(a.get('reference_control_artifact_sha256', {})) != 12):
        raise ValueError('Requires all twelve audited isolation conditions and original control evidence')
    for name, arm in arms.items():
        _, memory, target = name.split('-', 2)
        pool = arms[name.split('-')[0]+'-pristine-both']['pool']
        if arm != {'pool':pool, 'memory':memory, 'current':10, 'target_ids':SETS[target], 'target_set':target}:
            raise ValueError('Plot stimulation differs from declared targets')
        rows = a['rows'][name]
        if len(rows) != 3:raise ValueError('Missing observed image')
        for j, row in enumerate(rows, 1):
            counts = row['recipient_counts']; difference = row['difference_hz']; gate = row['gate_spikes']
            if (row['observation'] != j or set(counts) != set(TARGETS)
                    or any(type(v) is not int or v < 0 for v in counts.values())
                    or type(difference) is not int or type(gate) is not int or gate < 0):
                raise ValueError('Invalid recipient or decoder counts')
            side = 'HOLD' if gate == 0 or abs(difference) < 2 else 'BUY' if difference > 0 else 'SELL'
            if side != row['side']:raise ValueError('Recorded action differs from fixed decoder')
    expected = {(f'pool{i}-{memory}-both', f'pool{i}-{memory}-{target}')
                for i in (0, 1) for memory in ('pristine', 'trained') for target in ('only_10704', 'only_11402')}
    comparisons = a['isolation_comparisons']
    if len(comparisons) != 8 or {(c['baseline'], c['variant']) for c in comparisons} != expected:
        raise ValueError('Missing matched single-target comparison')
    for c in comparisons:
        base, variant = a['rows'][c['baseline']], a['rows'][c['variant']]
        if (len(c['changed_neurons']) != 3 or len(c['different_actions']) != 3
                or any(type(n) is not int or n < 0 or n > 166700 for n in c['changed_neurons'])
                or c['different_actions'] != [x['side'] != y['side'] for x, y in zip(base, variant)]
                or any((n == 0) != (x['spike_sha256'] == y['spike_sha256'])
                       for n, x, y in zip(c['changed_neurons'], base, variant))):
            raise ValueError('Isolation comparison does not reconcile with recorded actions or counts')
    return names


def render(audit_file, output, *, fixture=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    a = json.loads(Path(audit_file).read_text()); names = evidence(a); out = Path(output)
    if out.exists():raise ValueError('Refuse to overwrite an isolation figure')
    arms = a['protocol']['arms']; labels = {'both': 'Both', 'only_10704': '10704 only', 'only_11402': '11402 only'}
    pools = [POOL_LABELS.get(arms[f'pool{i}-pristine-both']['pool'], f'Pool {i}') for i in (0, 1)]
    with plt.rc_context({'font.family':'DejaVu Sans', 'font.size':10, 'svg.fonttype':'none',
                         'axes.spines.top':False, 'axes.spines.right':False}):
        fig = plt.figure(figsize=(14, 12), facecolor='#f7f8fa')
        grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.5], hspace=.47, wspace=.5,
                               left=.19, right=.96, top=.86, bottom=.16)
        for pool in (0, 1):
            ax = fig.add_subplot(grid[0, pool]); ax.set_facecolor('#f7f8fa')
            order = [f'pool{pool}-{memory}-{target}' for target in SETS for memory in ('pristine', 'trained')]
            bottom = np.zeros(6)
            for identity, color in zip(TARGETS, ('#426cc1', '#bd4a24')):
                values = np.array([sum(r['recipient_counts'][identity] for r in a['rows'][name]) for name in order])
                ax.bar(range(6), values, bottom=bottom, width=.72, color=color, label=identity)
                bottom += values
            for i, total in enumerate(bottom):ax.annotate(str(int(total)), (i, total), xytext=(0, 5), textcoords='offset points', ha='center', fontsize=9)
            ax.set_ylim(0, max(1, max(bottom))*1.5)
            ax.set_xticks(range(6), [labels[arms[n]['target_set']].replace(' only', '')+'\n'+('Pristine' if arms[n]['memory']=='pristine' else 'Trained') for n in order], fontsize=8)
            ax.set_xlabel('Directly stimulated target(s)', fontsize=9)
            ax.set_ylabel('Recipient spikes across three images'); ax.set_title(pools[pool]+' · both recipients recorded', loc='left', weight='bold')
            ax.legend(title='Recorded neuron', frameon=False, ncols=2, fontsize=9); ax.grid(axis='y', alpha=.15)
        ax = fig.add_subplot(grid[1, 0]); comparisons = a['isolation_comparisons']
        matrix = np.array([c['changed_neurons'] for c in comparisons])
        im = ax.imshow(matrix, cmap='Blues', aspect='auto', vmin=0, vmax=max(1, int(matrix.max())))
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):ax.text(j, i, f'{value:,}', ha='center', va='center', fontsize=8,
                                                  color='white' if value > matrix.max()*.6 else '#152a40')
        row_labels = []
        for c in comparisons:
            arm = arms[c['variant']]; pool = int(c['variant'][4]); row_labels.append(f"{pools[pool]} / {arm['memory']} / {labels[arm['target_set']]}")
        ax.set_yticks(range(8), row_labels, fontsize=8); ax.set_xticks(range(3), ['Image 1', 'Image 2', 'Image 3'])
        ax.set_title('Single target versus both\nNeurons with different 500 ms totals', loc='left', weight='bold', fontsize=11)
        fig.colorbar(im, ax=ax, orientation='horizontal', fraction=.07, pad=.1, label='Changed neuron counts')
        ax = fig.add_subplot(grid[1, 1]); order = sorted(names, key=lambda n:(int(n[4]), list(SETS).index(arms[n]['target_set']), arms[n]['memory']))
        actions = {'SELL':0, 'HOLD':1, 'BUY':2}; matrix = np.array([[actions[r['side']] for r in a['rows'][n]] for n in order])
        ax.imshow(matrix, cmap=ListedColormap(['#f0c4b8','#dce1e8','#bddcce']), vmin=0, vmax=2, aspect='auto')
        for i, name in enumerate(order):
            for j, row in enumerate(a['rows'][name]):ax.text(j, i, row['side'], ha='center', va='center', fontsize=8)
        ax.set_yticks(range(12), [f"{pools[int(n[4])]} / {labels[arms[n]['target_set']]} / {arms[n]['memory']}" for n in order], fontsize=8)
        ax.set_xticks(range(3), ['Image 1', 'Image 2', 'Image 3']); ax.set_title('Fixed decoder · all conditions', loc='left', weight='bold', fontsize=11)
        fig.suptitle('SYNTHETIC FIXTURE — layout only' if fixture else 'Which stimulated recipient changes the response?',
                     x=.06, y=.973, ha='left', fontsize=20, weight='bold')
        fig.text(.06, .932, 'Current 10 to the declared target(s) · frozen memory · same six historical images · activity carried between images', fontsize=11)
        fig.text(.06, .901, 'All four both-target controls must reproduce the original full-neuron count bins before isolation.', fontsize=10, color='#42536a')
        fig.text(.06, .106, 'A neuron can spike when it is not directly stimulated. Counts alone do not separate direct current from network feedback.', fontsize=10, color='#42536a')
        fig.text(.06, .082, 'The heatmap compares full-observation totals, not spike timing. Different neural responses need not change the decoded action.', fontsize=10, color='#42536a')
        fig.text(.06, .058, 'Post-hoc mechanism diagnostic: no trading account, returns, learning, or automatic policy promotion.', fontsize=10, color='#42536a')
        fig.text(.06, .031, ('SYNTHETIC DISPLAY DATA — not model or market results. ' if fixture else '')+'Audit SHA-256: '+digest(audit_file)[:24]+'…', fontsize=9, color='#42536a')
        out.mkdir(parents=True)
        try:
            for suffix in ('png', 'svg'):fig.savefig(out/('isolation.'+suffix), dpi=150, facecolor=fig.get_facecolor())
        finally:plt.close(fig)
    atomic_json(out/'provenance.json', {'audit_sha256':digest(audit_file), 'source_sha256':digest(__file__), 'fixture':fixture,
                'artifacts':{s:digest(out/('isolation.'+s)) for s in ('png','svg')}})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--fixture', action='store_true'); a = p.parse_args(); render(a.audit, a.out, fixture=a.fixture)


if __name__ == '__main__':main()
