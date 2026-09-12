"""Map verified paper-trained memory to native recipients; no neural propagation."""
import argparse
import json
from pathlib import Path

import numpy as np

from .core import atomic_json, digest
from .fly_market_activity import SCALARS, dynamic_state
from .fly_paper_memory import array_hash
from .fly_paper_protocol import validate_registration
from .fly_paper_study import imported_memory


def recipient_summary(baseline, state, post_ids, labels):
    baseline = np.asarray(baseline);posts = np.asarray(post_ids)
    if (baseline.ndim != 1 or not len(baseline) or posts.shape != baseline.shape
            or not np.isfinite(baseline).all() or np.any(baseline <= 0)
            or set(state) != {'weights', 'u', 'w'}
            or any(np.asarray(a).shape != baseline.shape or not np.isfinite(a).all() for a in state.values())):
        raise ValueError('Expected finite aligned native plastic memory and positive baseline weights')
    weights = state['weights'].astype(np.float64);base = baseline.astype(np.float64)
    delta = weights - base
    rows = []
    for identity in sorted(set(posts.tolist())):
        mask = posts == identity;changes = delta[mask];original = base[mask];current = weights[mask]
        if str(identity) not in labels:raise ValueError('Missing recipient annotation')
        rows.append({'post_id': str(identity), 'type': labels[str(identity)], 'edges': int(mask.sum()),
                     'changed': int(np.count_nonzero(changes)), 'increased': int(np.count_nonzero(changes > 0)),
                     'decreased': int(np.count_nonzero(changes < 0)),
                     'baseline_weight_sum': float(original.sum()), 'trained_weight_sum': float(current.sum()),
                     'net_weight_change_pct': float(100 * changes.sum() / original.sum()),
                     'weight_delta_l2': float(np.linalg.norm(changes)),
                     'efficacy_quantiles': np.quantile(current / original, [0, .25, .5, .75, 1]).tolist(),
                     'u_l2': float(np.linalg.norm(state['u'][mask])), 'w_l2': float(np.linalg.norm(state['w'][mask]))})
    return rows


def compare_memories(baseline, first, second):
    a = first['weights'].astype(np.float64) - baseline
    b = second['weights'].astype(np.float64) - baseline
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b));common = (a != 0) & (b != 0)
    return {'weight_delta_cosine': float(np.dot(a, b) / denominator) if denominator else None,
            'changed_in_both': int(common.sum()),
            'same_sign_fraction_when_both_changed': float(np.mean(np.sign(a[common]) == np.sign(b[common]))) if common.any() else None,
            'interpretation': 'Aligned stored weight differences across two pool histories; not a causal reward effect or evidence of generalization.'}


def create(registration, memory, data, output):
    from .fly_trace import TraceLab
    root = Path(output)
    if root.exists():raise ValueError('Refuse to overwrite a memory map')
    r = json.loads(Path(registration).read_text());a = json.loads((Path(memory)/'audit.json').read_text())
    validate_registration(r, a)
    lab = TraceLab(data);b = lab.brain
    clock = {k: getattr(b, k) for k in SCALARS}
    initial = {k: array_hash(v) for k, v in dynamic_state(b).items()}
    all_weight_hash = array_hash(b.weight)
    states = imported_memory(b, {'registration': r, 'training_audit': a}, memory)
    edges = b.circuit['edges'];pre = b.circuit['pre'];posts = b.post[edges]
    if not np.array_equal(pre, np.searchsorted(b.ptr, edges, side='right')-1):
        raise ValueError('Plastic source indices differ from retained CSR graph')
    labels = {str(b.ids[i]): str(lab.types[i]) for i in np.unique(posts)}
    pools = {}
    for key in r['cohort']:
        rows = recipient_summary(b.baseline_plastic, states[key], b.ids[posts], labels)
        if sum(row['changed'] for row in rows) != a['pools'][key]['changed_weights']:
            raise ValueError('Recipient counts differ from exported training audit')
        pools[key] = {'source': r['source_memories'][key], 'recipients': rows}
    after = {k: array_hash(v) for k, v in dynamic_state(b).items()}
    if initial != after or clock != {k: getattr(b, k) for k in SCALARS} or all_weight_hash != array_hash(b.weight) or any(clock.values()):
        raise ValueError('Read-only mapping advanced or modified the native model')
    root.mkdir(parents=True)
    mapping = root/'plastic-map.npz'
    np.savez_compressed(mapping, edge_ids=edges, pre_ids=b.ids[pre], post_ids=b.ids[posts], baseline=b.baseline_plastic)
    report = {'schema': 1, 'kind': 'verified_paper_memory_map',
              'registration_sha256': digest(registration), 'training_audit_sha256': digest(Path(memory)/'audit.json'),
              'source_sha256': digest(__file__), 'plastic_map_sha256': digest(mapping),
              'graph': a['graph'], 'pools': pools,
              'between_pool_memory': compare_memories(b.baseline_plastic, *[states[k] for k in r['cohort']]),
              'verification': {'imported_memories_match_registration': True, 'recipients_reconcile_to_graph': True,
                               'all_weights_and_dynamics_unchanged': True, 'neural_observations': 0},
              'interpretation': 'Descriptive map of stored KC-to-MBON weights after observed paper reward exposure. It does not isolate reward-driven changes from ongoing plasticity, measure firing or imply profitable learning.'}
    atomic_json(root/'report.json', report)
    return report


def render(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if report.get('kind') != 'verified_paper_memory_map' or report['verification']['neural_observations'] != 0:
        raise ValueError('Use a read-only verified memory map')
    pools = list(report['pools'].items())
    if len(pools) != 2:raise ValueError('Expected the two captured pools')
    recipients = [p['recipients'] for _, p in pools]
    if [x['post_id'] for x in recipients[0]] != [x['post_id'] for x in recipients[1]]:
        raise ValueError('Recipient identities differ between pools')
    bg, fg, muted = '#0e1728', '#e8edf6', '#b9c6da'
    bound = max(1, *(abs(row['net_weight_change_pct']) for rows in recipients for row in rows)) * 1.35
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 11, 'svg.fonttype': 'none',
                         'text.color': fg, 'axes.labelcolor': fg, 'xtick.color': muted, 'ytick.color': fg}):
        fig, axes = plt.subplots(1, 2, figsize=(13, 8), sharex=True, sharey=True, facecolor=bg)
        fig.subplots_adjust(left=.16, right=.97, top=.77, bottom=.22, wspace=.12)
        fig.text(.04, .94, 'Stored KC → MBON weights after paper training', fontsize=21, weight='bold')
        fig.text(.04, .903, f"All {report['graph']['plastic_edges']:,} plastic connections grouped by their {len(recipients[0])} native recipients. No propagation or new training.", fontsize=11)
        for i, (key, pool) in enumerate(pools):
            ax = axes[i];ax.set_facecolor('#172237');rows = pool['recipients'];x = pool['source']
            ax.set_title(f"Pool {i}: {x['observations']} observations\n{x['positive_rewards']} positive / {x['negative_rewards']} negative paper rewards", color=fg, fontsize=12, pad=18)
            values = [r['net_weight_change_pct'] for r in rows]
            ax.barh(range(len(rows)), values, height=.42, color=['#67e8cf' if v >= 0 else '#ff8e8e' for v in values])
            ax.axvline(0, color=muted, linewidth=1)
            ax.set_xlim(-bound, bound);ax.set_xlabel('Change in summed input weight (%)')
            ax.set_yticks(range(len(rows)), [f"{r['type']} · {r['post_id']}\n{r['edges']:,} connections" for r in rows], fontsize=10)
            for j, (row, value) in enumerate(zip(rows, values)):
                ax.annotate(f'{value:+.2f}%', (value, j), xytext=(5 if value >= 0 else -5, 0),
                            textcoords='offset points', va='center', ha='left' if value >= 0 else 'right', fontsize=10)
                ax.text(.03, j+.34, f"{row['changed']:,} changed · {row['increased']:,} increased · {row['decreased']:,} decreased",
                        transform=ax.get_yaxis_transform(), color=muted, fontsize=8.5, va='top')
            ax.set_ylim(len(rows)-.3, -.5)
            ax.grid(axis='x', alpha=.1)
            for spine in ax.spines.values():spine.set_color('#35445c')
        comparison = report['between_pool_memory']
        cosine='undefined (zero delta norm)' if comparison['weight_delta_cosine'] is None else f"{comparison['weight_delta_cosine']:.3f}"
        fig.text(.04, .14, f"Across pools: delta-vector cosine {cosine}; "
                 f"{comparison['changed_in_both']:,} connections changed in both.", fontsize=11)
        fig.text(.04, .102, 'Net sums can hide opposing changes; counts and the full report retain that distinction. Weights are model units.', fontsize=10, color=muted)
        fig.text(.04, .069, 'These snapshots do not isolate the effect of rewards, measure downstream firing, or establish trading profitability.', fontsize=10, color=muted)
        fig.text(.04, .035, 'Training audit SHA-256: '+report['training_audit_sha256'][:32]+'…', fontsize=9, color=muted)
        path = Path(output);path.parent.mkdir(parents=True, exist_ok=True)
        try:fig.savefig(path, dpi=140, facecolor=bg)
        finally:plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('registration', 'memory', 'fly-data', 'out'):parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--figures', action='store_true')
    args = parser.parse_args();report = create(args.registration, args.memory, args.fly_data, args.out)
    if args.figures:
        for suffix in ('svg', 'png'):render(report, args.out/('memory-map.'+suffix))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':main()
