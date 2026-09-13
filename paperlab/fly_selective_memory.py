"""Resolve recorded weight movement into stored memory and trace-origin drive.

This applies the linear memory filter to saved rates. No native propagation or
counterfactual trading is performed. Saturated memory cannot use this analysis.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from .core import atomic_json, digest
from .fly_selective_trace_figure import load_audit, MODES, LABELS, GROUPS, GROUP_LABELS

ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = ('stored_memory', 'earlier_trace_drive', 'current_trace_drive', 'rounding_residual')
INTERPRETATION = ('Algebraic contributions on each condition\'s recorded firing trajectory. '
    'These are not neural interventions, causal credit assignments, gradients or trading returns. '
    'A signed projection measures alignment with the observed weight-change vector; '
    'individual component norms do not add to the total norm.')


def response(t, tu=1800., tw=.05):
    """Exact u-to-w and constant-drive-to-w responses from zero state."""
    if not all(math.isfinite(x) for x in (t, tu, tw)) or t < 0 or not 0 < tw < tu:
        raise ValueError('Require finite nonnegative time and 0 < filter < decay')
    c = tu/(tu-tw)*math.exp(-t/tu)*(-math.expm1(-t*(1/tw-1/tu)))
    return c, tu*(-math.expm1(-t/tw)-c)


def decompose(u0, w0, old_drive, current_drive, dt=.01, tu=1800., tw=.05, *, frozen=False):
    """Convolve piecewise-constant drives; independent of the stepwise rule."""
    u0, w0, old_drive, current_drive = map(lambda v: np.asarray(v, dtype=np.float64),
                                        (u0, w0, old_drive, current_drive))
    if (u0.ndim != 1 or w0.shape != u0.shape or old_drive.ndim != 2
            or old_drive.shape != current_drive.shape or old_drive.shape[1] != len(u0)
            or not 0 < dt <= .0100001
            or not all(np.isfinite(v).all() for v in (u0, w0, old_drive, current_drive))):
        raise ValueError('Finite aligned memory and drive arrays are required')
    n = len(old_drive)
    if frozen:
        return {k: np.zeros((n, len(u0))) for k in COMPONENTS[:3]}
    times = np.arange(1, n+1)*dt
    coefficients = np.array([response(t, tu, tw)[1]-response(max(0., t-dt), tu, tw)[1]
                             for t in times])
    # Row j weighs every earlier input bin by elapsed time since that input.
    kernel = np.zeros((n, n))
    for j in range(n): kernel[j, :j+1] = coefficients[j::-1]
    return {'stored_memory': np.array([w0*math.expm1(-t/tw)+u0*response(t, tu, tw)[0] for t in times]),
            'earlier_trace_drive': kernel@old_drive,
            'current_trace_drive': kernel@current_drive}


def describe(vectors, actual):
    norm = float(np.linalg.norm(actual))
    return {'observed_change_l2': norm, 'components': {
        key: {'l2': float(np.linalg.norm(value)),
              'signed_projection': float(np.dot(value, actual)/norm) if norm else 0.}
        for key, value in vectors.items()}}


def validate_report(report):
    expected = {g+'_'+m for g in GROUPS for m in MODES}
    v = report['verification']
    if (report['status'] != 'selective_memory_components_reconstructed'
            or set(report['arms']) != expected or v['observations'] != 36
            or v['bins'] != 1800 or v['edges_per_bin'] != 7835
            or any(v.get(k) is not True for k in ('all_recorded_drive_integrals_match_audit',
                'all_recorded_memory_filters_reconstructed', 'all_weight_residuals_within_endpoint_precision',
                'all_final_projection_sums_verified', 'no_saturated_memory_used'))):
        raise ValueError('Require the complete verified component reconstruction')
    for rows in report['arms'].values():
        if len(rows) != 3 or [r['observation'] for r in rows] != [1, 2, 3]:
            raise ValueError('All three images must be represented')
        for row in rows:
            if len(row['series']) != 50 or row['final'] != row['series'][-1]:
                raise ValueError('Missing bins or inconsistent final projection')
            for point in row['series']:
                norm = point['observed_change_l2']; parts = point['components']
                if set(parts) != set(COMPONENTS) or not math.isfinite(norm) or norm < 0:
                    raise ValueError('Invalid component schema or norm')
                for p in parts.values():
                    if (not math.isfinite(p['l2']) or not math.isfinite(p['signed_projection'])
                            or p['l2'] < 0 or abs(p['signed_projection']) > p['l2']+1e-10):
                        raise ValueError('Projection exceeds its component norm')
                if not math.isclose(sum(p['signed_projection'] for p in parts.values()), norm,
                                    rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError('Signed projections fail to sum to observed norm')


def analyze(audit_path, artifacts):
    audit_path, artifacts = Path(audit_path), Path(artifacts)
    a = load_audit(audit_path)
    if digest(audit_path) != digest(ROOT/'reports/fly-selective-audit-01.json'):
        raise ValueError('Use the published complete selective audit')
    rule_path = ROOT/'vendor/stonkfly/stonkfly/neural/rule.py'
    if digest(rule_path) != a['code_sha256'][str(rule_path.relative_to(ROOT))]:
        raise ValueError('Recorded rule source changed')
    used = {}
    def arrays(name, fields=None):
        path = artifacts/name; sha = digest(path)
        if sha != a['artifact_sha256'][name]: raise ValueError('Recorded artifact changed: '+name)
        used[name] = sha
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in (fields if fields is not None else z.files)}
    c = arrays('circuit.npz')
    if len(c['edges']) != 7835: raise ValueError('Changed learning circuit')
    baseline = c['baseline'].astype(np.float64)
    result = {}; all_final = {}; maximum_error = 0.; maximum_rounding_fraction = 0.
    for name, arm in a['protocol']['arms'].items():
        rows = []
        for i, observed in enumerate(a['arms'][name], 1):
            if observed['rule']['saturated_edge_bins']:
                raise ValueError('Clipping invalidates the linear decomposition')
            z = arrays(f'{name}/step-{i:02}.npz', ('ms', 'counts', 'weights', 'kc', 'dan', 'u', 'w', 'initial_weights'))
            b = arrays(f'{name}/boundary-{i:02}.npz', ('memory_after__u', 'memory_after__w', 'after__rate_kc', 'after__rate_dan'))
            u0, w0, k0, d0 = (b[k] for k in ('memory_after__u', 'memory_after__w', 'after__rate_kc', 'after__rate_dan'))
            old, new = [], []
            for j, counts in enumerate(z['counts']):
                K, D = counts[c['pre']]/.01, counts[c['dan']]/.01
                k, d = (k0, d0) if j == 0 else (z['kc'][j-1], z['dan'][j-1])
                mid_decay = math.exp(-.005)
                kmid, dmid = k*mid_decay+K*(1-mid_decay), d*mid_decay+D*(1-mid_decay)
                total = .001*(K*(c['gain'].T@dmid)-(c['gain'].T@D)*kmid)
                old_decay = math.exp(-(.01*j+.005))
                earlier = .001*(K*(c['gain'].T@(d0*old_decay))-(c['gain'].T@D)*(k0*old_decay))
                current = total-earlier
                if not arm['learning']: earlier[:], current[:] = 0., 0.
                old.append(earlier); new.append(current)
            old, new = np.asarray(old), np.asarray(new)
            for key, values in (('earlier', old), ('current', new), ('total', old+new)):
                integrated = float(.01*np.abs(values).sum())
                if not np.isclose(integrated, observed['rule']['integrated_absolute_drive'][key], rtol=1e-11, atol=1e-12):
                    raise ValueError('Drive reconstruction disagrees with independent audit')
            parts = decompose(u0, w0, old, new, frozen=not arm['learning'])
            predicted = w0+sum(parts.values())
            error = float(np.max(np.abs(predicted-z['w'])))
            # A direct convolution subtracts two step responses; its scalar
            # roundoff differs from the previously audited per-bin recurrence.
            if not np.allclose(predicted, z['w'], rtol=1e-9, atol=2e-11):
                raise ValueError('Closed-form memory filter fails to reproduce recording')
            maximum_error = max(maximum_error, error)
            actual = z['weights'].astype(np.float64)-z['initial_weights'].astype(np.float64)
            vectors = {key: value*baseline for key, value in parts.items()}
            residual = actual-sum(vectors.values())
            # Each endpoint is stored as float32. Allow one spacing at each
            # endpoint, plus the measured convolution error in weight units.
            bound = (np.abs(np.spacing(z['weights']).astype(np.float64))
                     + np.abs(np.spacing(z['initial_weights']).astype(np.float64))
                     + np.abs(baseline)*error + 1e-12)
            if np.any(np.abs(residual) > bound):
                raise ValueError('Weight residual exceeds recorded endpoint precision')
            vectors['rounding_residual'] = residual
            series = [describe({k: v[j] for k, v in vectors.items()}, actual[j]) for j in range(50)]
            last = series[-1]
            fraction = last['components']['rounding_residual']['l2']/last['observed_change_l2'] if last['observed_change_l2'] else 0.
            maximum_rounding_fraction = max(maximum_rounding_fraction, fraction)
            if not np.isclose(sum(v['signed_projection'] for v in last['components'].values()),
                              last['observed_change_l2'], rtol=1e-12, atol=1e-12):
                raise ValueError('Signed component projections do not sum to observed norm')
            rows.append({'observation': i, 'side': observed['side'], 'gate_spikes': observed['gate_spikes'],
                         'times_ms': z['ms'].tolist(), 'maximum_memory_error': error,
                         'final': last, 'series': series})
            all_final[name+f'__image_{i}__observed'] = actual[-1]
            for key, values in vectors.items(): all_final[name+f'__image_{i}__'+key] = values[-1]
        result[name] = rows
        print('memory components verified: '+name, flush=True)
    return {'status': 'selective_memory_components_reconstructed', 'audit_sha256': digest(audit_path),
            'source_sha256': digest(__file__), 'rule_sha256': digest(rule_path),
            'fixed_parameters': {'eta': .001, 'trace_seconds': 1., 'memory_decay_seconds': 1800., 'filter_seconds': .05},
            'arms': result, 'artifact_sha256': used,
            'verification': {'observations': 36, 'bins': 1800, 'edges_per_bin': 7835,
                             'maximum_w_error': maximum_error, 'maximum_endpoint_residual_fraction': maximum_rounding_fraction,
                             'all_recorded_drive_integrals_match_audit': True,
                             'all_recorded_memory_filters_reconstructed': True,
                             'all_weight_residuals_within_endpoint_precision': True,
                             'all_final_projection_sums_verified': True, 'no_saturated_memory_used': True},
            'model_submissions': 0, 'neural_observations': 0, 'interpretation': INTERPRETATION}, all_final


def figure(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    validate_report(report)
    bg, fg, muted = '#0e1728', '#e8edf6', '#b9c6da'
    colors = ('#ffbf69', '#e6a8e8', '#8fbcff', '#65758e')
    labels = ('Stored u/w relaxation', 'Earlier-image trace drive', 'Current-image trace drive', 'Endpoint residual')
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10, 'text.color': fg,
                        'axes.labelcolor': fg, 'axes.titlecolor': fg, 'xtick.color': muted,
                        'ytick.color': muted, 'axes.edgecolor': '#52617a', 'svg.fonttype': 'none',
                        'text.parse_math': False}):
        fig, axes = plt.subplots(1, 3, figsize=(15, 7.8), facecolor=bg)
        fig.subplots_adjust(left=.105, right=.975, top=.72, bottom=.28, wspace=.34)
        fig.text(.035, .955, 'What drives the larger third-image weight updates?', fontsize=22, weight='bold')
        fig.text(.035, .90, 'Memory filtering reconstructed from all saved bins. Each bar separates components along the observed update direction.', color=muted)
        for ax, group, title in zip(axes, GROUPS, GROUP_LABELS):
            ax.set_facecolor('#172237'); positive=np.zeros(4); negative=np.zeros(4)
            for component, color, label in zip(COMPONENTS, colors, labels):
                values=np.array([report['arms'][group+'_'+m][2]['final']['components'][component]['signed_projection'] for m in MODES])
                ax.barh(np.arange(4), values, left=np.where(values>=0,positive,negative), color=color, label=label)
                positive += np.maximum(values,0); negative += np.minimum(values,0)
            totals=[report['arms'][group+'_'+m][2]['final']['observed_change_l2'] for m in MODES]
            ax.scatter(totals,np.arange(4),color='#67e8cf',marker='D',s=40,label='Observed weight-change L2',zorder=4)
            for y, total in enumerate(totals): ax.text(max(total,positive[y])+.4,y,f'{total:.2f}',va='center',color=fg,fontsize=9)
            ax.set_yticks(np.arange(4),LABELS);ax.invert_yaxis();ax.set_xlim(-3,25)
            ax.axvline(0,color=muted,lw=.8);ax.grid(axis='x',alpha=.12)
            ax.set_title(title,loc='left',fontsize=11,pad=15)
            ax.set_xlabel('Signed contribution to the observed\nweight-change direction')
        handles, legend_labels = axes[0].get_legend_handles_labels()
        fig.legend(handles,legend_labels,loc='upper left',bbox_to_anchor=(.035,.85),ncol=3,frameon=False,labelcolor=fg,fontsize=10)
        fig.text(.035,.185,'Positive bars align with the recorded update; negative bars oppose it. Component norms are not additive.',color=muted)
        fig.text(.035,.14,'Each condition uses its own recorded firing. This algebra does not predict the effect of removing a component in a running brain.',color=muted)
        fig.text(.035,.095,'All 36 observations / 1,800 bins checked; this figure shows image 3. No clipping occurred in these recordings.',color=muted)
        fig.text(.035,.05,'No model runs, new training, trading returns or policy promotion. The prospective paper comparison remains unsuccessful.',color=muted)
        for ext in ('.png','.svg'):
            path=Path(output).with_suffix(ext);fig.savefig(path,dpi=150,facecolor=bg)
            if ext=='.svg':path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
        plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path);p.add_argument('--artifacts',type=Path)
    p.add_argument('--report',type=Path,help='Reproduce the figure from an existing complete analysis, without arrays')
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    if args.out.exists():raise ValueError('Preserve prior analyses; choose a new output directory')
    if args.report:
        if args.audit or args.artifacts:raise ValueError('Use either a report or audit plus artifacts')
        report=json.loads(args.report.read_text());validate_report(report);args.out.mkdir(parents=True)
        figure(report,args.out/'components')
        atomic_json(args.out/'figure.json',{'report_sha256':digest(args.report),'source_sha256':digest(__file__),
            'figure_sha256':{e:digest((args.out/'components').with_suffix(e)) for e in ('.png','.svg')},
            'model_submissions':0})
        return
    if not args.audit or not args.artifacts:raise ValueError('Supply both --audit and --artifacts')
    report, vectors=analyze(args.audit,args.artifacts)
    args.out.mkdir(parents=True)
    np.savez_compressed(args.out/'final-vectors.npz',**vectors)
    report['final_vectors_sha256']=digest(args.out/'final-vectors.npz')
    figure(report,args.out/'components')
    report['figure_sha256']={e:digest((args.out/'components').with_suffix(e)) for e in ('.png','.svg')}
    atomic_json(args.out/'report.json',report)
    print(json.dumps(report['verification'],indent=2))


if __name__=='__main__':main()
