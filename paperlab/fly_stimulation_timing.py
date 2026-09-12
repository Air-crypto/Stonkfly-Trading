"""Locate count differences in audited native bins; never execute a neural model."""
import argparse
import json
from pathlib import Path
from urllib.parse import urlencode

import numpy as np

from .core import atomic_json, digest

TARGETS = ('10704', '11402')
OUTPUTS = ('10162', '10059', '10527', '555871')


def interval(index):
    return None if index is None else {'bin_index': int(index), 'start_ms': int(index)*10,
                                      'end_ms': (int(index)+1)*10}


def first_bin(mask):
    indices = np.flatnonzero(mask)
    return int(indices[0]) if len(indices) else None


def count_difference(pristine, trained):
    """Keep timing differences even when per-neuron observation totals match."""
    a, b = np.asarray(pristine), np.asarray(trained)
    if (a.shape != b.shape or a.ndim != 2 or a.shape[0] != 50
            or a.dtype.kind not in 'iu' or b.dtype.kind not in 'iu'
            or np.any(a < 0) or np.any(b < 0)):
        raise ValueError('Expected aligned nonnegative native count bins')
    changed = np.count_nonzero(a != b, axis=1)
    return {'first_difference': interval(first_bin(changed > 0)),
            'changed_neurons_per_bin': changed.tolist(),
            'observation_totals_equal': bool(np.array_equal(a.sum(axis=0), b.sum(axis=0)))}


def analyze(audit_file, artifacts):
    audit_file, root = Path(audit_file), Path(artifacts)
    audit = json.loads(audit_file.read_text()); p = audit['protocol']
    if (audit.get('status') != 'verified' or audit.get('full_observations_verified') != 36
            or audit.get('native_bins_verified') != 1800 or p['target_ids'] != list(TARGETS)
            or p['currents'] != [0, 5, 10] or len(p['arms']) != 12):
        raise ValueError('Requires the complete recipient stimulation audit')
    hashes = {}; rows = []

    def load(name, observation):
        relative = f'{name}/step-{observation:02}.npz'; path = root/relative
        sha = digest(path)
        if sha != audit['artifact_sha256'][relative]:
            raise ValueError('Native recording differs from the audited artifact: '+relative)
        hashes[relative] = sha
        with np.load(path, allow_pickle=False) as raw:
            ids = raw['neuron_ids']; counts = raw['counts']; times = raw['ms']
            if (ids.shape != (166700,) or len(np.unique(ids)) != 166700
                    or counts.shape != (50, 166700)
                    or not np.array_equal(times, np.arange(10, 501, 10)+(observation-1)*500)):
                raise ValueError('Incomplete native graph or mismatched observation clock')
            lookup = {str(identity): i for i, identity in enumerate(ids)}
            voltage = raw['voltage'][:, [lookup[x] for x in TARGETS]]
        return ids, counts, voltage, lookup

    for pool in (0, 1):
        for current in p['currents']:
            pristine_name = f'pool{pool}-pristine-current{current}'
            trained_name = f'pool{pool}-trained-current{current}'
            for observation in (1, 2, 3):
                ids, a, av, lookup = load(pristine_name, observation)
                other_ids, b, bv, _ = load(trained_name, observation)
                if not np.array_equal(ids, other_ids):raise ValueError('Neuron identity order differs')
                full = count_difference(a, b); first = full['first_difference']
                if first is not None:
                    changed = np.flatnonzero(a[first['bin_index']] != b[first['bin_index']])
                    first.update(changed_neurons=len(changed), first_20_changed_ids=[str(x) for x in ids[changed[:20]]])
                recipients = {}
                for k, identity in enumerate(TARGETS):
                    ia, ib = a[:, lookup[identity]], b[:, lookup[identity]]
                    difference = interval(first_bin(ia != ib))
                    recipients[identity] = {
                        'pristine_spikes': int(ia.sum()), 'trained_spikes': int(ib.sum()),
                        'pristine_first_spike_bin': interval(first_bin(ia > 0)),
                        'trained_first_spike_bin': interval(first_bin(ib > 0)),
                        'first_count_difference': difference,
                        'pristine_counts': ia.tolist(), 'trained_counts': ib.tolist(),
                        'pristine_voltage_bin_end_mv': av[:, k].tolist(),
                        'trained_voltage_bin_end_mv': bv[:, k].tolist(),
                        'viewer_query': '?' + urlencode({'run': 'stimulation01-'+trained_name,
                            'compare': 'stimulation01-'+pristine_name, 'step': observation-1,
                            'bin': difference['bin_index'] if difference else 0, 'neuron': identity})}
                output_indices = [lookup[x] for x in OUTPUTS]
                row_a, row_b = (audit['rows'][name][observation-1] for name in (pristine_name, trained_name))
                if row_a['input_sha256'] != row_b['input_sha256']:raise ValueError('Market images differ')
                rows.append({'pool_index': pool, 'current': current, 'observation': observation,
                    'pristine': pristine_name, 'trained': trained_name,
                    'decision_ts': row_a['decision_ts'], 'input_sha256': row_a['input_sha256'],
                    'full_graph': full, 'recipients': recipients,
                    'output_neurons': count_difference(a[:, output_indices], b[:, output_indices]),
                    'final_sides': {'pristine': row_a['side'], 'trained': row_b['side']}})
    return {'kind': 'audited_stimulation_bin_timing', 'audit_sha256': digest(audit_file),
            'source_sha256': digest(__file__), 'artifact_sha256': hashes, 'rows': rows,
            'observations_checked': len(hashes), 'native_bins_checked': len(hashes)*50,
            'interpretation': 'All intervals are relative to the start of one 500 ms observation. '
                'Native counts resolve 10 ms bins, not exact spike times or within-bin causal order. '
                'Voltages are sampled at bin ends and can miss threshold crossings and resets. '
                'Output-neuron count differences are intermediate activity, not new trading decisions; '
                'the fixed decoder emits one action after 500 ms. Later observations carry activity from '
                'earlier ones. This post-hoc analysis adds no model runs, trading returns or policy selection.'}


def markdown(report):
    def stamp(value):return 'none' if value is None else f"{value['start_ms']}–{value['end_ms']} ms"
    lines = ['# Native count differences in the recipient diagnostic', '', report['interpretation'], '',
             '| Pool | Current | Image | First full-graph count difference | First output-neuron count difference | Final actions, pristine / trained | Recipient traces |',
             '|---|---:|---:|---|---|---|---|']
    for row in report['rows']:
        links = ', '.join(f'[{identity}](http://127.0.0.1:8766/{data["viewer_query"]})'
                          for identity, data in row['recipients'].items())
        lines.append(f"| {('ALL', 'baton')[row['pool_index']]} | {row['current']} | {row['observation']} | "
                     f"{stamp(row['full_graph']['first_difference'])} | {stamp(row['output_neurons']['first_difference'])} | "
                     f"{row['final_sides']['pristine']} / {row['final_sides']['trained']} | {links} |")
    return '\n'.join(lines)+'\n'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('audit', 'artifacts', 'out'):p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():raise ValueError('Refuse to overwrite a timing analysis')
    report = analyze(a.audit, a.artifacts);a.out.mkdir(parents=True)
    atomic_json(a.out/'timing.json', report);(a.out/'timing.md').write_text(markdown(report))
    print(json.dumps({k: report[k] for k in ('observations_checked', 'native_bins_checked')}))


if __name__ == '__main__':main()
