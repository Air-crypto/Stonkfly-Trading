"""Audit original full-phase debugger views using retained study 12 recordings.

Repeat the independent ledger, image and native-bin audit, then verify topology
and every base chart against the complete graph and arrays. No neural propagation
or cloud submission occurs. Synthetic recordings remain labeled as validation.
"""
import argparse
import json
from pathlib import Path

from .core import atomic_json, digest
from .fly_rate_audit import audit_chunk
from .fly_rate_inputs import validate
from .fly_view_projection import audit_view, load_graph, require

ROOT = Path(__file__).resolve().parents[1]


def run(root, data, output, *, view_path=None):
    root, output = Path(root), Path(output)
    if output.exists():
        raise ValueError('Preserve earlier full-phase projection evidence')
    envelope = json.loads((root / 'protocol.json').read_text())
    summary = json.loads((root / 'summary.json').read_text())
    plan = validate(envelope)
    print('Reconstructing the recorded phase ledger and every learning bin.', flush=True)
    checked, reconstructed = audit_chunk(envelope, summary, root, data)
    count = checked['verification']['observations']
    require(type(count) is int and 0 <= count <= 24
            and checked['verification']['decision_slots'] == 25
            and checked['verification']['native_bins'] == 50 * count,
            'Full phase audit has inconsistent observation counts')
    if view_path is not None:
        target = json.loads(Path(view_path).read_text())
        # This entrypoint verifies original phase views, including reconstructed
        # credit and boundary panels. It does not accept a reselected subgraph.
        require(target == reconstructed, 'Supplied phase view differs from fresh independent reconstruction')
    else:
        target = reconstructed
    if count:
        require(target is not None, 'Missing view for recorded observations')
        print(f'Checking topology and plotted series across {count} observations.', flush=True)
        graph = load_graph(data)
        projection = audit_view(target, root / 'trace', graph, observations=count)
        graph_hashes = graph.source_sha256
    else:
        require(target is None, 'An unobserved phase cannot have a fabricated view')
        projection = {'observations': 0, 'bins': 0, 'view_status': 'no_neural_observations'}
        graph_hashes = None
    fixture = any('synthetic' in tick['source'] for series in plan['series'].values() for tick in series)
    report = {'status': 'rate_view_projection_audited', 'chunk': summary['chunk'],
        'plan_sha256': envelope['sha256'], 'summary_sha256': digest(root / 'summary.json'),
        'validation_only': fixture, 'projection': projection, 'graph_data_sha256': graph_hashes,
        'source_sha256': {name: digest(ROOT / name) for name in (
            'paperlab/fly_rate_projection.py', 'paperlab/fly_view_projection.py',
            'paperlab/fly_rate_audit.py')},
        'artifact_sha256': checked['artifact_sha256'],
        'supplied_view_sha256': digest(view_path) if view_path is not None else None,
        'neural_observations_computed': 0, 'cloud_submissions': 0,
        'interpretation': 'Original phase visualization checked against independently audited full recordings. '
            'Missing market slots stay missing; observation count does not include the terminal account mark. '
            'This repeats offline reconstruction only and does not create trajectories, training or a market result. '
            + ('Prices and inputs are synthetic validation data.' if fixture else
               'The existing paper results require their full study context; this check does not select a model.')}
    output.mkdir(parents=True)
    atomic_json(output / 'audit.json', checked)
    report['native_audit_sha256'] = digest(output / 'audit.json')
    if target is not None:
        atomic_json(output / 'view.json', target)
        report['audited_view_sha256'] = digest(output / 'view.json')
    else:
        report['audited_view_sha256'] = None
    atomic_json(output / 'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'fly-data', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--view', type=Path,
                        help='Optional original expanded phase view to compare with a fresh audit')
    args = parser.parse_args()
    report = run(args.root, args.fly_data, args.out, view_path=args.view)
    print(json.dumps({key: report[key] for key in ('status', 'chunk', 'validation_only',
        'projection', 'neural_observations_computed', 'cloud_submissions')}, indent=2))


if __name__ == '__main__':
    main()
