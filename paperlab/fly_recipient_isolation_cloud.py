"""Prepare or resume one isolation assay after the pending paper comparison ends."""
import argparse
import json
from pathlib import Path

from .core import atomic_json
from .fly_recipient_isolation import protocol, validate


def prior_comparison_complete():
    """A receipt alone is insufficient: the exact owning Modal call must finish."""
    import modal
    volume = modal.Volume.from_name('fly-paper-lab-state', environment_name='main')
    receipt = json.loads(b''.join(volume.read_file('/registered-paper-10/cloud-call.json')))
    if receipt.get('status') != 'completed' or not receipt.get('call_id'):
        raise RuntimeError('Wait for study 10; no isolation compute submitted')
    try:result = modal.FunctionCall.from_id(receipt['call_id']).get(timeout=0)
    except TimeoutError as exc:raise RuntimeError('Exact study 10 call is still pending; no isolation compute submitted') from exc
    if result.get('status') != 'paper_checkpoint_study_completed' or result.get('run_id') != receipt['run_id']:
        raise RuntimeError('Study 10 terminal result differs; no isolation compute submitted')


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    pack = sub.add_parser('pack')
    for field in ('paper-payload', 'reference', 'audit', 'receipt', 'out'):
        pack.add_argument('--'+field, type=Path, required=True)
    cloud = sub.add_parser('cloud')
    for field in ('payload', 'reference-artifacts', 'out'):cloud.add_argument('--'+field, type=Path, required=True)
    a = p.parse_args()
    if a.command == 'pack':
        if a.out.exists():raise ValueError('Refuse to overwrite isolation payload')
        parent = json.loads(a.paper_payload.read_text()); raw = a.reference.read_text(); audit = a.audit.read_text()
        receipt = json.loads(a.receipt.read_text())
        if receipt.get('status') != 'completed' or not receipt.get('call_id'):raise ValueError('Parent assay must be completed and audited')
        payload = {'protocol':protocol(parent, raw, audit, receipt['run_id']), 'paper_payload':parent,
                   'reference_json':raw, 'audit_json':audit}
        validate(payload); atomic_json(a.out, payload)
    else:
        from .fly_paper_stimulation_cloud import cloud_run
        cloud_run(json.loads(a.payload.read_text()), a.out, isolation=True, reference_artifacts=a.reference_artifacts)


if __name__ == '__main__':main()
