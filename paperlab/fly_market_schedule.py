"""Once-only cloud execution of the already registered market study 08."""
import json
from pathlib import Path
import shutil
import sqlite3
import time

from .core import atomic_json, digest
from .fly_market_input import seal_registered, source_files
from .fly_market_study import signature

DIRECTORY = 'registered-market-08'
REGISTRATION = 'fly-market-study-08-preregistration.json'
PARENT = 'fly-market-study-07-plan.json'


def execute_due(state, discovery, specifications, *, call_id, input_id, commit,
                run, settle_budget, now=None):
    """Called inside the existing worker lease and its 600-second reservation.

    A persisted claim prevents every automatic retry, including hard termination
    during sealing, execution or result commit. Later ticks resume ordinary paper
    collection/trading. No new Modal call, schedule or trading policy is created.
    """
    state, discovery, specifications = map(Path, (state, discovery, specifications))
    root = state / DIRECTORY
    receipt_path = root / 'cloud-call.json'
    if receipt_path.exists():
        return None
    registration_path = specifications / REGISTRATION
    registration = json.loads(registration_path.read_text())
    now = time.time() if now is None else now
    if now < registration['end']:
        return None
    archive = discovery / 'universe-snapshot.db'
    if not archive.exists():
        return None
    db = sqlite3.connect(f'file:{archive.resolve()}?mode=ro', uri=True)
    try:
        last = db.execute("SELECT max(json_extract(payload,'$.observed')) FROM observations").fetchone()[0]
    finally:
        db.close()
    if last is None or last < registration['end']:
        return None
    if not call_id or not input_id:
        raise ValueError('Scheduled study requires the owning Modal call and input IDs')
    root.mkdir(parents=True, exist_ok=True)
    receipt = {'status': 'claimed', 'call_id': call_id, 'input_id': input_id,
               'run_id': 'assay-market-08-' + digest(registration_path)[:16],
               'registration_sha256': digest(registration_path),
               'parent_file_sha256': digest(specifications / PARENT),
               'claimed_at': now, 'dispatch': 'scheduled-worker-once'}
    atomic_json(receipt_path, receipt)
    commit()  # Claim must be durable before sealing or invoking the native brain.
    try:
        shutil.copyfile(archive, root / 'universe.db')
        shutil.copyfile(registration_path, root / 'preregistration.json')
        envelope = seal_registered(root / 'universe.db', specifications / PARENT,
                                   root / 'preregistration.json', root / 'plan.json')
        atomic_json(root / 'source-hashes.json', {
            name: digest(Path(__file__).with_name(name)) for name in source_files(6)})
        receipt.update(status='pending', plan_sha256=envelope['sha256'],
                       request_sha256=signature({'plan': envelope, 'registration': registration}))
        atomic_json(receipt_path, receipt)
        commit()
        print(json.dumps({'event': 'registered_market_started', **receipt}), flush=True)
        result = run({'run_id': receipt['run_id'], 'market_plan': envelope},
                     str(state / 'fly-debugger'), str(state / 'fly-data'))
        if result.get('status') != 'market_study_completed' or result.get('run_id') != receipt['run_id']:
            raise ValueError('Unexpected registered market result')
        result['budget'] = settle_budget()
        atomic_json(state / 'fly-debugger' / receipt['run_id'] / 'cloud-result.json', result)
        atomic_json(root / 'cloud-result.json', result)
        receipt.update(status='completed', budget=result['budget'])
        atomic_json(receipt_path, receipt)
        commit()
        print(json.dumps({'event': 'registered_market_completed', 'run_id': receipt['run_id'],
                          'budget': result['budget']}), flush=True)
        return result
    except Exception as exc:
        receipt.update(status='failed', error_type=type(exc).__name__, error=str(exc)[:300])
        atomic_json(receipt_path, receipt)
        commit()  # Keep the worst-case reservation after failure.
        raise
