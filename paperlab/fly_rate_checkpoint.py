"""Freeze current paper memories for the next rate comparison, without trading.

The cohort is selected from pre-evaluation availability and volume, never P&L.
The actual native reference is constructed only by the bounded cloud worker;
it exports arrays and does not propagate the brain or restore a checkpoint.
"""
import json
from pathlib import Path
import re
import shutil
import sqlite3
import time

from .budget import reserve, settle
from .core import atomic_json, digest
from .fly_market_study import quote_at, signature
from .fly_paper_memory import exposure, export
from .multi import MIN_CONTEXT, read_archive

DIRECTORY = 'rate-checkpoint-12'
APP = 'fly-paper-rate-checkpoint-12'
POLICY = ('Choose two distinct contracts from the currently assigned Solana fly paper sleeves. '
    'Require a current eligible quote, at least 64 historical eligible observations, at least 18 '
    'eligible observations in the last 24 completed five-minute slots, a versioned checkpoint, '
    'and complete recorded positive and negative reward exposure. Rank eligible candidates by '
    'descending current five-minute volume, then pool key. Do not use return, equity, fees or '
    'reward magnitude for ranking. Freeze the cohort, ledger and checkpoint bytes before any '
    'new evaluation registration. This is a current Solana pool sample, not an exhaustive or '
    'verified memecoin classification. Preserve missing pools in later evaluation.')


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    from .fly_online_study import SOURCE_FILES
    names = {'paperlab/'+n for n in SOURCE_FILES}
    names.update(('paperlab/fly_rate_checkpoint.py', 'paperlab/fly_rate_checkpoint_cloud.py', 'rate_checkpoint_cloud.py',
                  'cloud.py', 'paperlab/budget.py', 'paperlab/universe.py',
                  'paperlab/fly_recording_download.py'))
    names.update(str(p.relative_to(root)) for p in (root/'vendor/stonkfly').rglob('*')
                 if p.suffix in ('.py', '.cpp') and '__pycache__' not in p.parts)
    return {name: digest(root/name) for name in sorted(names)}


def select_cohort(ledger, archive, now):
    """Pure data selection: all candidates and rejection reasons are retained."""
    db = sqlite3.connect(f'file:{Path(ledger).resolve()}?mode=ro', uri=True)
    latest, series = read_archive(archive, now)
    decisions = [int(now//300)*300-300*i for i in range(24, 0, -1)]
    candidates = []; eligible = []
    try:
        if db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
            raise ValueError('Invalid paper ledger')
        saved = db.execute("SELECT payload FROM state WHERE name='fly'").fetchone()
        if saved is None: raise ValueError('No fly paper sleeves')
        lanes = json.loads(saved[0])
        for sleeve, lane in enumerate(lanes):
            key = lane.get('pool'); pool = latest.get(key); seq = series.get(key, [])
            row = {'sleeve': sleeve, 'pool': key, 'symbol': pool.symbol if pool else None,
                   'eligible_slots': 0, 'context': sum(t.available for t in seq), 'reasons': []}
            candidates.append(row)
            if pool is None:
                row['reasons'].append('missing_pool'); continue
            row.update(network=pool.network, token=pool.token, quote_ts=pool.observed,
                       volume5=pool.volume5)
            if pool.network != 'solana': row['reasons'].append('not_solana')
            if pool.rejection(now): row['reasons'].append(pool.rejection(now))
            if row['context'] < MIN_CONTEXT: row['reasons'].append('short_context')
            last = 0; coverage = []
            for stamp in decisions:
                t = quote_at(seq, stamp)[1] if seq and seq[0].ts <= stamp else None
                ok = t is not None and t.available and t.ts > last
                if ok: last = t.ts
                coverage.append({'decision_ts': stamp, 'quote_ts': t.ts if t else None,
                                 'eligible': bool(ok)})
            row['coverage'] = coverage
            row['eligible_slots'] = sum(c['eligible'] for c in coverage)
            if row['eligible_slots'] < 18: row['reasons'].append('sparse_recent_coverage')
            checkpoint = lane.get('checkpoint')
            if not isinstance(checkpoint, str) or not re.fullmatch(r'fly-\d+-\d+\.npz', Path(checkpoint).name):
                row['reasons'].append('missing_versioned_checkpoint')
            try:
                events = exposure(db, key, sleeve, lane['assigned'])
            except (ValueError, KeyError) as exc:
                row['reasons'].append('unverified_exposure: '+str(exc)); continue
            row.update(observations=len(events), positive_rewards=sum(e['reward'] > .01 for e in events),
                       negative_rewards=sum(e['reward'] < -.01 for e in events))
            if not row['reasons']:
                if Path(checkpoint).name != f"fly-{sleeve}-{events[-1]['slot']}.npz":
                    raise ValueError('Assigned checkpoint differs from the final exposure slot')
                eligible.append((pool, sleeve, lane, events))
    finally:
        db.close()
    eligible.sort(key=lambda x: (-x[0].volume5, x[0].key))
    selected = []; tokens = set()
    for pool, sleeve, lane, events in eligible:
        if pool.token in tokens: continue
        tokens.add(pool.token)
        selected.append({'key': pool.key, 'sleeve': sleeve, 'assigned': lane['assigned'],
                         'remote_checkpoint': lane['checkpoint'], 'events': events,
                         'observations': len(events)})
        if len(selected) == 2: break
    if len(selected) != 2: raise ValueError('Fewer than two eligible checkpoint pools; do not replace with unqualified pools')
    return {'policy': POLICY, 'selected_at': now, 'candidates': candidates,
            'selected': selected, 'cohort': [x['key'] for x in selected]}


def native_reference(data):
    from .fly_trace import TraceLab
    return TraceLab(data).brain


def execute(request, state, discovery, *, call_id, input_id, commit,
            build_reference=native_reference, clock=time.time, sleep=time.sleep):
    """Called within the shared worker lease. One durable attempt, never resumed."""
    if (set(request) != {'policy', 'source_sha256'} or request['policy'] != POLICY
            or request['source_sha256'] != source_hashes()):
        raise ValueError('Prepared checkpoint policy or sources differ')
    if not call_id or not input_id: raise ValueError('Cloud ownership is required')
    state, discovery = Path(state), Path(discovery); root = state/DIRECTORY
    if root.exists(): raise ValueError('Checkpoint capture already claimed; inspect its original call')
    budget = reserve(state/'budget.json', True, seconds=1800, startup_seconds=30,
                     memory_gib=8, limit_override=25)
    if budget is None: return {'status': 'budget_stopped'}
    budget['rate'] *= 3; budget['startup_seconds'] = 10
    root.mkdir()
    claim = {'status': 'claimed', 'call_id': call_id, 'input_id': input_id,
             'claimed_at': clock(), 'request_sha256': signature(request),
             'reserved_usd': budget['reserve'], 'source_sha256': request['source_sha256']}
    atomic_json(root/'claim.json', claim); atomic_json(root/'request.json', request); commit()
    started = time.monotonic()
    try:
        capture = root/'capture'; capture.mkdir()
        shutil.copyfile(state/'meme-pools-v1/paper.db', capture/'paper.db')
        shutil.copyfile(discovery/'universe-snapshot.db', root/'universe.db')
        selected = select_cohort(capture/'paper.db', root/'universe.db', clock())
        atomic_json(root/'selection.json', selected)
        manifest = {'cohort': selected['cohort'], 'pools': {}, 'ledger_sha256': digest(capture/'paper.db')}
        for i, chosen in enumerate(selected['selected']):
            remote = Path(chosen['remote_checkpoint'])
            if remote.parent != state/'meme-pools-v1' or not remote.is_file() or remote.is_symlink():
                raise ValueError('Checkpoint path is outside the committed paper state')
            target = capture/f'pool{i}-paper-checkpoint.npz'; shutil.copyfile(remote, target)
            manifest['pools'][chosen['key']] = {k: v for k, v in chosen.items() if k != 'key'}
            manifest['pools'][chosen['key']]['checkpoint_sha256'] = digest(target)
        commit()
        # The preserved exporter requires capture after the complete last slot.
        # Keep the shared lease while waiting; no new observations or trading occur.
        cutoff = max((x['events'][-1]['slot']+1)*300 for x in selected['selected'])
        if cutoff-clock() > 300: raise ValueError('Checkpoint clock is ahead of the current slot')
        while clock() <= cutoff:
            if time.monotonic()-started > 330: raise TimeoutError('Checkpoint boundary wait exceeded its bound')
            sleep(min(10, max(.01, cutoff-clock()+.05)))
        manifest['captured_at'] = clock()
        atomic_json(capture/'capture.json', manifest); commit()
        audit = export(capture, root/'export', build_reference(state/'fly-data'))
        artifacts = {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*'))
                     if p.is_file() and p.name not in ('claim.json', 'result.json')}
        result = {'status': 'rate_checkpoint_exported', 'call_id': call_id, 'input_id': input_id,
                  'request_sha256': signature(request), 'remote_path': str(root),
                  'cohort': selected['cohort'], 'selection': selected, 'training_audit': audit,
                  'artifact_sha256': artifacts, 'native_reference_constructions': 1,
                  'neural_observations': 0, 'paper_orders': 0,
                  'budget': {**settle(state/'budget.json', budget, clock()-budget['started']),
                             'reserved_usd': budget['reserve'], 'nonpreemptible': True, 'price_multiplier': 3},
                  'interpretation': 'New pre-evaluation cohort and audited memory snapshot. '
                      'No evaluation window registered, no training performed, no policy selected.'}
        atomic_json(root/'result.json', result)
        claim.update(status='completed', completed_at=clock(), result_sha256=digest(root/'result.json'))
        atomic_json(root/'claim.json', claim); commit()
        return result
    except BaseException as exc:
        claim.update(status='failed', failed_at=clock(), error_type=type(exc).__name__, error=str(exc)[:300])
        atomic_json(root/'claim.json', claim); commit(); raise
