"""Reconstruct study 12 prices from raw collector receipts, without model compute."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3

from .core import Tick, atomic_json, digest
from .fly_market_study import quote_at
from .fly_rate_inputs import validate
from .multi import pool_tick
from .universe import Pool


def audit_prices(envelope, archive):
    p = validate(envelope);r = p['registration'];archive = Path(archive)
    if digest(archive) != p['snapshot_sha256']:
        raise ValueError('Raw price snapshot hash differs from sealed inputs')
    db = sqlite3.connect(f'file:{archive.resolve()}?mode=ro', uri=True)
    checked = {}
    try:
        if db.execute('PRAGMA integrity_check').fetchall() != [('ok',)]:
            raise ValueError('Invalid price snapshot database')
        end = db.execute("SELECT max(json_extract(payload,'$.observed')) FROM observations").fetchone()[0]
        if end != p['snapshot_end'] or end < r['end']:
            raise ValueError('Snapshot endpoint differs from sealed provenance')
        for key in r['cohort']:
            sequence = [];receipts = 0;future_excluded = 0;gap_markers = 0
            earliest_slot = int((r['end'] - 48 * 3600) // 60)
            rows = db.execute('SELECT slot,payload,reason FROM observations WHERE key=? AND slot>=? ORDER BY slot',
                              (key, earliest_slot))
            for slot, payload, reason in rows:
                pool = Pool(**json.loads(payload))
                if pool.key != key or slot != int(pool.observed // 60) or reason != pool.rejection():
                    raise ValueError('Collector receipt identity, minute or eligibility differs')
                if pool.observed > r['end']:
                    future_excluded += 1
                    continue
                if sequence and pool.observed <= sequence[-1].ts:
                    raise ValueError('Collector receipts are not chronological')
                if sequence and pool.observed - sequence[-1].ts > 180:
                    old = sequence[-1]
                    sequence.append(Tick(old.ts + 180, old.bid, old.ask, 0, key,
                                         old.source, old.ts + 180, False))
                    gap_markers += 1
                tick = pool_tick(pool)
                if not tick.available and sequence:
                    old = sequence[-1]
                    tick = Tick(tick.ts, old.bid, old.ask, 0, key, tick.source, tick.received_at, False)
                sequence.append(tick);receipts += 1
            if not sequence:
                raise ValueError('Registered pool absent from raw observation history')
            if r['end'] - sequence[-1].ts > 180:
                old = sequence[-1]
                sequence.append(Tick(old.ts + 180, old.bid, old.ask, 0, key,
                                     old.source, old.ts + 180, False))
                gap_markers += 1
            past = [t for t in sequence if t.ts <= r['development_start']]
            future = [t for t in sequence if t.ts > r['development_start']]
            if len(future) > 412:
                raise ValueError('Evaluation receipts exceed registered context budget')
            retained = past[-(512 - len(future)):] + future
            if [asdict(t) for t in retained] != p['series'][key]:
                raise ValueError('Sealed price history differs from raw collector receipts')
            coverage = {}
            for phase in ('development', 'test'):
                slots = [];last_quote = 0
                for i in range(r['phase_steps']+1):
                    stamp = r[phase + '_start'] + i * 300
                    _, quote = quote_at(retained, stamp)
                    observed = i < r['phase_steps'] and quote.available and quote.ts > last_quote
                    if observed:last_quote = quote.ts
                    slots.append({'decision_ts': stamp, 'quote_ts': quote.ts,
                                  'available': quote.available, 'terminal': i == r['phase_steps'],
                                  'neural_observation_expected': observed})
                coverage[phase] = slots
            checked[key] = {'raw_receipts_through_endpoint': receipts,
                            'post_endpoint_receipts_excluded': future_excluded,
                            'gap_markers': gap_markers, 'retained_context_rows': len(retained),
                            'coverage': coverage}
    finally:
        db.close()
    return {'plan_sha256': envelope['sha256'], 'snapshot_sha256': p['snapshot_sha256'],
            'snapshot_end': p['snapshot_end'], 'prices_reconstructed_from_snapshot': True,
            'verifier_sha256': digest(__file__), 'pools': checked}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('plan', 'archive', 'out'):parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    atomic_json(args.out, audit_prices(json.loads(args.plan.read_text()), args.archive))


if __name__ == '__main__':
    main()
