import copy
from dataclasses import replace
import json
from pathlib import Path
import sqlite3

import pytest

from paperlab.core import digest
from paperlab.fly_market_study import signature
from paperlab.fly_paper_inputs import seal
from paperlab.fly_paper_price_audit import audit_prices
from paperlab.news import News
from paperlab.universe import Pool, Store


@pytest.fixture
def sealed(tmp_path):
    r = json.loads(Path('reports/fly-market-study-09-preregistration.json').read_text())
    archive = tmp_path/'universe.db';store = Store(archive)
    start = r['development_start']
    for minute in range(-680, 46):
        ts = start + minute*60
        for i, key in enumerate(r['cohort']):
            # One pool disappears mid-development; the other has a shorter gap
            # and a low-activity receipt. Future provider prices must be excluded.
            if i == 1 and minute >= 11:continue
            if i == 0 and 2 <= minute <= 8:continue
            pool = Pool(key, 'solana', key.split(':')[1], f'token{i}', f'FIXTURE{i}',
                        1 + (minute+680)/1000 if ts <= r['end'] else 99999,
                        100000, 0 if minute == 17 else 10000, 20, 20, ts-3600, ts,
                        source='synthetic')
            store.add([pool])
            # Repeated observations within a minute cannot revise the receipt.
            store.add([replace(pool, price=123456, observed=ts+1)])
    store.db.close()
    news = tmp_path/'news.db';reader = News(news);reader.db.close()
    env = seal(archive, news, 'reports/fly-market-study-09-preregistration.json',
               'reports/fly-paper-memory-audit-01.json', 'reports/fly-market-study-08-plan.json',
               tmp_path/'plan.json')
    return env, archive


def test_raw_prices_match_sealed_context_including_disappearance_and_truncation(sealed):
    env, archive = sealed;result = audit_prices(env, archive)
    assert result['prices_reconstructed_from_snapshot']
    first, second = [result['pools'][k] for k in env['plan']['registration']['cohort']]
    assert first['retained_context_rows'] == second['retained_context_rows'] == 512
    assert first['post_endpoint_receipts_excluded'] == 15
    assert first['gap_markers'] == second['gap_markers'] == 1
    assert all(not r['neural_observation_expected'] for r in second['coverage']['test'])
    assert any(not r['available'] for r in first['coverage']['development'])
    assert all(t['bid'] < 10 for values in env['plan']['series'].values() for t in values)


@pytest.mark.parametrize('mutation', ['price', 'availability', 'history', 'endpoint'])
def test_rehashed_plan_cannot_substitute_another_price_history(sealed, mutation):
    original, archive = sealed;env = copy.deepcopy(original)
    key = env['plan']['registration']['cohort'][0]
    if mutation == 'price':env['plan']['series'][key][0]['bid'] += .01
    if mutation == 'availability':env['plan']['series'][key][0]['available'] = False
    if mutation == 'history':env['plan']['series'][key].pop(0)
    if mutation == 'endpoint':env['plan']['snapshot_end'] += 1
    env['sha256'] = signature(env['plan'])
    with pytest.raises(ValueError, match='history|endpoint'):audit_prices(env, archive)


def test_snapshot_hash_and_receipt_identity_cannot_be_substituted(sealed):
    env, archive = sealed
    with sqlite3.connect(archive) as db:
        key, slot, payload = db.execute('SELECT key,slot,payload FROM observations LIMIT 1').fetchone()
        p = json.loads(payload);p['key'] = 'solana:wrong'
        db.execute('UPDATE observations SET payload=? WHERE key=? AND slot=?', (json.dumps(p), key, slot))
    with pytest.raises(ValueError, match='hash'):audit_prices(env, archive)
    changed = copy.deepcopy(env);changed['plan']['snapshot_sha256'] = digest(archive)
    changed['sha256'] = signature(changed['plan'])
    with pytest.raises(ValueError, match='identity'):audit_prices(changed, archive)
