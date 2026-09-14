"""Classify historical unavailable marks from archived receipts, without inference."""
import argparse
import bisect
from collections import Counter
from decimal import Decimal
import json
from pathlib import Path
import sqlite3
import tempfile

from paperlab.core import digest
from paperlab.solana_events import Feed
from paperlab.solana_execution_audit import verify_fill_intent
from paperlab.solana_paper import tick_for
from paperlab.solana_quotes import quote_for, QUOTE_PROTOCOL
from scripts.attribute_paper_episodes import attribute


def audit_episode(path):
    rows = [json.loads(s) for s in (path/'decisions.jsonl').read_text().splitlines()]
    final = rows[-1]
    now = final['at']
    fx = [json.loads(s) for s in (path/'fx.jsonl').read_text().splitlines()]
    fx_index = bisect.bisect_right([r['at'] for r in fx], now)-1
    if fx_index < 0:
        raise ValueError('No archived contemporaneous FX')
    issued, qty = {}, {}
    cash, fills = Decimal(1000), 0
    if rows[0]['portfolio']['positions'] or Decimal(rows[0]['portfolio']['cash']) != cash:
        raise ValueError('Expected independently reset historical paper episode')
    for row in rows:
        for execution in row['executions']:
            fill, mint = execution['fill'], execution['mint']
            if fill['status'] != 'filled':
                continue
            verify_fill_intent(issued[mint, fill['decision_ts']], fill, execution['tick'],
                              cash_before=cash, qty_before=qty.get(mint, Decimal(0)), max_exposure=1.)
            amount, price, fee = (Decimal(fill[k]) for k in ('quantity', 'price', 'fee'))
            buying = fill['side'] == 'BUY'
            cash += -amount*price-fee if buying else amount*price-fee
            qty[mint] = qty.get(mint, Decimal(0))+(amount if buying else -amount)
            fills += 1
        if row['decision']:
            d = row['decision']; issued[d['mint'], d['issued']] = d
    unavailable = []
    db = sqlite3.connect(f'file:{path / "events.db"}?mode=ro', uri=True)
    try:
        with tempfile.TemporaryDirectory(prefix='stonkfly-quote-audit-') as scratch:
            feed = Feed(scratch)
            for (body,) in db.execute('SELECT body FROM events WHERE received<=? ORDER BY received,rowid', (now,)):
                feed.accept(json.loads(body))
            for mint in final['unavailable_positions']:
                token = feed.tokens.get(mint)
                if token is None:
                    raise ValueError('Historical held mint not retained in reconstruction')
                f = fx[fx_index]
                _, reason = tick_for(token, now, f['sol_usd'], f['at'], selected=True)
                connected = final['feed']['status'] == 'connected' and 0 <= now-final['feed']['last_message'] <= 10
                reason = reason if connected else 'disconnected_feed'
                quote = quote_for(token, now, f['sol_usd'], f['at'], selected=True, connected=connected)
                position = final['portfolio']['positions'][mint]
                unavailable.append(dict(mint=mint, basis_usd=float(position['basis']),
                    historical_reason=reason, last_trade_age_seconds=now-token['trades'][-1]['received'],
                    v2_observation_reason=quote.observation_reason, v2_entry_reason=quote.entry_reason,
                    v2_exit_reason=quote.exit_reason,
                    diagnostic_v2_values=quote.position_values(position['qty'])))
    finally:
        db.close()
    accounting = attribute(path/'decisions.jsonl')
    return dict(run_id=path.name, ended=now, fills_with_intent_verified=fills,
                historical_pnl_usd=accounting['pnl_usd'], unavailable_positions=unavailable,
                source_sha256={name: digest(path/name) for name in ('events.db', 'fx.jsonl', 'decisions.jsonl')})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    episodes = [audit_episode(p) for p in sorted(args.source.glob('solana-online-*'))]
    counts, basis = Counter(), Counter()
    for episode in episodes:
        for position in episode['unavailable_positions']:
            reason = position['historical_reason'] or 'no_rejection_at_final_row_time'
            counts[reason] += 1
            basis[reason] += position['basis_usd']
    result = dict(episodes=episodes, episode_count=len(episodes),
        fills_with_intent_verified=sum(e['fills_with_intent_verified'] for e in episodes),
        historical_total_pnl_usd=sum(e['historical_pnl_usd'] for e in episodes),
        unavailable_positions_by_reason=dict(counts), unavailable_basis_by_reason_usd=dict(basis),
        prospective_quote_protocol=QUOTE_PROTOCOL,
        method='Read-only archive replay through final recorded receipt time; historical entry/quote rejection classified separately from v2 observation and exit eligibility. Fill intents and accounting independently checked.',
        limitations=['V2 values are diagnostic sensitivities, not rewritten historical returns or executed policy counterfactuals.',
                     'Raw indicative mark assumes all inventory at that price; next-fill capacity is separately bounded.',
                     'Public confirmed Pump stream has gaps and no completeness guarantee.',
                     'Sample consists of eleven independently reset episodes on one date; not evidence of learning.'])
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'episodes'}, indent=2))


if __name__ == '__main__':
    main()
