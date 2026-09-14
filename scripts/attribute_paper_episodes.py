"""Reconcile marked episode PnL by mint; no inference or remote mutations."""
import argparse
import json
from decimal import Decimal as D
from pathlib import Path


def attribute(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert float(rows[0]['portfolio']['cash']) == 1000
    assert not rows[0]['portfolio']['positions']
    final = rows[-1]
    flows, fees, fills = {}, {}, {}
    for row in rows:
        for execution in row['executions']:
            f, mint = execution['fill'], execution['mint']
            if f['status'] != 'filled':
                continue
            assert f['simulation']
            amount, fee = D(f['quantity']) * D(f['price']), D(f['fee'])
            delta = -amount-fee if f['side'] == 'BUY' else amount-fee
            flows[mint] = flows.get(mint, D(0)) + delta
            fees[mint] = fees.get(mint, D(0)) + fee
            fills[mint] = fills.get(mint, 0) + 1
    tokens = []
    for mint, p in final['portfolio']['positions'].items():
        assert abs(flows.get(mint, D(0))-D(p['cash_flow'])) < D('1e-7')
        if not fills.get(mint):
            continue
        quote = final['marks'].get(mint)
        value = float(p['qty'])*quote['bid']*.99*.9875 if quote and quote['available'] else 0.
        # Acquisition basis includes entry fees; sale proceeds include exit fees.
        realized = float(p['cash_flow']) + float(p['basis'])
        unrealized = value - float(p['basis'])
        tokens.append(dict(mint=mint, realized_usd=realized, unrealized_usd=unrealized,
                           pnl_usd=realized+unrealized, marked_inventory_usd=value,
                           paid_fees_usd=float(fees[mint]), fills=fills[mint],
                           quote_available=bool(quote and quote['available']),
                           remaining_basis_usd=float(p['basis'])))
    pnl = final['equity_stress_usd']-1000
    assert abs(sum(t['pnl_usd'] for t in tokens)-pnl) < 1e-7
    assert abs(sum(fees.values())-D(final['portfolio']['fees'])) < D('1e-7')
    assert abs(D(1000)+sum(flows.values())-D(final['portfolio']['cash'])) < D('1e-7')
    return dict(run_id=final['episode_id'], rows=len(rows), pnl_usd=pnl,
                cash_usd=float(final['portfolio']['cash']),
                realized_usd=sum(t['realized_usd'] for t in tokens),
                unrealized_usd=sum(t['unrealized_usd'] for t in tokens),
                paid_fees_usd=float(sum(fees.values())), tokens=tokens,
                source=str(path), ended=final['at'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ledgers', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    episodes = [attribute(p) for p in args.ledgers]
    totals = {key: sum(e[key] for e in episodes) for key in
              ('pnl_usd', 'realized_usd', 'unrealized_usd', 'paid_fees_usd')}
    result = dict(episodes=episodes, totals=totals,
                  scope='Three independent fresh $1,000 paper accounts; not compounded.',
                  valuation='Recorded bid less 1% slippage and 1.25% exit fee; missing quote values inventory at zero.',
                  realized_method='Average acquisition cost including entry fees; net simulated sale proceeds.',
                  checks='Per-mint fill cash flows, cash, fees and summed marked PnL reconcile.',
                  profitable_learning_proven=False)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(totals, indent=2))


if __name__ == '__main__':
    main()
