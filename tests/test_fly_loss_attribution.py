"""Hand-calculated accounting bridges; no native model construction."""
from copy import deepcopy

import pytest

from paperlab.core import Broker, Costs, Tick
from paperlab.fly_loss_attribution import decompose
from paperlab.fly_market_study import quote_at


def example(*, sell=False, gap=False, stale=False, zero_cost=False):
    costs = Costs(fee_bps=0 if zero_cost else 100,
                  slippage_bps=0 if zero_cost else 200,
                  max_order=101 if zero_cost else 103.02, max_spread_bps=500)
    ticks = [Tick(100, 99, 101), Tick(200, 119, 121)]
    stamps = [100, 200]
    if gap:
        ticks.append(Tick(300, 119, 121, available=False)); stamps.append(300)
    if stale:
        stamps.append(400)
    if gap or stale:
        ticks.append(Tick(500, 129, 131)); stamps.append(500)
    broker = Broker(costs); rows = []
    for i, stamp in enumerate(stamps):
        _, quote = quote_at(ticks, stamp)
        fill = broker.execute(.5, 99, quote) if i == 0 else {'status':'hold'}
        if i == 1 and sell:
            # max_order bounds sales too; this is deliberately a partial sale.
            fill = broker.execute(0, 199, quote)
        rows.append({'decision_ts':stamp, 'quote_ts':quote.ts, 'available':quote.available,
                     'fill':fill, 'broker':broker.state(), 'equity':broker.equity(quote)})
    return rows, ticks, costs


def test_one_unit_open_inventory_hand_calculation():
    result = decompose(*example())
    end = result['terminal']
    assert end['inventory_quantity'] == '1'
    assert end['gross_reference_pnl_usd'] == pytest.approx(20)
    assert end['paid_fees_usd'] == pytest.approx(1.0302)
    assert end['execution_spread_usd'] == pytest.approx(1)
    assert end['execution_slippage_usd'] == pytest.approx(2.02)
    assert end['exit_allowance_usd'] == pytest.approx(4.5462)
    assert end['net_pnl_usd'] == pytest.approx(11.4036)
    assert result['maximum_identity_error_usd'] < 1e-10


def test_partial_sale_spread_and_slippage():
    end = decompose(*example(sell=True))['terminal']
    sold = 103.02 / 116.62
    assert end['gross_reference_pnl_usd'] == pytest.approx(20)
    assert end['paid_fees_usd'] == pytest.approx(2.0604)
    assert end['execution_spread_usd'] == pytest.approx(1 + sold)
    assert end['execution_slippage_usd'] == pytest.approx(2.02 + sold * 2.38)
    assert end['exit_allowance_usd'] == pytest.approx((1-sold) * 4.5462)
    # Selling at the current liquidation assumptions realizes costs already marked.
    assert end['net_pnl_usd'] == pytest.approx(11.4036)


def test_zero_fees_and_slippage_still_has_bid_ask_cost():
    end = decompose(*example(zero_cost=True))['terminal']
    assert end['gross_reference_pnl_usd'] == pytest.approx(20)
    assert end['net_pnl_usd'] == pytest.approx(18)
    assert end['execution_spread_usd'] == 1
    assert end['exit_allowance_usd'] == 1
    assert end['execution_slippage_usd'] == end['paid_fees_usd'] == 0


@pytest.mark.parametrize('kind', ['gap', 'stale'])
def test_missing_quote_deducts_stale_inventory_then_recovers(kind):
    result = decompose(*example(**{kind:True}))
    missing = result['rows'][2]
    assert not missing['available']
    assert missing['reference_ts'] == 200
    assert missing['reference_kind'] == 'stale_reference_only_not_liquidatable'
    assert missing['missing_reference_deduction_usd'] == 120
    assert missing['exit_allowance_usd'] == 0
    assert missing['net_pnl_usd'] == pytest.approx(-104.0502)
    assert result['terminal']['missing_reference_deduction_usd'] == 0
    assert result['terminal']['gross_reference_pnl_usd'] == 30
    assert result['missing_marks'] == 1


@pytest.mark.parametrize('field', ['fee', 'price', 'quantity', 'cash', 'quote', 'equity'])
def test_damaged_evidence_rejected(field):
    rows, ticks, costs = example(); rows = deepcopy(rows)
    if field in ('fee', 'price', 'quantity'):
        rows[0]['fill'][field] = '42'
    elif field == 'cash': rows[0]['broker']['cash'] = '42'
    elif field == 'quote': rows[0]['quote_ts'] += 1
    else: rows[-1]['equity'] += 1
    with pytest.raises(ValueError): decompose(rows, ticks, costs)
