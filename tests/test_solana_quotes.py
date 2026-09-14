from copy import deepcopy

import pytest

from paperlab.solana_events import SOL, TOKEN
from paperlab.solana_paper import tick_for
from paperlab.solana_quotes import QUOTE_PROTOCOL, quote_for


def token(now=1000):
    return dict(complete=False, created=dict(timestamp=now-100, received=now-99,
        quote_mint=SOL, token_program=TOKEN, is_mayhem_mode=False),
        trades=[dict(received=now-6+i, timestamp=now-6+i, is_buy=bool(i % 2),
            mint='fixture', quote_mint=SOL, mayhem_mode=False,
            virtual_sol_reserves=100_000_000_000,
            virtual_token_reserves=1_000_000_000_000,
            real_sol_reserves=10_000_000_000) for i in range(6)])


def test_dynamic_capacity_preserves_historical_protocol():
    source = token()
    before = deepcopy(source)
    old, reason = tick_for(source, 1000, 100., 999)
    assert not old.available and reason == 'order_exceeds_one_percent_real_sol_reserves'
    current = quote_for(source, 1000, 100., 999)
    assert current.protocol == QUOTE_PROTOCOL
    assert current.entry_tick and current.exit_tick and current.observed_tick
    assert current.liquidity_notional_usd == 10.
    assert source == before


def test_one_sided_flow_blocks_entry_but_not_observation_or_exit():
    source = token()
    for trade in source['trades']:
        trade['is_buy'] = False
    current = quote_for(source, 1000, 100., 999)
    assert current.entry_tick is None and current.entry_reason == 'one_sided'
    assert current.observed_tick is not None and current.exit_tick is not None
    assert current.observation_reason is None and current.exit_reason is None


def test_positive_target_rebalance_uses_exit_eligibility():
    source = token()
    for trade in source['trades']:
        trade['is_buy'] = False
    current = quote_for(source, 1000, 100., 999)
    # A tiny positive target against a $100 position requests risk reduction.
    tick, reason = current.for_target(.001, 1000, 10_000_000_000)
    assert tick is current.exit_tick and reason is None
    tick, reason = current.for_target(1., 1000, 0)
    assert tick is None and reason == 'one_sided'


def test_thin_market_can_allow_partial_exit_without_new_entry():
    source = token()
    source['trades'][-1]['real_sol_reserves'] = 2_000_000_000
    current = quote_for(source, 1000, 100., 999)
    assert current.entry_reason == 'thin_real_reserves' and current.entry_tick is None
    assert current.exit_tick and current.liquidity_notional_usd == 2.
    values = current.position_values(20_000_000_000)
    assert values['indicative_liquidation_value_usd'] > 190.
    assert values['next_fill_proceeds_usd'] == pytest.approx(2*.9875)
    assert not values['full_exit_within_next_fill']
    assert values['capacity_stress_value_usd'] < values['indicative_liquidation_value_usd']


def test_sub_dollar_capacity_retains_price_without_inventing_fill():
    source = token()
    source['trades'][-1]['real_sol_reserves'] = 500_000_000
    current = quote_for(source, 1000, 100., 999)
    assert current.observed_tick and current.exit_tick is None
    assert current.exit_reason == 'insufficient_observed_liquidity'
    assert current.position_values(100_000_000)['next_fill_proceeds_usd'] == 0.


@pytest.mark.parametrize('case,reason', [
    ('receipt', 'stale_trade'), ('future', 'stale_trade'), ('fx', 'stale_sol_usd'),
    ('disconnected', 'disconnected_feed'), ('complete', 'curve_completed'),
    ('invalid_reserves', 'invalid_reserves'), ('unsupported', 'unsupported_quote')])
def test_observation_provenance_fails_closed(case, reason):
    source = token(); fx_seen = 999; connected = True
    if case == 'receipt': source['trades'][-1]['received'] = 989
    if case == 'future': source['trades'][-1]['received'] = 1001
    if case == 'fx': fx_seen = 900
    if case == 'disconnected': connected = False
    if case == 'complete': source['complete'] = True
    if case == 'invalid_reserves': source['trades'][-1]['virtual_token_reserves'] = 0
    if case == 'unsupported': source['created']['quote_mint'] = 'not-sol'
    current = quote_for(source, 1000, 100., fx_seen, connected=connected)
    assert current.observation_reason == reason
    assert current.observed_tick is current.entry_tick is current.exit_tick is None
    assert current.position_values(100)['indicative_liquidation_value_usd'] == 0.


def test_full_exit_capacity_and_invalid_inventory():
    current = quote_for(token(), 1000, 100., 999)
    values = current.position_values(200_000_000)
    assert values['full_exit_within_next_fill']
    assert values['next_fill_proceeds_usd'] == values['indicative_liquidation_value_usd']
    assert values['indicative_value_beyond_next_fill_usd'] == 0.
    for qty in ('NaN', '-1', 'Infinity'):
        with pytest.raises(ValueError, match='quantity'): current.position_values(qty)
