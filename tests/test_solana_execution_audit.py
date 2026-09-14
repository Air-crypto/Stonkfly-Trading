from copy import deepcopy

import pytest

from paperlab.solana_execution_audit import verify_fill_intent


def records():
    # Deliberately hand-constructed: no production broker computes this fixture.
    decision = dict(mint='mint-a', action=1, target=1., issued=100.)
    tick = dict(product='mint-a', bid=1., ask=1.01, ts=105., available=True)
    fill = dict(side='BUY', quantity='5', decision_ts=100.)
    return decision, fill, tick


def verify(decision, fill, tick, **kwargs):
    return verify_fill_intent(decision, fill, tick, cash_before=1000,
                             qty_before=kwargs.get('qty', 0), max_exposure=1.)


def test_partial_buy_implements_long_intent():
    assert verify(*records())


@pytest.mark.parametrize('corruption', ['action', 'target', 'mint', 'time', 'side', 'quantity'])
def test_tampered_intent_does_not_pass_accounting_audit(corruption):
    decision, fill, tick = deepcopy(records())
    if corruption == 'action': decision['action'] = 0
    if corruption == 'target': decision.update(action=0, target=0.)
    if corruption == 'mint': decision['mint'] = 'mint-b'
    if corruption == 'time': decision['issued'] = 99.
    if corruption == 'side': fill['side'] = 'SELL'
    if corruption == 'quantity': fill['quantity'] = '2000'
    with pytest.raises(ValueError): verify(decision, fill, tick)


def test_exit_and_long_rebalance_can_both_sell():
    decision, fill, tick = records()
    decision.update(action=0, target=0.)
    fill.update(side='SELL', quantity='5')
    assert verify(decision, fill, tick, qty=10)
    # LONG can issue a reduced target; no blanket ban on SELL for action 1.
    decision.update(action=1, target=.001)
    assert verify(decision, fill, tick, qty=10)


def test_sub_dollar_noop_target_cannot_create_fill():
    decision, fill, tick = records()
    decision['target'] = .0001
    fill['quantity'] = '.01'
    with pytest.raises(ValueError, match='target'):
        verify(decision, fill, tick)
