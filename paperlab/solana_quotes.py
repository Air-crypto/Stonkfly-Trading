"""Versioned indicative quotes, entry guards, and exit capacity for paper episodes.

An entry filter is not a statement that an already held token is worth zero.
These reserve-derived observations are still not executable exchange quotes.
Historical ``solana_paper.tick_for`` behavior is intentionally unchanged.
"""
from dataclasses import dataclass
from decimal import Decimal
import math

from .core import Tick
from .solana_events import SOL, ZERO, TOKEN, TOKEN22, eligible

QUOTE_PROTOCOL = 'solana_observed_entry_exit_v2'
LIQUIDITY_FRACTION = .01


@dataclass(frozen=True)
class QuoteSnapshot:
    observed_tick: Tick | None
    entry_tick: Tick | None
    exit_tick: Tick | None
    observation_reason: str | None
    entry_reason: str | None
    exit_reason: str | None
    liquidity_notional_usd: float
    protocol: str = QUOTE_PROTOCOL

    def for_target(self, target, cash, qty):
        """Pick entry/exit eligibility from actual risk at the later receipt.

        A positive LONG target can request a sale after the price changes. It
        must not inherit a new-entry rejection merely because its target is nonzero.
        """
        if not math.isfinite(target) or not 0 <= target <= 1:
            raise ValueError('Invalid exposure target')
        cash, qty = Decimal(str(cash)), Decimal(str(qty))
        if not cash.is_finite() or not qty.is_finite() or cash < 0 or qty < 0:
            raise ValueError('Invalid cash or inventory')
        tick = self.observed_tick
        if tick is None:
            return None, self.observation_reason
        equity = float(cash+qty*Decimal(str(tick.bid*.99*.9875)))
        requested = Decimal(str(equity*target/tick.mid))-qty
        buying = requested*Decimal(str(tick.mid)) >= 1
        return (self.entry_tick, self.entry_reason) if buying else (self.exit_tick, self.exit_reason)

    def position_values(self, qty, *, fee_bps=125., slippage_bps=100.):
        """Report a full indicative mark and a *single* permitted fill separately.

        The unsized mark assumes the entire lot could obtain the indicated price.
        The stress value includes only the next fill's allowed capacity, valuing
        its remainder at zero. Neither figure proves exchange executability, and
        the latter is a capacity sensitivity, not a claim of permanent loss.
        """
        qty = Decimal(str(qty))
        if not qty.is_finite() or qty < 0:
            raise ValueError('Invalid inventory quantity')
        if not all(math.isfinite(v) and 0 <= v < 10000 for v in (fee_bps, slippage_bps)):
            raise ValueError('Invalid valuation costs')
        tick = self.observed_tick
        exit_price = tick.bid * (1-slippage_bps/10000) if tick else 0.
        gross = float(qty) * exit_price
        factor = 1-fee_bps/10000
        mark = gross * factor
        capacity = min(1000., self.liquidity_notional_usd) if self.exit_tick else 0.
        next_gross = min(gross, capacity)
        if next_gross < 1.:
            next_gross = 0.
        proceeds = next_gross * factor
        return dict(indicative_liquidation_value_usd=mark,
                    next_fill_proceeds_usd=proceeds,
                    capacity_stress_value_usd=proceeds,
                    indicative_value_beyond_next_fill_usd=max(0., mark-proceeds),
                    full_exit_within_next_fill=bool(gross >= 1. and gross <= capacity),
                    quote_protocol=self.protocol)


def quote_for(token, now, sol_usd, fx_seen, *, selected=False, connected=True):
    """Separate observing a price, admitting risk, and allowing a smaller exit.

    Entries retain the launch/flow guards. Both entry and exit order sizes are
    bounded at execution time to 1% of *actual* SOL reserves, rather than treating
    the legacy fixed $25 order as a prerequisite for observing any price at all.
    """
    def unavailable(reason):
        return QuoteSnapshot(None, None, None, reason, reason, reason, 0.)

    if not connected:
        return unavailable('disconnected_feed')
    if not math.isfinite(sol_usd) or sol_usd <= 0 or not 0 <= now-fx_seen <= 90:
        return unavailable('stale_sol_usd')
    created = token['created']
    if created['quote_mint'] not in (SOL, ZERO):
        return unavailable('unsupported_quote')
    if created['token_program'] not in (TOKEN, TOKEN22):
        return unavailable('unsupported_token_program')
    if created['is_mayhem_mode']:
        return unavailable('mayhem_mode')
    if token['complete'] and not token.get('migrated'):
        return unavailable('curve_completed')
    if not token['trades']:
        return unavailable('missing_trade')
    last = token['trades'][-1]
    if token['complete'] and last.get('venue') != 'pumpswap':
        return unavailable('awaiting_migration_quote')
    if not 0 <= now-last['received'] <= 10 or not -2 <= now-last['timestamp'] <= 20:
        return unavailable('stale_trade')
    if last.get('quote_mint') not in (SOL, ZERO):
        return unavailable('unsupported_trade_quote')
    if last['mayhem_mode']:
        return unavailable('mayhem_mode')
    if min(last['virtual_sol_reserves'], last['virtual_token_reserves']) <= 0 or last['real_sol_reserves'] < 0:
        return unavailable('invalid_reserves')
    mid = last['virtual_sol_reserves']/last['virtual_token_reserves']/1e9*sol_usd
    tick = Tick(last['received'], mid*.995, mid*1.005, product=last['mint'],
                source='solana_confirmed_pump_reserve_indicative',
                received_at=last['received'], available=True)
    liquidity = last['real_sol_reserves']/1e9*sol_usd*LIQUIDITY_FRACTION
    exit_reason = None if liquidity >= 1. else 'insufficient_observed_liquidity'
    entry_reason = eligible(token, now, selected=selected) or exit_reason
    return QuoteSnapshot(tick, tick if entry_reason is None else None,
                         tick if exit_reason is None else None, None,
                         entry_reason, exit_reason, liquidity)
