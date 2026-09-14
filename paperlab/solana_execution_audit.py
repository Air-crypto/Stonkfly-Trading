"""Independent checks that a recorded fill implements the issued paper intent."""
from decimal import Decimal
import math


def verify_fill_intent(decision, fill, tick, *, cash_before, qty_before,
                       max_exposure, fee_bps=125., slippage_bps=100.):
    """Validate action semantics and the requested side/quantity without a broker.

    A LONG target may sell after the market moves, so the side is determined from
    the issued target at the actual later receipt, not merely the action label.
    Risk and liquidity guards may further reduce a requested quantity; independent
    ledger auditing verifies those limits and the resulting cash flows separately.
    """
    action = decision['action']
    target = float(decision['target'])
    if type(action) is not int or action not in (0, 1):
        raise ValueError('Invalid issued action')
    if not math.isfinite(target) or not 0 <= target <= max_exposure:
        raise ValueError('Issued target outside exposure limit')
    if (action == 0 and target != 0) or (action == 1 and target <= 0):
        raise ValueError('Issued target contradicts action')
    if tick['product'] != decision['mint']:
        raise ValueError('Fill market differs from issued mint')
    if fill['decision_ts'] != decision['issued']:
        raise ValueError('Fill refers to a different issued decision')
    cash, qty = Decimal(str(cash_before)), Decimal(str(qty_before))
    if not cash.is_finite() or not qty.is_finite() or cash < 0 or qty < 0:
        raise ValueError('Invalid opening fill inventory')
    if not tick['available'] or not tick['ts'] > decision['issued']:
        raise ValueError('Unavailable or noncausal target fill')
    mid = (tick['bid']+tick['ask'])/2
    exit_factor = (1-slippage_bps/10000)*(1-fee_bps/10000)
    equity = float(cash + qty*Decimal(str(tick['bid']*exit_factor)))
    requested = Decimal(str(equity*target/mid))-qty
    expected_side = 'BUY' if requested > 0 else 'SELL'
    if fill['side'] != expected_side or abs(float(requested)*mid) < 1:
        raise ValueError('Fill side contradicts issued target')
    amount = Decimal(fill['quantity'])
    tolerance = max(Decimal('1e-8'), abs(requested)*Decimal('1e-10'))
    if not amount.is_finite() or amount <= 0 or amount > abs(requested)+tolerance:
        raise ValueError('Filled quantity exceeds issued target')
    return True
