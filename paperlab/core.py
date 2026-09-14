from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class Tick:
    ts: float
    bid: float
    ask: float
    volume: float = 0
    product: str = "BTC-USD"
    source: str = "synthetic"
    received_at: float = 0
    available: bool = True

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.ts, self.bid, self.ask, self.volume, self.received_at)):
            raise ValueError("Nonfinite tick")
        if self.ts <= 0 or self.bid <= 0 or self.ask < self.bid or self.volume < 0:
            raise ValueError("Invalid tick")

    @property
    def mid(self):
        return (self.bid + self.ask) / 2


@dataclass(frozen=True)
class Costs:
    capital: float = 1000
    fee_bps: float = 60
    slippage_bps: float = 10
    max_exposure: float = 0.5
    max_order: float = 100
    max_spread_bps: float = 50
    max_delay: float = 1800
    loss_stop: float = 0.2

    def __post_init__(self):
        if not all(math.isfinite(float(x)) for x in asdict(self).values()):
            raise ValueError("Nonfinite cost")
        if self.capital <= 0 or self.max_order <= 0 or not 0 < self.max_exposure <= 1:
            raise ValueError("Invalid risk limits")
        if not 0 <= self.fee_bps < 10000 or not 0 <= self.slippage_bps < 10000:
            raise ValueError("Invalid costs")
        if not 0 < self.loss_stop <= 1 or self.max_delay <= 0 or self.max_spread_bps <= 0:
            raise ValueError("Invalid guard")


class Broker:
    """Decimal, long-only paper execution at a subsequent observation; never an API order."""
    def __init__(self, costs=Costs(), state=None):
        self.c = costs
        self.cash = Decimal(str(costs.capital))
        self.qty = Decimal(0)
        self.fees = Decimal(0)
        self.halted = False
        if state:
            for key in ("cash", "qty", "fees"):
                setattr(self, key, Decimal(state[key]))
            self.halted = bool(state["halted"])

    def state(self):
        return {"cash": str(self.cash), "qty": str(self.qty), "fees": str(self.fees), "halted": self.halted}

    def equity(self, tick):
        if not tick.available:
            # DEX inventory without a usable market has zero recoverable value in
            # the stress mark. It is not silently liquidated or sold at a stale price.
            return float(self.cash)
        # Bid liquidation value, including the assumed future exit fee and slippage.
        exit_factor = (1 - self.c.fee_bps / 10000) * (1 - self.c.slippage_bps / 10000)
        return float(self.cash + self.qty * Decimal(str(tick.bid * exit_factor)))

    def execute(self, target, decision_ts, tick):
        if not math.isfinite(target) or not 0 <= target <= self.c.max_exposure:
            raise ValueError("Target outside exposure limit")
        if not tick.available:
            return {"status": "rejected", "reason": "unavailable_market"}
        if tick.ts <= decision_ts:
            return {"status": "rejected", "reason": "not_after_decision"}
        if tick.ts - decision_ts > self.c.max_delay:
            return {"status": "rejected", "reason": "expired_decision"}
        if (tick.ask - tick.bid) / tick.mid * 10000 > self.c.max_spread_bps:
            return {"status": "rejected", "reason": "spread"}
        eq = self.equity(tick)
        if eq <= self.c.capital * (1 - self.c.loss_stop):
            self.halted = True
        desired = Decimal(str(eq * target / tick.mid))
        delta = desired - self.qty
        # The guard rejects new risk. It never substitutes its own trade for the model.
        if self.halted and delta > 0:
            return {"status": "rejected", "reason": "loss_stop"}
        if abs(float(delta) * tick.mid) < 1:
            return {"status": "hold"}
        buy = delta > 0
        px = Decimal(str(tick.ask * (1 + self.c.slippage_bps / 10000) if buy else tick.bid * (1 - self.c.slippage_bps / 10000)))
        rate = Decimal(str(self.c.fee_bps / 10000))
        qty = min(abs(delta), Decimal(str(self.c.max_order)) / px)
        qty = min(qty, self.cash / (px * (1 + rate)) if buy else self.qty)
        notional, fee = qty * px, qty * px * rate
        if notional < 1:
            return {"status": "hold"}
        self.cash += -notional - fee if buy else notional - fee
        self.qty += qty if buy else -qty
        self.fees += fee
        return {"status": "filled", "side": "BUY" if buy else "SELL", "quantity": str(qty), "price": str(px), "fee": str(fee), "decision_ts": decision_ts, "fill_ts": tick.ts, "simulation": True}


def load_ticks(path):
    ticks = [Tick(**json.loads(line)) for line in Path(path).read_text().splitlines() if line.strip()]
    if len(ticks) < 100:
        raise ValueError("At least 100 ticks required")
    if len({t.product for t in ticks}) != 1 or len({t.source for t in ticks}) != 1:
        raise ValueError("Do not mix products or historical/forward/synthetic data")
    if any(b.ts <= a.ts for a, b in zip(ticks, ticks[1:])):
        raise ValueError("Ticks must be strictly chronological and unique")
    return ticks


def features(ticks, i, broker, news):
    p = np.array([t.mid for t in ticks[max(0, i - 63):i + 1]], dtype=float)
    r = np.diff(np.log(p))
    rets = [math.log(p[-1] / p[max(0, len(p) - 1 - lag)]) * 100 for lag in (1, 3, 6, 12, 24, 48)]
    vol = [float(np.std(r[-n:]) * 100) if len(r) else 0 for n in (12, 48)]
    trend = [(p[-1] / p[-n:].mean() - 1) * 100 for n in (12, 48)]
    vols = [t.volume for t in ticks[max(0, i - 47):i + 1]]
    volume_ratio = math.log1p(ticks[i].volume) - math.log1p(float(np.mean(vols)))
    t = ticks[i]
    eq = max(broker.equity(t), 1e-6)
    market = rets + vol + trend + [volume_ratio, (t.ask - t.bid) / t.mid * 100, float(broker.qty) * t.mid / eq, (eq / broker.c.capital - 1)]
    obs = np.concatenate([market, news.features(t.ts)])
    if obs.shape != (64,) or not np.isfinite(obs).all():
        raise ValueError("Invalid observation")
    return np.clip(obs, -10, 10).astype(np.float32)


class Environment:
    def __init__(self, ticks, news, costs, start, end, stride=1):
        self.ticks, self.news, self.costs = ticks, news, costs
        self.start, self.end = start, end
        if not 63 <= start < end < len(ticks):
            raise ValueError("Invalid chronological environment bounds")
        if not isinstance(stride,int) or not 1<=stride<=5:
            raise ValueError("Invalid decision stride")
        self.stride=stride
        self.reset()

    def reset(self):
        self.i = self.start
        self.broker = Broker(self.costs)
        self.peak = self.costs.capital
        self.drawdown = 0
        return features(self.ticks, self.i, self.broker, self.news)

    def step(self, action):
        previous = self.broker.equity(self.ticks[self.i])
        target = [0, self.costs.max_exposure / 2, self.costs.max_exposure][int(action)]
        next_i=min(self.i+self.stride,self.end)
        fill = self.broker.execute(target, self.ticks[self.i].ts, self.ticks[next_i])
        self.i = next_i
        equity = self.broker.equity(self.ticks[self.i])
        self.peak = max(self.peak, equity)
        dd = 1 - equity / self.peak
        reward = 100 * (math.log(max(equity, 1e-8) / max(previous, 1e-8)) - 0.02 * max(0, dd - self.drawdown))
        self.drawdown = dd
        done = self.i >= self.end
        return features(self.ticks, self.i, self.broker, self.news), reward, done, {"equity": equity, "drawdown": dd, "fill": fill, "ts": self.ticks[self.i].ts}
