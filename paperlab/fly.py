"""Adapter around the unchanged full Stonkfly graph, decoder and candidate plasticity."""
from dataclasses import asdict
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

from .core import Broker, Costs, atomic_json

UPSTREAM_COMMIT = "78ef3e05ab0fa086032098558d893667068944a0"


def configure(data):
    data = str(Path(data).resolve())
    if "stonkfly.neural.common" in sys.modules and str(sys.modules["stonkfly.neural.common"].DATA) != data:
        raise RuntimeError("Fly data directory cannot change after importing upstream")
    os.environ["STONKFLY_DATA"] = data
    vendor = Path(__file__).resolve().parents[1] / "vendor/stonkfly"
    if not (vendor / "stonkfly/neural/controller.py").exists():
        raise RuntimeError("Run from the source distribution with vendor/stonkfly present")
    sys.path.insert(0, str(vendor))


def prepare(data):
    configure(data)
    from stonkfly.data import prepare as upstream_prepare
    upstream_prepare()


def frame(ticks, i, news):
    from stonkfly.display import market_frame
    t = ticks[i]
    im = Image.fromarray(market_frame(t.product, [x.mid for x in ticks[max(0, i - 99):i + 1]], t.bid, t.ask))
    d = ImageDraw.Draw(im)
    features = news.features(t.ts)
    # Fixed visual code, no hidden suggested action. Colors encode sentiment distributions;
    # intensity cells encode hashed headline features. No headline-reading claim for the fly.
    d.rectangle((0, 140, 319, 179), fill=(22, 27, 40))
    for k, color in enumerate(((30, 210, 120), (220, 60, 85), (125, 135, 160))):
        x = k * 106
        d.rectangle((x, 141, x + int(features[k] * 104), 149), fill=color)
    for k, value in enumerate(features[3:]):
        level = int(np.clip(128 + 35 * value, 0, 255))
        x = k * 320 // 47
        d.rectangle((x, 154, x + 5, 174), fill=(level, level, level))
    return np.array(im, dtype=np.uint8)


class Fly:
    def __init__(self, data, learning=True, checkpoint=None):
        configure(data)
        from stonkfly.data import verify
        from stonkfly.config import Settings
        from stonkfly.neural.controller import FlyController
        self.graph = verify()
        self.learning = learning
        self.controller = FlyController(Settings(learning=learning))
        if checkpoint:
            self.controller.restore(checkpoint)
            self.controller.brain.weights_frozen = not learning

    def observe(self, ticks, i, news, delta):
        reinforcement = "reward" if delta > .01 else "aversive" if delta < -.01 else "none"
        return self.controller.observe(frame(ticks, i, news), reinforcement)

    def save(self, path):
        self.controller.save(path)


def replay(ticks, news, data, output, steps=8, learning=True, checkpoint=None, costs=Costs(), start=63):
    if steps < 1 or start < 63 or start + steps >= len(ticks):
        raise ValueError("Invalid replay bounds")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    fly = Fly(data, learning, checkpoint)
    init_seconds = time.perf_counter() - started
    broker = Broker(costs)
    anchor = costs.capital
    pending = None
    rows = []
    for i in range(start, start + steps):
        t = ticks[i]
        fill = broker.execute(pending[0], pending[1], t) if pending else {"status": "hold"}
        equity = broker.equity(t)
        delta = equity - anchor
        event = fly.observe(ticks, i, news, delta)
        # HOLD means no new order. Limits never map a rejected neural output to another action.
        pending = (costs.max_exposure if event["side"] == "BUY" else 0, t.ts) if event["side"] != "HOLD" else None
        rows.append({"ts": t.ts, "equity": equity, "delta": delta, "fill": fill, "neural": event})
        anchor = equity
        atomic_json(output / "ledger.json", rows)
    # Mark the final open position at a new observation without inventing a final model decision.
    last = ticks[start + steps]
    final_fill = broker.execute(pending[0], pending[1], last) if pending else {"status": "hold"}
    fly.save(output / "fly.npz")
    result = {"upstream_commit": UPSTREAM_COMMIT, "graph": fly.graph, "learning": learning, "news_enabled": news.enabled, "source": ticks[0].source, "product": ticks[0].product, "steps": steps, "start_ts": ticks[start].ts, "end_ts": last.ts, "initialization_seconds": init_seconds, "wall_seconds": time.perf_counter() - started, "neural_seconds": sum(r["neural"]["compute_seconds"] for r in rows), "end_equity": broker.equity(last), "fees": float(broker.fees), "final_fill": final_fill, "memory": fly.controller.brain.memory(), "costs": asdict(costs), "note": "Short mechanics/learning experiment. No evidence of profitable trading. Online fly plasticity is not PPO."}
    atomic_json(output / "metrics.json", result)
    return result
