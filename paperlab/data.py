from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import requests

from .core import Tick, atomic_json, digest

BASE = "https://api.exchange.coinbase.com"
PRODUCTS = ("BTC-USD", "ETH-USD")


def write_ticks(path, ticks, note):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial")
    tmp.write_text("".join(json.dumps(asdict(t), allow_nan=False) + "\n" for t in ticks))
    tmp.replace(path)
    atomic_json(str(path) + ".manifest.json", {"sha256": digest(path), "rows": len(ticks), "note": note, "created_at": time.time()})


def synthetic(path, count=1200, seed=7):
    rng = np.random.default_rng(seed)
    prices = 60000 * np.exp(np.cumsum(rng.normal(0, .003, count)))
    ticks = [Tick(1700000000 + i * 900, float(p * .9999), float(p * 1.0001), float(rng.lognormal(3, 1))) for i, p in enumerate(prices)]
    write_ticks(path, ticks, "SYNTHETIC fixture: verifies mechanics, provides no evidence of a market edge.")


def historical(path, product="BTC-USD", days=14, granularity=900):
    if product not in PRODUCTS or granularity not in (300, 900, 3600, 21600, 86400) or not 2 <= days <= 365:
        raise ValueError("Unsupported dataset request")
    end = int(time.time() // granularity * granularity)
    start = end - days * 86400
    candles = {}
    for a in range(start, end, 299 * granularity):
        b = min(end, a + 299 * granularity)
        iso = lambda s: datetime.fromtimestamp(s, timezone.utc).isoformat()
        response = requests.get(f"{BASE}/products/{product}/candles", params={"start": iso(a), "end": iso(b), "granularity": granularity}, timeout=30)
        response.raise_for_status()
        for ts, low, high, opening, close, volume in response.json():
            if start <= ts and ts + granularity <= end:
                candles[ts] = Tick(ts + granularity, close * .9999, close * 1.0001, volume, product, "historical_candle_proxy", time.time())
        time.sleep(.2)
    ticks = [candles[k] for k in sorted(candles)]
    write_ticks(path, ticks, "Public closed candles; assumed 2 bps spread. Fills use the NEXT close plus slippage/fees. No historical book, intrabar execution, latency or market impact reconstruction. Current RSS is not backdated.")


def quote(product="BTC-USD"):
    if product not in PRODUCTS:
        raise ValueError("Unsupported product")
    response = requests.get(f"{BASE}/products/{product}/book", params={"level": 1}, timeout=10)
    response.raise_for_status()
    book = response.json()
    now = time.time()
    return Tick(now, float(book["bids"][0][0]), float(book["asks"][0][0]), 0, product, "forward_rest_book", now)
