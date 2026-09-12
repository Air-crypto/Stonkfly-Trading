"""Bounded public DEX discovery. Prices are indicative observations, never quotes.

No token-supplied URL is fetched. Identity is network + contract + pool, not ticker.
The append-only registry retains rejected/missing pools as well as selected ones.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import time

import requests

GT = "https://api.geckoterminal.com/api/v2/"
MAX_TRACKED = 240
MIN_CONTEXT = 64
SAFE_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,150}$")
MAJORS = {"bitcoin", "wrapped-bitcoin", "ethereum", "weth", "solana", "wrapped-solana",
          "usd-coin", "tether", "dai", "binancecoin", "wbnb", "wrapped-avax"}


def clean_id(value):
    if not isinstance(value, str) or not SAFE_ID.fullmatch(value):
        raise ValueError("Invalid provider identifier")
    return value


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) and value >= 0 else 0.
    except (ValueError, TypeError):
        return 0.


@dataclass(frozen=True)
class Pool:
    key: str
    network: str
    address: str
    token: str
    symbol: str
    price: float
    liquidity: float
    volume5: float
    buys5: int
    sells5: int
    created: float
    observed: float
    major: bool = False
    source: str = "geckoterminal_indicative"

    def rejection(self, now=None):
        now = self.observed if now is None else now
        if self.observed > now + 5 or now-self.observed > 180:
            return "stale_observation"
        if self.major:
            return "major_or_stablecoin"
        if self.price <= 0:
            return "missing_price"
        if not self.created or self.created > now or now-self.created < 900:
            return "age_under_15m_or_unknown"
        if self.liquidity < 25000:
            return "liquidity_under_25000"
        if self.volume5 < 1000 or min(self.buys5, self.sells5) < 3:
            return "insufficient_two_sided_activity"
        return None


def parse_pools(payload, observed):
    included = {v["id"]: v.get("attributes", {}) for v in payload.get("included", [])}
    rows = []
    for item in payload.get("data", []):
        try:
            a, r = item["attributes"], item["relationships"]
            network = clean_id(r["network"]["data"]["id"] if "network" in r else item["id"].split("_", 1)[0])
            address = clean_id(a["address"])
            token_id = r["base_token"]["data"]["id"]
            token = clean_id(token_id.removeprefix(network + "_"))
            info = included.get(token_id, {})
            created = datetime.fromisoformat(a["pool_created_at"].replace("Z", "+00:00")).timestamp() if a.get("pool_created_at") else 0
            txns = (a.get("transactions") or {}).get("m5") or {}
            rows.append(Pool(network+":"+address, network, address, token,
                str(info.get("symbol") or a.get("name", "unknown").split(" / ")[0])[:32],
                number(a.get("base_token_price_usd")), number(a.get("reserve_in_usd")),
                number((a.get("volume_usd") or {}).get("m5")), int(number(txns.get("buys"))),
                int(number(txns.get("sells"))), created, observed,
                info.get("coingecko_coin_id") in MAJORS))
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
    return rows


class Store:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=DELETE")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS pools(key TEXT PRIMARY KEY, first_seen REAL, last_seen REAL, payload TEXT);
            CREATE TABLE IF NOT EXISTS observations(key TEXT, slot INTEGER, payload TEXT, reason TEXT, PRIMARY KEY(key,slot));
            CREATE TABLE IF NOT EXISTS launches(id TEXT PRIMARY KEY, mint TEXT, seen REAL, payload TEXT);
            CREATE TABLE IF NOT EXISTS health(id INTEGER PRIMARY KEY, seen REAL, payload TEXT);
            CREATE INDEX IF NOT EXISTS launch_time ON launches(seen);
            CREATE INDEX IF NOT EXISTS obs_slot ON observations(slot);
        """)

    def add(self, pools):
        with self.db:
            for p in pools:
                raw = json.dumps(asdict(p), allow_nan=False)
                self.db.execute("INSERT INTO pools VALUES (?,?,?,?) ON CONFLICT(key) DO UPDATE SET last_seen=excluded.last_seen,payload=excluded.payload",
                                (p.key, p.observed, p.observed, raw))
                # One real response per pool per minute. Do not count repeated requests as extra examples.
                self.db.execute("INSERT OR IGNORE INTO observations VALUES (?,?,?,?)",
                                (p.key, int(p.observed//60), raw, p.rejection()))

    def snapshot(self, path):
        """Publish a closed, consistent SQLite image; never expose a hot journal."""
        if self.db.in_transaction:
            raise RuntimeError("Publish snapshots only after committing discovery updates")
        path = Path(path)
        temporary = path.with_suffix(".partial")
        with sqlite3.connect(temporary) as destination:
            self.db.backup(destination)
            if destination.execute("PRAGMA quick_check").fetchone() != ("ok",):
                raise RuntimeError("Discovery snapshot integrity failure")
        destination.close()
        temporary.replace(path)

    def launch(self, obj, seen):
        if obj.get("txType") not in ("create", "migration", "migrate"):
            return False
        try:
            mint = clean_id(obj.get("mint"))
        except ValueError:
            return False
        # Store only bounded data, not arbitrary URLs/descriptions or trader wallet identities.
        event = {k: str(obj[k])[:150] for k in ("mint", "txType", "signature", "name", "symbol", "pool") if k in obj}
        identity = hashlib.sha256(json.dumps(event, sort_keys=True).encode()).hexdigest()
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO launches VALUES (?,?,?,?)", (identity, mint, seen, json.dumps(event)))
        return True

    def health(self, event, **fields):
        with self.db:
            self.db.execute("INSERT INTO health(seen,payload) VALUES (?,?)", (time.time(), json.dumps({"event":event, **fields})))

    def tracked(self, priorities=(), now=None):
        now = time.time() if now is None else now
        priority = set(priorities)
        # Bound working memory while the durable registry grows indefinitely.
        placeholders=",".join("?" for _ in priority) or "NULL"
        query=f"SELECT first_seen,payload FROM pools WHERE key IN ({placeholders})"
        rows=list(self.db.execute(query,tuple(sorted(priority))))
        rows+=list(self.db.execute("SELECT first_seen,payload FROM pools WHERE last_seen>=? AND json_extract(payload,'$.liquidity')>=25000 ORDER BY last_seen DESC LIMIT 1000",(now-86400,)))
        rows+=list(self.db.execute("SELECT first_seen,payload FROM pools ORDER BY last_seen DESC LIMIT 500"))
        values=list({json.loads(raw)["key"]:(Pool(**json.loads(raw)),first) for first,raw in rows}.values())
        # Holdings first even if missing/rejected, then eligible incumbents; rotate a discovery
        # tranche so a trending-only survivor list cannot consume all observation capacity.
        pinned = [p for p,_ in values if p.key in priority]
        eligible = sorted([p for p,_ in values if p.key not in priority and not p.rejection(now)], key=lambda p:(-p.volume5,p.key))
        other = sorted([p for p,first in values if p.key not in priority and p not in eligible and now-first < 86400],
                       key=lambda p: (int(hashlib.sha256(p.key.encode()).hexdigest()[:8],16)+int(now//60)*7919) % 104729)
        chosen = pinned + eligible[:max(0, MAX_TRACKED-len(pinned)-40)]
        return (chosen + other[:MAX_TRACKED-len(chosen)])[:MAX_TRACKED]


class PublicAPI:
    def __init__(self):
        self.last = 0.
        self.calls = 0

    async def get(self, path, **params):
        # At most 15 public calls/minute, below the documented 30/minute limit.
        await asyncio.sleep(max(0, self.last+4-time.monotonic()))
        self.last = time.monotonic()
        self.calls += 1
        def request():
            r = requests.get(GT+path, params=params, timeout=12, headers={"Accept":"application/json", "User-Agent":"FlyPaperLab/0.2 research"})
            if r.status_code==429:
                self.last=time.monotonic()+60
            r.raise_for_status()
            if len(r.content) > 4_000_000:
                raise ValueError("Oversized provider response")
            return r.json()
        return await asyncio.to_thread(request)


def refresh_requests(watch, priority, minute):
    grouped={}
    for p in watch:
        grouped.setdefault(p.network,[]).append(p.address)
    pinned={p.network for p in watch if p.key in priority}
    regular=sorted(set(grouped)-pinned)
    if regular:
        offset=minute%len(regular)
        regular=regular[offset:]+regular[:offset]
    # The public request cap must not permanently starve alphabetically later chains.
    result=[]
    for network in sorted(pinned)+regular:
        addresses=grouped[network]
        for offset in range(0,len(addresses),30):
            result.append((f"networks/{clean_id(network)}/pools/multi/"+",".join(addresses[offset:offset+30]),{}))
    return result


async def launch_stream(store, deadline):
    import websockets
    backoff = 2
    while time.monotonic() < deadline:
        try:
            # Only free launch/migration streams. Never subscribe to paid token/account trades.
            async with websockets.connect("wss://pumpportal.fun/api/data", open_timeout=10, max_size=65536) as ws:
                store.health("launch_stream_connected")
                for method in ("subscribeNewToken", "subscribeMigration"):
                    await ws.send(json.dumps({"method":method}))
                backoff = 2
                while time.monotonic() < deadline:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), min(20, max(.1,deadline-time.monotonic())))
                        obj = json.loads(raw)
                        if isinstance(obj,dict):
                            if obj.get("errors") or obj.get("error"):
                                raise RuntimeError("Launch provider rejected subscription")
                            store.launch(obj,time.time())
                    except TimeoutError:
                        pass
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            store.health("launch_stream_gap", error=type(exc).__name__, retry_seconds=backoff)
            await asyncio.sleep(min(backoff,max(0,deadline-time.monotonic())))
            backoff = min(30,backoff*2)
    store.health("launch_stream_window_ended")


async def collect_window(root, seconds=240, publish=lambda:None, priorities=lambda:()):
    """One writer; bounded windows make shutdown and budget checks inevitable."""
    from .core import atomic_json
    if not 1 <= seconds <= 285:
        raise ValueError("Collector window must be bounded")
    root=Path(root)
    store=Store(root/"universe.db")
    api=PublicAPI()
    deadline=time.monotonic()+seconds
    stream=asyncio.create_task(launch_stream(store,deadline))
    last_report={}
    try:
        while time.monotonic()<deadline:
            started=time.monotonic()
            errors=[]
            minute=int(time.time()//60)
            # Indexer discovery spans networks; pages rotate through the free API's accessible range.
            requests_to_make=[("networks/new_pools",{"page":1}),
                              ("networks/new_pools",{"page":2+minute%4}),
                              ("networks/trending_pools",{"page":1+minute%5}),
                              ("search/pools",{"query":["PEPE","BONK","WIF","SHIB","DOGE","FLOKI","BRETT","POPCAT","MOG","FARTCOIN"][minute%10]})]
            # Retain and refresh every open/assigned pool irrespective of current ranking.
            priority=set(priorities())
            watch=store.tracked(priority,time.time())
            requests_to_make+=refresh_requests(watch,priority,minute)
            # Limit requests, not incoming launch events. Overflow/gaps are explicitly reported.
            for path,params in requests_to_make[:13]:
                if time.monotonic()+13>=deadline:
                    break
                try:
                    payload=await asyncio.wait_for(api.get(path,include="base_token,quote_token",**params),max(.1,deadline-time.monotonic()-12))
                    store.add(parse_pools(payload,time.time()))
                except Exception as exc:
                    errors.append({"endpoint":path[:120],"error":type(exc).__name__,
                                   "http_status":getattr(getattr(exc,"response",None),"status_code",None)})
            last_report={"status":"collecting", "as_of":time.time(), "tracked_capacity":MAX_TRACKED,
                "registered_pools":store.db.execute("SELECT count(*) FROM pools").fetchone()[0],
                "launch_events":store.db.execute("SELECT count(*) FROM launches").fetchone()[0],
                "observations":store.db.execute("SELECT count(*) FROM observations").fetchone()[0],
                "provider_errors":errors,"request_overflow":max(0,len(requests_to_make)-13),
                "coverage":"Sampled all-network indexer discovery plus received Pump.fun launches; not every memecoin.",
                "prices":"Indicative indexer prices; no executable bid/ask or verified sellability."}
            store.health("collection_snapshot",**last_report)
            atomic_json(root/"latest.json",last_report)
            store.snapshot(root/"universe-snapshot.db")
            import inspect
            published = publish()
            if inspect.isawaitable(published):
                await published
            print(json.dumps({"event":"universe_collected",**last_report}),flush=True)
            await asyncio.sleep(min(max(0,60-(time.monotonic()-started)),max(0,deadline-time.monotonic())))
    finally:
        stream.cancel()
        await asyncio.gather(stream,return_exceptions=True)
        store.health("collector_window_finished")
        last_report.update(as_of=time.time(),launch_events=store.db.execute("SELECT count(*) FROM launches").fetchone()[0])
        atomic_json(root/"latest.json",last_report)
        store.snapshot(root/"universe-snapshot.db")
        store.db.close()
        import inspect
        published = publish()
        if inspect.isawaitable(published):
            await published
    return last_report
