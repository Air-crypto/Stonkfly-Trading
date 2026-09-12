"""Bounded public DEX discovery. Prices are indicative observations, never quotes.

No token-supplied URL is fetched. Identity is network + contract + pool, not ticker.
The append-only registry retains rejected/missing pools as well as selected ones.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime
from email.utils import parsedate_to_datetime
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
REQUEST_INTERVAL = 7.5  # Eight/minute; public reference currently says approximately ten.
REQUESTS_PER_ROUND = 8
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

    def priority_health(self, priorities, now):
        result=[]
        for key in sorted(set(priorities)):
            row=self.db.execute("SELECT payload FROM pools WHERE key=?", (key,)).fetchone()
            p=Pool(**json.loads(row[0])) if row else None
            result.append({"pool":key,"observed":p.observed if p else None,
                           "age_seconds":now-p.observed if p else None,
                           "rejection":p.rejection(now) if p else "missing_pool"})
        return result


class CollectionDeadline(Exception):
    """No request can finish inside this bounded collector window."""


class PublicAPI:
    def __init__(self, state_path):
        self.state_path = Path(state_path)
        self.state = {"schema": 1, "next_request_at": 0., "consecutive_429": 0}
        if self.state_path.exists():
            saved = json.loads(self.state_path.read_text())
            if (saved.get("schema") != 1 or
                not isinstance(saved.get("next_request_at"), (int, float)) or
                not math.isfinite(saved["next_request_at"]) or saved["next_request_at"] < 0 or
                type(saved.get("consecutive_429")) is not int or not 0 <= saved["consecutive_429"] <= 10):
                raise ValueError("Invalid persisted provider cooldown")
            self.state = saved
        self.calls = 0

    def save(self):
        from .core import atomic_json
        atomic_json(self.state_path, self.state)

    async def get(self, path, *, deadline, **params):
        delay = max(0., self.state["next_request_at"] - time.time())
        if time.monotonic() + delay + 13 >= deadline:
            raise CollectionDeadline()
        await asyncio.sleep(delay)
        self.state["next_request_at"] = time.time() + REQUEST_INTERVAL
        self.save()  # A restart cannot reset the request pace or a provider cooldown.
        self.calls += 1
        def request():
            return requests.get(GT+path, params=params, timeout=12,
                headers={"Accept":"application/json;version=20230203", "User-Agent":"FlyPaperLab/0.3 research"})
        try:
            r = await asyncio.wait_for(asyncio.to_thread(request), timeout=13)
        except TimeoutError:
            # The HTTP thread may finish later; no new requests while its result is unknown.
            self.state["next_request_at"] = time.time() + 60
            self.save()
            raise
        if r.status_code == 429:
            self.state["consecutive_429"] = min(10, self.state["consecutive_429"] + 1)
            cooldown = min(900, 60 * 2 ** (self.state["consecutive_429"] - 1))
            retry = r.headers.get("Retry-After", "")
            try:
                retry_seconds = float(retry)
            except ValueError:
                try:
                    retry_seconds = parsedate_to_datetime(retry).timestamp() - time.time()
                except (TypeError, ValueError, OverflowError):
                    retry_seconds = 0
            if math.isfinite(retry_seconds):
                cooldown = max(cooldown, retry_seconds)
            self.state["next_request_at"] = time.time() + cooldown
            self.save()
        r.raise_for_status()
        if len(r.content) > 4_000_000:
            raise ValueError("Oversized provider response")
        payload = r.json()
        self.state["consecutive_429"] = 0
        self.save()
        return payload


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


def collection_requests(watch, priority, minute):
    """Assigned pools first, two rotating discovery calls, then other refreshes."""
    pinned = [p for p in watch if p.key in priority]
    other = [p for p in watch if p.key not in priority]
    discovery = [("networks/new_pools", {"page": 1})]
    rotating = [("networks/new_pools", {"page": 2 + minute % 4}),
                ("networks/trending_pools", {"page": 1 + minute % 5}),
                ("search/pools", {"query": ["PEPE","BONK","WIF","SHIB","DOGE","FLOKI","BRETT","POPCAT","MOG","FARTCOIN"][minute % 10]})]
    discovery.append(rotating[minute % len(rotating)])
    return refresh_requests(pinned, priority, minute) + discovery + refresh_requests(other, (), minute)


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
    api=PublicAPI(root/"provider-rate.json")
    deadline=time.monotonic()+seconds
    stream=asyncio.create_task(launch_stream(store,deadline))
    last_report={}
    window_completed=0
    priority=set()
    try:
        while time.monotonic()<deadline:
            started=time.monotonic()
            errors=[]
            minute=int(time.time()//60)
            # Retain and refresh every open/assigned pool irrespective of current ranking.
            priority=set(priorities())
            watch=store.tracked(priority,time.time())
            requests_to_make=collection_requests(watch,priority,minute)
            attempted_before=api.calls
            completed=0
            exhausted=False
            # Limit requests, not incoming launch events. Overflow/gaps are explicitly reported.
            for path,params in requests_to_make[:REQUESTS_PER_ROUND]:
                if time.monotonic()+13>=deadline:
                    exhausted=True
                    break
                try:
                    payload=await api.get(path,deadline=deadline,include="base_token,quote_token",**params)
                    store.add(parse_pools(payload,time.time()))
                    completed+=1
                    window_completed+=1
                except CollectionDeadline:
                    exhausted=True
                    break
                except Exception as exc:
                    status=getattr(getattr(exc,"response",None),"status_code",None)
                    errors.append({"endpoint":path[:120],"error":type(exc).__name__,
                                   "http_status":status})
                    if status == 429:
                        # Publish the pause now; the next round resumes with assigned pools.
                        # Do not spend recovery capacity on the remaining discovery requests.
                        break
            last_report={"status":"collecting", "as_of":time.time(), "tracked_capacity":MAX_TRACKED,
                "registered_pools":store.db.execute("SELECT count(*) FROM pools").fetchone()[0],
                "launch_events":store.db.execute("SELECT count(*) FROM launches").fetchone()[0],
                "observations":store.db.execute("SELECT count(*) FROM observations").fetchone()[0],
                "provider_errors":errors,"request_overflow":max(0,len(requests_to_make)-REQUESTS_PER_ROUND),
                "requests_attempted":api.calls-attempted_before,"requests_completed":completed,
                "window_requests_attempted":api.calls,"window_requests_completed":window_completed,
                "request_interval_seconds":REQUEST_INTERVAL,"requests_per_round":REQUESTS_PER_ROUND,
                "provider_next_request_at":api.state["next_request_at"],
                "provider_consecutive_429":api.state["consecutive_429"],"window_exhausted":exhausted,
                "priority_quotes":store.priority_health(priority,time.time()),
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
        last_report.update(as_of=time.time(),launch_events=store.db.execute("SELECT count(*) FROM launches").fetchone()[0],
                           priority_quotes=store.priority_health(priority,time.time()))
        atomic_json(root/"latest.json",last_report)
        store.snapshot(root/"universe-snapshot.db")
        store.db.close()
        import inspect
        published = publish()
        if inspect.isawaitable(published):
            await published
    return last_report
