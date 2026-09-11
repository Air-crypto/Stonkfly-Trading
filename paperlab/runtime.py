"""Restartable single-writer forward paper runner, using public market data only."""
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
import sqlite3
import time

from .core import Broker, Costs, Tick, atomic_json, digest, features
from .data import quote, write_ticks
from .news import News


def cycle(root, product="BTC-USD", use_fly=False, tick=None, ingest=True, train_daily=False, model=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    # The deployment also limits the entire function to one container. A shared Modal
    # volume is not a distributed SQLite database; no second writer is permitted.
    with (root / "writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _cycle(root, product, use_fly, tick, ingest, train_daily, model)


def _cycle(root, product, use_fly, tick, ingest, train_daily, model):
    costs = Costs()
    db = sqlite3.connect(root / "paper.db")
    db.execute("PRAGMA journal_mode=DELETE")
    db.execute("PRAGMA synchronous=FULL")
    db.execute("CREATE TABLE IF NOT EXISTS ticks (slot INTEGER PRIMARY KEY, payload TEXT)")
    db.execute("CREATE TABLE IF NOT EXISTS state (name TEXT PRIMARY KEY, payload TEXT)")
    db.execute("CREATE TABLE IF NOT EXISTS ledger (slot INTEGER, name TEXT, payload TEXT, PRIMARY KEY(slot,name))")
    news = News(root / "news.db")
    try:
        config = {"product": product, "costs": asdict(costs), "use_fly": use_fly, "schema": "64-market14-news50-v1"}
        stored = db.execute("SELECT payload FROM state WHERE name='config'").fetchone()
        if stored and json.loads(stored[0]) != config:
            raise ValueError("Runtime configuration changed. Use a separate state directory for a new experiment.")
        news_report = news.ingest() if ingest else []
        tick = tick or quote(product)
        if tick.product != product or tick.source != "forward_rest_book":
            raise ValueError("Forward runner accepts only matching, observed public book ticks")
        slot = int(tick.ts // 900)
        previous = db.execute("SELECT max(slot) FROM ticks").fetchone()[0]
        if previous is not None and slot <= previous:
            return {"status": "duplicate_or_old_slot", "slot": slot}
        # A single transaction publishes quote, model decision, accounting and checkpoint pointer.
        with db:
            db.execute("INSERT OR IGNORE INTO state VALUES ('config',?)", (json.dumps(config),))
            db.execute("INSERT INTO ticks VALUES (?,?)", (slot, json.dumps(asdict(tick))))
            ticks = [Tick(**json.loads(r[0])) for r in db.execute("SELECT payload FROM ticks ORDER BY slot")]
            result = {"status": "collecting" if len(ticks) < 64 else "paper", "slot": slot, "ticks": len(ticks), "news": news_report, "policies": {}}
            if len(ticks) >= 64:
                from .compact import load_policy
                for name in (["compact", "fly"] if use_fly else ["compact"]):
                    row = db.execute("SELECT payload FROM state WHERE name=?", (name,)).fetchone()
                    state = json.loads(row[0]) if row else {}
                    broker = Broker(costs, state.get("broker"))
                    pending = state.get("pending")
                    fill = broker.execute(pending["target"], pending["decision_ts"], tick) if pending else {"status": "hold"}
                    equity = broker.equity(tick)
                    delta = equity - state.get("anchor", costs.capital)
                    pending = None
                    detail = {}
                    if name == "compact":
                        model_path = Path(model) if model else root / "active-policy.pt"
                        if model_path.exists():
                            policy, meta = load_policy(model_path)
                            if meta["costs"] != asdict(costs) or meta["product"] != product:
                                raise ValueError("Model costs/product differ from forward experiment")
                            action = policy.action(features(ticks, len(ticks) - 1, broker, news))
                            pending = {"target": [0, costs.max_exposure / 2, costs.max_exposure][action], "decision_ts": time.time()}
                            detail = {"action": action, "model_sha256": digest(model_path), "training_source": meta["source"]}
                        else:
                            detail = {"status": "collecting_until_model_available"}
                    else:
                        from .fly import Fly
                        fly = Fly(root / "fly-data", checkpoint=state.get("checkpoint"))
                        detail = fly.observe(ticks, len(ticks) - 1, news, delta)
                        decision_ts = time.time()
                        if detail["side"] != "HOLD":
                            pending = {"target": costs.max_exposure if detail["side"] == "BUY" else 0, "decision_ts": decision_ts}
                        checkpoint = root / f"fly-{slot}.npz"
                        fly.save(checkpoint)
                        state["checkpoint"] = str(checkpoint)
                        del fly
                    state.update(broker=broker.state(), anchor=equity, pending=pending)
                    db.execute("INSERT OR REPLACE INTO state VALUES (?,?)", (name, json.dumps(state)))
                    event = {"ts": tick.ts, "equity": equity, "delta": delta, "fill": fill, "decision": pending, "detail": detail}
                    db.execute("INSERT INTO ledger VALUES (?,?,?)", (slot, name, json.dumps(event)))
                    result["policies"][name] = event
            atomic_json(root / "latest.json", result)
        # SQLite handles are closed before Modal commits the volume. Training is sequential
        # and candidates use only the already-observed chronological training partition.
        if train_daily and len(ticks) >= 400:
            last = db.execute("SELECT payload FROM state WHERE name='trained_day'").fetchone()
            day = slot // 96
            if not last or int(last[0]) < day:
                from .compact import train
                data_path = root / "forward.jsonl"
                write_ticks(data_path, ticks, "First-observed REST book snapshots; a forward collection, not exchange-guaranteed fills.")
                candidate_dir = root / "candidates" / str(day)
                metrics = train(ticks, news, candidate_dir, costs, steps=8192, seed=7, dataset_sha=digest(data_path))
                # Paper-only automatic replacement, explicitly logged. Never select by test profit.
                active = root / "active-policy.pt"
                tmp = active.with_suffix(".partial")
                tmp.write_bytes((candidate_dir / "policy.pt").read_bytes())
                tmp.replace(active)
                with db:
                    db.execute("INSERT OR REPLACE INTO state VALUES ('trained_day',?)", (str(day),))
                result["paper_model_update"] = {"day": day, "model_sha256": metrics["model_sha256"], "selection": "daily fixed protocol, no profit-based selection"}
                atomic_json(root / "latest.json", result)
        # Checkpoints are versioned before the transaction; clean only unreferenced older files.
        referenced = {json.loads(r[0]).get("checkpoint") for r in db.execute("SELECT payload FROM state WHERE name='fly'")}
        for p in root.glob("fly-*.npz"):
            if str(p) not in referenced and p.stat().st_mtime < time.time() - 86400:
                p.unlink()
        return result
    finally:
        news.db.close()
        db.close()
