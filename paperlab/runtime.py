"""Restartable single-writer forward paper runner, using public market data only."""
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import sqlite3
import time

from .core import Broker, Costs, Tick, atomic_json, digest, features
from .data import quote, write_ticks
from .news import FinBERT, News
from .telemetry import emit


def cycle(root, product="BTC-USD", use_fly=False, tick=None, ingest=True, train_daily=False, model=None,
          interval_seconds=900, train_every_seconds=86400, news_every_seconds=900, fly_data=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    # The deployment also limits the entire function to one container. A shared Modal
    # volume is not a distributed SQLite database; no second writer is permitted.
    with (root / "writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if interval_seconds not in (300, 900) or train_every_seconds < 3600 or news_every_seconds < interval_seconds:
            raise ValueError("Unsupported runtime cadence")
        return _cycle(root, product, use_fly, tick, ingest, train_daily, model,
                      interval_seconds, train_every_seconds, news_every_seconds, fly_data)


def _cycle(root, product, use_fly, tick, ingest, train_daily, model,
           interval_seconds, train_every_seconds, news_every_seconds, fly_data):
    costs = Costs()
    db = sqlite3.connect(root / "paper.db")
    db.execute("PRAGMA journal_mode=DELETE")
    db.execute("PRAGMA synchronous=FULL")
    db.execute("CREATE TABLE IF NOT EXISTS ticks (slot INTEGER PRIMARY KEY, payload TEXT)")
    db.execute("CREATE TABLE IF NOT EXISTS state (name TEXT PRIMARY KEY, payload TEXT)")
    db.execute("CREATE TABLE IF NOT EXISTS ledger (slot INTEGER, name TEXT, payload TEXT, PRIMARY KEY(slot,name))")
    news = News(root / "news.db")
    try:
        revision = os.environ.get("PAPERLAB_FINBERT_REVISION", "")
        config = {"product": product, "costs": asdict(costs), "use_fly": use_fly, "schema": "64-market14-news50-v1", "news_encoder": revision or "lexical-v1",
                  "interval_seconds": interval_seconds, "train_every_seconds": train_every_seconds, "news_every_seconds": news_every_seconds}
        stored = db.execute("SELECT payload FROM state WHERE name='config'").fetchone()
        if stored and json.loads(stored[0]) != config:
            raise ValueError("Runtime configuration changed. Use a separate state directory for a new experiment.")
        emit("cycle_started", product=product, interval_seconds=interval_seconds, fly=use_fly)
        last_news = db.execute("SELECT payload FROM state WHERE name='news_checked'").fetchone()
        news_due = ingest and (not last_news or int(time.time() // news_every_seconds) > int(float(last_news[0]) // news_every_seconds))
        news_report = news.ingest(encoder=FinBERT(revision) if revision else None) if news_due else []
        if news_due:
            with db:
                db.execute("INSERT OR REPLACE INTO state VALUES ('news_checked',?)", (str(time.time()),))
        emit("news_scan", status="checked" if news_due else "not_due", feeds=news_report)
        tick = tick or quote(product)
        if tick.product != product or tick.source != "forward_rest_book":
            raise ValueError("Forward runner accepts only matching, observed public book ticks")
        slot = int(tick.ts // interval_seconds)
        previous = db.execute("SELECT max(slot) FROM ticks").fetchone()[0]
        if previous is not None and slot <= previous:
            emit("duplicate_slot", slot=slot)
            return {"status": "duplicate_or_old_slot", "slot": slot}
        # A single transaction publishes quote, model decision, accounting and checkpoint pointer.
        with db:
            db.execute("INSERT OR IGNORE INTO state VALUES ('config',?)", (json.dumps(config),))
            db.execute("INSERT INTO ticks VALUES (?,?)", (slot, json.dumps(asdict(tick))))
            ticks = [Tick(**json.loads(r[0])) for r in db.execute("SELECT payload FROM ticks ORDER BY slot")]
            result = {"status": "collecting" if len(ticks) < 64 else "paper", "slot": slot, "ticks": len(ticks), "news": news_report, "policies": {}}
            result.update(interval_seconds=interval_seconds, train_every_seconds=train_every_seconds,
                          warmup_remaining=max(0, 64-len(ticks)), available_news=len(news.rows(tick.ts)))
            emit("market_observation", slot=slot, observations=len(ticks), bid=tick.bid, ask=tick.ask,
                 warmup_remaining=result["warmup_remaining"], available_news=result["available_news"])
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
                            detail = policy.inspect(features(ticks, len(ticks) - 1, broker, news))
                            action = detail["action"]
                            pending = {"target": [0, costs.max_exposure / 2, costs.max_exposure][action], "decision_ts": time.time()}
                            detail.update(model_sha256=digest(model_path), training_source=meta["source"])
                        else:
                            detail = {"status": "collecting_until_model_available"}
                    else:
                        from .fly import Fly
                        fly = Fly(fly_data or root / "fly-data", checkpoint=state.get("checkpoint"))
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
                    first_event = db.execute("SELECT payload FROM ledger WHERE name=? ORDER BY slot LIMIT 1", (name,)).fetchone()
                    first_ts = json.loads(first_event[0])["ts"] if first_event else tick.ts
                    elapsed_months = max(0, tick.ts - first_ts) / (30 * 86400)
                    event = {"ts": tick.ts, "bid": tick.bid, "ask": tick.ask, "equity": equity, "delta": delta, "fill": fill, "decision": pending, "detail": detail, "broker": broker.state(),
                             "net_after_hosting_scenarios": {str(monthly): equity - costs.capital - elapsed_months * monthly for monthly in (20, 40, 100)}}
                    db.execute("INSERT INTO ledger VALUES (?,?,?)", (slot, name, json.dumps(event)))
                    result["policies"][name] = event
                    emit("paper_decision", trader=name, slot=slot, equity=equity, equity_change=delta,
                         fill=fill, decision=pending, broker=broker.state(), detail=detail)
        atomic_json(root / "latest.json", result)
        # SQLite handles are closed before Modal commits the volume. Training is sequential
        # and candidates use only the already-observed chronological training partition.
        if train_daily and len(ticks) >= 400:
            last = db.execute("SELECT payload FROM state WHERE name='trained_period'").fetchone()
            day = int(tick.ts // train_every_seconds)
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
                    db.execute("INSERT OR REPLACE INTO state VALUES ('trained_period',?)", (str(day),))
                result["paper_model_update"] = {"period": day, "model_sha256": metrics["model_sha256"], "selection": "fixed cadence protocol, no profit-based selection"}
                atomic_json(root / "latest.json", result)
        # Checkpoints are versioned before the transaction; clean only unreferenced older files.
        referenced = {json.loads(r[0]).get("checkpoint") for r in db.execute("SELECT payload FROM state WHERE name='fly'")}
        for p in root.glob("fly-*.npz"):
            if str(p) not in referenced and p.stat().st_mtime < time.time() - 86400:
                p.unlink()
        emit("cycle_completed", status=result["status"], observations=len(ticks),
             warmup_remaining=result["warmup_remaining"], model_update=result.get("paper_model_update"))
        return result
    finally:
        news.db.close()
        db.close()
