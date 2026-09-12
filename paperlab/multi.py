"""Isolated multi-pool paper experiment with a fixed total capital allocation.

Four persistent capital sleeves per strategy, never $1,000 per discovered token.
Indexer prices use a declared adverse execution scenario, not executable quotes.
"""
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
import sqlite3
import time

from .core import Broker, Costs, Tick, atomic_json, digest, features
from .news import News, FinBERT
from .telemetry import emit
from .universe import Pool, MIN_CONTEXT

SLEEVES=4
DEX_COSTS=Costs(capital=1000/SLEEVES, fee_bps=125, slippage_bps=100, max_order=25,
                max_exposure=.5, max_spread_bps=150, max_delay=420)
CONFIG={"schema":"meme-pools-v1", "sleeves":SLEEVES,"costs":asdict(DEX_COSTS),
        "decision_seconds":300,"observation_seconds":60,"warmup":MIN_CONTEXT,
        "rotation_seconds":3600,"training_seconds":21600}


def pool_tick(p, now=None):
    # Artificial 100bps spread + 100bps/side slippage + 125bps/side fees.
    # This scenario does not model token taxes, gas, priority, MEV or exact AMM impact.
    return Tick(p.observed, max(1e-12,p.price)*.995, max(1e-12,p.price)*1.005,
                p.volume5,p.key,p.source,p.observed,available=p.rejection(now) is None)


def read_archive(path, now):
    db=sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro",uri=True)
    try:
        latest={key:Pool(**json.loads(raw)) for key,raw in db.execute("SELECT key,payload FROM pools WHERE last_seen<=?",(now,))}
        # Bounded training window, but the durable archive never drops delisted tokens.
        start=int((now-48*3600)//60)
        series={}
        for key,slot,raw in db.execute("SELECT key,slot,payload FROM observations WHERE slot>=? ORDER BY key,slot",(start,)):
            p=Pool(**json.loads(raw))
            if p.observed>now:
                continue
            seq=series.setdefault(key,[])
            if seq and p.observed-seq[-1].ts>180:
                old=seq[-1]
                seq.append(Tick(old.ts+180,old.bid,old.ask,0,old.product,old.source,old.ts+180,False))
            tick=pool_tick(p)
            if not tick.available and seq:
                tick=Tick(tick.ts,seq[-1].bid,seq[-1].ask,0,tick.product,tick.source,tick.received_at,False)
            seq.append(tick)
        # A disappeared pool remains an unsuccessful/unpriced episode in training.
        for key,seq in series.items():
            if now-seq[-1].ts>180:
                old=seq[-1]
                seq.append(Tick(old.ts+180,old.bid,old.ask,0,old.product,old.source,old.ts+180,False))
        return latest,series
    finally:
        db.close()


def select_pool(latest, series, used, now):
    ready=[p for p in latest.values() if p.key not in used and not p.rejection(now)
           and sum(t.available for t in series.get(p.key,[]))>=MIN_CONTEXT and all(t.available for t in series[p.key][-3:])]
    # One pool per contract; a second pool cannot create a second independent opportunity.
    occupied={(latest[k].network,latest[k].token) for k in used if k in latest}
    ready=[p for p in ready if (p.network,p.token) not in occupied]
    ready.sort(key=lambda p:(-p.volume5,p.key))
    return ready[0].key if ready else None


def cycle(root, archive, fly_data, use_fly=True, ingest=True, now=None, fly_factory=None):
    root=Path(root)
    root.mkdir(parents=True,exist_ok=True)
    with (root/"writer.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _cycle(root,archive,fly_data,use_fly,ingest,now,fly_factory)


def _cycle(root,archive,fly_data,use_fly,ingest,now,fly_factory):
    now=time.time() if now is None else now
    slot=int(now//300)
    latest,series=read_archive(archive,now)
    db=sqlite3.connect(root/"paper.db")
    db.execute("PRAGMA journal_mode=DELETE")
    db.execute("PRAGMA synchronous=FULL")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS state(name TEXT PRIMARY KEY,payload TEXT);
        CREATE TABLE IF NOT EXISTS ledger(slot INTEGER,name TEXT,payload TEXT,PRIMARY KEY(slot,name));
    """)
    def get(name,default):
        row=db.execute("SELECT payload FROM state WHERE name=?",(name,)).fetchone()
        return json.loads(row[0]) if row else default
    def put(name,value):
        db.execute("INSERT OR REPLACE INTO state VALUES (?,?)",(name,json.dumps(value,allow_nan=False)))
    import os
    revision=os.environ.get("PAPERLAB_FINBERT_REVISION","")
    config={**CONFIG,"use_fly":use_fly,"news_encoder":revision or "lexical-v1"}
    news=News(root/"news.db")
    try:
        if get("config",config)!=config:
            raise ValueError("Multi-pool config drift; use a new experiment directory")
        if get("last_slot",-1)>=slot:
            return {"status":"duplicate_or_old_slot","slot":slot}
        if ingest and now-get("news_checked",0)>=900:
            report=news.ingest(encoder=FinBERT(revision) if revision else None)
            with db: put("news_checked",now)
            emit("news_scan",feeds=report)
        model=None
        model_path=root/"active-policy.pt"
        if model_path.exists():
            from .compact import load_policy
            model,meta=load_policy(model_path)
            if meta["product"]!="MULTI-DEX" or meta["costs"]!=asdict(DEX_COSTS) or meta.get("decision_stride")!=5:
                raise ValueError("Wrong compact policy for memecoin experiment")
        policies={}
        fly=None
        with db:
            put("config",config)
            names=["compact"]+(["fly"] if use_fly else [])
            # Both strategies start with the same candidates. Subsequent replacement is
            # allowed only when that strategy's sleeve is flat, so capital never resets.
            for name in names:
                lanes=get(name,[{"broker":Broker(DEX_COSTS).state(),"pool":None,"anchor":DEX_COSTS.capital} for _ in range(SLEEVES)])
                used={s["pool"] for s in lanes if s["pool"]}
                events=[]
                for idx,state in enumerate(lanes):
                    broker=Broker(DEX_COSTS,state["broker"])
                    key=state["pool"]
                    if not key or (broker.qty==0 and not state.get("pending") and now-state.get("assigned",0)>=3600):
                        replacement=select_pool(latest,series,used,now)
                        if replacement:
                            if key: used.discard(key)
                            key=replacement
                            used.add(key)
                            # New token gets independent neural dynamics. Retain sleeve money,
                            # cumulative fees, realized losses and halt state through rotations.
                            state.update(pool=key,assigned=now,checkpoint=None,anchor=float(broker.cash),last_quote=0)
                    detail={"status":"waiting_for_64_observations"}
                    fill={"status":"hold"}
                    p=latest.get(key)
                    reason=p.rejection(now) if p else "no_market"
                    equity=float(broker.cash)
                    pending=state.pop("pending",None)
                    if key and not reason:
                        t=pool_tick(p,now)
                        seq=series[key]
                        if p.observed<=state.get("last_quote",0):
                            reason="no_new_observation"
                        else:
                            # A recorded market receipt strictly after the previous decision.
                            fill=broker.execute(pending["target"],pending["ts"],t) if pending else fill
                            equity=broker.equity(t)
                            delta=equity-state.get("anchor",DEX_COSTS.capital)
                            if name=="compact":
                                if model:
                                    detail=model.inspect(features(seq,len(seq)-1,broker,news))
                                    detail["model_sha256"]=digest(model_path)
                                    state["pending"]={"target":[0,.25,.5][detail["action"]],"ts":time.time()}
                                else:
                                    detail={"status":"waiting_for_pooled_ppo","gradient_update":False}
                            else:
                                from .fly import Fly
                                factory=fly_factory or Fly
                                initial=root/"fly-initial.npz"
                                if fly is None:
                                    fly=factory(fly_data)
                                    if not initial.exists():
                                        fly.save(initial)
                                fly.controller.restore(state.get("checkpoint") or str(initial))
                                # Missing-market stress marks never manufacture dopamine rewards
                                # when a quote returns. Reset the reward anchor across that gap.
                                detail=fly.observe(seq,len(seq)-1,news,0 if state.get("unpriced") else delta)
                                if detail["side"]!="HOLD":
                                    state["pending"]={"target":.5 if detail["side"]=="BUY" else 0,"ts":time.time()}
                                path=root/f"fly-{idx}-{slot}.npz"
                                fly.save(path)
                                state["checkpoint"]=str(path)
                            state.update(last_quote=p.observed,anchor=equity,unpriced=False)
                    if reason:
                        detail={"status":"blocked","reason":reason}
                        # Preserve quantities and fees. Never fill, infer, or mark a stale
                        # holding at its last optimistic price. Show last-mark separately.
                        equity=float(broker.cash)
                        state["unpriced"]=bool(broker.qty)
                        if pending: fill={"status":"rejected","reason":reason}
                    state["broker"]=broker.state()
                    event={"sleeve":idx,"pool":key,"symbol":p.symbol if p else None,"equity":equity,
                           "unpriced_inventory":bool(state.get("unpriced")),"broker":broker.state(),
                           "fill":fill,"decision":state.get("pending"),"detail":detail,
                           "observation":asdict(p) if p else None,"context_rows":len(series.get(key,[]))}
                    events.append(event)
                    emit("multi_paper_decision",trader=name,slot=slot,**event)
                put(name,lanes)
                summary={"initial_capital":1000,"equity":sum(e["equity"] for e in events),
                         "fees":sum(float(e["broker"]["fees"]) for e in events),"sleeves":events}
                db.execute("INSERT INTO ledger VALUES (?,?,?)",(slot,name,json.dumps(summary,allow_nan=False)))
                policies[name]=summary
            put("last_slot",slot)
        result={"status":"paper_research", "experiment":"meme-pools-v1","as_of":now,"slot":slot,
                "registered_pools":len(latest),"eligible_now":sum(not p.rejection(now) for p in latest.values()),
                "ready_pools":sum(not p.rejection(now) and sum(t.available for t in series.get(k,[]))>=MIN_CONTEXT
                                  and all(t.available for t in series[k][-3:]) for k,p in latest.items()),"policies":policies,
                "execution":"Adverse indicative-price simulation; not executable DEX quotes.","config":config}
        # Rotate a bounded sample of all archived series, including disappeared/rejected
        # pools. No sorting by realized returns or test performance.
        trainable={k:v for k,v in series.items() if sum(t.available for t in v)>=100}
        result["pooled_training_eligible_assets"]=len(trainable)
        result["training_gate"]="At least four pools with 100 eligible observations each; then fixed six-hour cadence."
        if len(trainable)>=4 and now-get("trained_at",0)>=21600:
            from .compact import train
            selected=dict(sorted(trainable.items())[:240])
            out=root/"candidates"/str(slot)
            metrics=train(next(iter(selected.values())),news,out,DEX_COSTS,steps=8192,series=selected,decision_stride=5)
            tmp=model_path.with_suffix(".partial")
            tmp.write_bytes((out/"policy.pt").read_bytes())
            tmp.replace(model_path)
            with db: put("trained_at",now)
            result["model_update"]={"model_sha256":metrics["model_sha256"],"assets":len(selected),"selection":"fixed cadence; no test-profit selection"}
        atomic_json(root/"latest.json",result)
        keys=sorted({e["pool"] for s in policies.values() for e in s["sleeves"] if e["pool"]})
        atomic_json(root/"watch.json",keys)
        # Versioned checkpoints become visible only when SQLite commits. Keep referenced
        # states and the previous two cycles, so a failed transaction cannot corrupt state.
        refs={s.get("checkpoint") for s in get("fly",[])}
        for p in root.glob("fly-*.npz"):
            if str(p) not in refs and p.stat().st_mtime<time.time()-600:
                p.unlink()
        emit("multi_cycle_completed",registered_pools=len(latest),ready_pools=result["ready_pools"],
             equity={k:v["equity"] for k,v in policies.items()},model_update=result.get("model_update"))
        return result
    finally:
        db.close()
        news.db.close()
