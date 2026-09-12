from dataclasses import asdict, replace
import json
import sqlite3

import numpy as np
import pytest

from paperlab.core import Broker,Tick
from paperlab.universe import Pool,Store,parse_pools,clean_id
from paperlab.multi import cycle,DEX_COSTS,pool_tick,read_archive,select_pool
from paperlab.compact import train,pooled_split
from paperlab.news import News


def pool(i=0,stamp=1700000100,**kwargs):
    p=Pool(f"solana:pool{i}","solana",f"pool{i}",f"token{i}",f"MEME{i}",.01,
           100000,10000,20,20,stamp-3600,stamp)
    return replace(p,**kwargs)


def archive(path,n=100,assets=5):
    s=Store(path)
    for i in range(n):
        s.add([pool(k,1700000100+i*60,price=.01*(1+.0003*i+np.sin(i/5+k)*.01)) for k in range(assets)])
    s.db.close()
    return 1700000100+(n-1)*60


def test_reject_unknown_liquidity_new_stale_and_one_sided():
    p=pool()
    assert p.rejection() is None
    assert replace(p,liquidity=0).rejection()=="liquidity_under_25000"
    assert replace(p,created=p.observed-60).rejection()=="age_under_15m_or_unknown"
    assert replace(p,sells5=0).rejection()=="insufficient_two_sided_activity"
    assert p.rejection(p.observed+181)=="stale_observation"
    assert replace(p,major=True).rejection()=="major_or_stablecoin"
    for malicious in ("../key","a/b","x?token=foo","a,b",""):
        with pytest.raises(ValueError): clean_id(malicious)


def test_pool_parser_uses_contract_identity_not_name():
    raw={"data":[{"id":"solana_pool0","attributes":{"address":"pool0","name":"same / SOL",
          "base_token_price_usd":"0.01","reserve_in_usd":None,"volume_usd":{"m5":"NaN"},
          "transactions":{"m5":{"buys":4,"sells":0}},"pool_created_at":"2026-09-11T00:00:00Z"},
          "relationships":{"base_token":{"data":{"id":"solana_token0"}}}}]}
    p=parse_pools(raw,1789178000)[0]
    assert p.key=="solana:pool0" and p.token=="token0"
    assert p.liquidity==0 and p.volume5==0 and p.rejection() is not None
    raw["data"][0]["attributes"]["address"]="../../etc"
    assert parse_pools(raw,1789178000)==[]


def test_registry_keeps_first_seen_rejections_and_deduplicates(tmp_path):
    s=Store(tmp_path/"u.db")
    p=pool()
    s.add([p,replace(p,price=.02)])
    s.add([replace(p,observed=p.observed+60,liquidity=0)])
    assert s.db.execute("SELECT first_seen FROM pools").fetchone()[0]==p.observed
    assert s.db.execute("SELECT count(*) FROM observations").fetchone()[0]==2
    assert s.db.execute("SELECT count(*) FROM observations WHERE reason IS NOT NULL").fetchone()[0]==1
    assert s.tracked([p.key],p.observed+100000)[0].key==p.key
    obj={"mint":"abc123pump","txType":"create","symbol":"DOGE","uri":"https://bad.invalid","traderPublicKey":"private"}
    s.launch(obj,p.observed)
    s.launch(obj,p.observed+60)
    raw=s.db.execute("SELECT payload FROM launches").fetchone()[0]
    assert "uri" not in raw and "private" not in raw
    assert s.db.execute("SELECT count(*) FROM launches").fetchone()[0]==1
    s.db.close()


def test_published_snapshot_is_closed_and_stable_during_source_writes(tmp_path):
    store=Store(tmp_path/"live.db")
    store.add([pool()])
    snapshot=tmp_path/"universe-snapshot.db"
    store.snapshot(snapshot)
    store.db.execute("UPDATE pools SET first_seen=0")
    with pytest.raises(RuntimeError): store.snapshot(snapshot)
    reader=sqlite3.connect(f"file:{snapshot}?mode=ro",uri=True)
    assert reader.execute("SELECT first_seen FROM pools").fetchone()[0]==pool().observed
    assert reader.execute("PRAGMA integrity_check").fetchone()==("ok",)
    reader.close()
    store.db.rollback()
    store.add([pool(1)])
    store.snapshot(snapshot)
    reader=sqlite3.connect(f"file:{snapshot}?mode=ro",uri=True)
    assert reader.execute("SELECT count(*) FROM pools").fetchone()[0]==2
    reader.close()
    store.db.close()
    assert not snapshot.with_name(snapshot.name+"-journal").exists()


def test_request_budget_rotates_chains_and_keeps_holdings_first():
    from paperlab.universe import refresh_requests
    watch=[replace(pool(i),network=f"chain{i:02d}",key=f"chain{i:02d}:pool{i}") for i in range(20)]
    priority={watch[-1].key}
    observed=set()
    for minute in range(20):
        requests=refresh_requests(watch,priority,minute)[:9]
        assert requests[0][0].startswith("networks/chain19/")
        observed.update(path.split("/")[1] for path,_ in requests)
    assert observed=={p.network for p in watch}


def test_unavailable_inventory_is_not_filled_or_marked_at_stale_price():
    p=pool()
    b=Broker(DEX_COSTS)
    fill=b.execute(.5,p.observed-60,pool_tick(p))
    assert fill["status"]=="filled"
    assert float(fill["price"])>p.price and b.equity(pool_tick(p))<DEX_COSTS.capital
    before=b.state()
    bad=pool_tick(replace(p,observed=p.observed+60,liquidity=0))
    assert b.equity(bad)==float(b.cash)
    assert b.execute(0,p.observed,bad)["reason"]=="unavailable_market"
    assert b.state()==before


class FakeFly:
    def __init__(self,data):
        self.controller=self
        self.count=0
    def observe(self,ticks,i,news,delta):
        self.count+=1
        return {"side":"BUY","count":self.count,"reward":delta}
    def save(self,path):
        path.write_text(json.dumps({"count":self.count}))
    def restore(self,path):
        from pathlib import Path
        self.count=json.loads(Path(path).read_text())["count"]


def test_portfolio_budget_restart_next_tick_and_independent_fly(tmp_path,monkeypatch):
    path=tmp_path/"u.db"
    now=archive(path,n=64)
    monkeypatch.setattr("paperlab.multi.time.time",lambda:now+2)
    root=tmp_path/"run"
    first=cycle(root,path,"unused",ingest=False,now=now,fly_factory=FakeFly)
    assert first["policies"]["fly"]["equity"]==1000
    assert len({e["pool"] for e in first["policies"]["fly"]["sleeves"]})==4
    assert [e["detail"]["count"] for e in first["policies"]["fly"]["sleeves"]]==[1]*4
    assert all(e["fill"]["status"]=="hold" for e in first["policies"]["fly"]["sleeves"])
    assert cycle(root,path,"unused",ingest=False,now=now,fly_factory=FakeFly)["status"]=="duplicate_or_old_slot"
    now+=300
    s=Store(path); s.add([pool(k,now) for k in range(5)]); s.db.close()
    second=cycle(root,path,"unused",ingest=False,now=now,fly_factory=FakeFly)
    events=second["policies"]["fly"]["sleeves"]
    assert all(e["fill"]["status"]=="filled" for e in events)
    assert all(e["fill"]["fill_ts"]>e["fill"]["decision_ts"] for e in events)
    assert all(e["detail"]["count"]==2 for e in events)
    assert second["policies"]["fly"]["equity"]<1000
    assert sum(float(e["broker"]["qty"])*e["observation"]["price"] for e in events)<=500
    # No new data: missing inventory remains recorded; no optimistic stale marking.
    now+=300
    third=cycle(root,path,"unused",ingest=False,now=now,fly_factory=FakeFly)
    assert all(e["unpriced_inventory"] for e in third["policies"]["fly"]["sleeves"])
    assert all(e["fill"]["status"]=="rejected" for e in third["policies"]["fly"]["sleeves"])


def test_cross_pool_duplicate_token_and_future_observations(tmp_path):
    path=tmp_path/"u.db"
    now=archive(path,n=64)
    latest,series=read_archive(path,now)
    used={"solana:pool0"}
    for key in latest:
        latest[key]=replace(latest[key],token="token0")
    assert select_pool(latest,series,used,now) is None
    assert pool(0,now+10).rejection(now)=="stale_observation"


def test_failed_fly_step_rolls_back_ledger_and_state(tmp_path,monkeypatch):
    path=tmp_path/"u.db"
    now=archive(path,n=64)
    class Failure(FakeFly):
        def observe(self,*args): raise RuntimeError("simulated kernel failure")
    with pytest.raises(RuntimeError,match="kernel failure"):
        cycle(tmp_path/"run",path,"unused",ingest=False,now=now,fly_factory=Failure)
    with sqlite3.connect(tmp_path/"run/paper.db") as db:
        assert db.execute("SELECT count(*) FROM ledger").fetchone()[0]==0
        assert not db.execute("SELECT 1 FROM state WHERE name='last_slot'").fetchone()


def test_pooled_ppo_keeps_asset_boundaries_and_global_time_cutoffs(tmp_path):
    path=tmp_path/"u.db"
    now=archive(path,n=110,assets=4)
    _,series=read_archive(path,now)
    cutoff,_=pooled_split(series)
    out=tmp_path/"train"
    result=train(next(iter(series.values())),News(),out,DEX_COSTS,steps=128,series=series,decision_stride=5)
    rows=[json.loads(r) for r in (out/"rollout.jsonl").read_text().splitlines()]
    assert result["product"]=="MULTI-DEX" and result["training_assets"]==4
    assert {r["product"] for r in rows}==set(series)
    assert all(r["ts"]<cutoff and r["decision_ts"]<r["ts"] for r in rows)
    assert result["decision_stride"]==5
    assert max(r["ts"]-r["decision_ts"] for r in rows)==300
    assert result["splits"]["test"]["cash"]["return_pct"]==0
    updates=json.loads((out/"optimizer-updates.json").read_text())
    assert any(u["update_norm"]>0 for u in updates)


def test_archive_excludes_future_and_keeps_disappearance_state(tmp_path):
    path=tmp_path/"u.db"
    now=archive(path,n=64,assets=1)
    s=Store(path)
    s.add([pool(1,now+600)])
    s.db.close()
    latest,series=read_archive(path,now+200)
    assert "solana:pool1" not in latest and "solana:pool1" not in series
    last=series["solana:pool0"][-1]
    assert not last.available and last.ts==now+180
    assert all(t.ts<=now+200 for seq in series.values() for t in seq)


def test_collector_and_trader_budget_cannot_exceed_total_cap(tmp_path):
    from paperlab.budget import reserve,settle
    main=reserve(tmp_path/"main.json",True,limit_override=25)
    collect=reserve(tmp_path/"collect.json",False,limit_override=15,cpu=.125,memory_gib=.25,seconds=300)
    assert main["limit"]+collect["limit"]==40
    assert collect["rate"]==pytest.approx(2*(.0000131*.125+.00000222*.25))
    with pytest.raises(ValueError): reserve(tmp_path/"bad.json",False,cpu=float("nan"))


def test_cloud_writer_guard_blocks_overlap_and_releases_on_error(monkeypatch):
    import cloud
    class Owners:
        def __init__(self): self.keys={}
        def put(self,key,value,skip_if_exists=False):
            if key in self.keys: return False
            self.keys[key]=value; return True
        def pop(self,key): return self.keys.pop(key)
    owners=Owners()
    monkeypatch.setattr(cloud,"writers",owners)
    monkeypatch.setattr(cloud.modal,"current_function_call_id",lambda:"fc-test-owner")
    monkeypatch.setattr(cloud.modal,"current_input_id",lambda:"in-test-owner")
    def nested():
        assert owners.keys["worker"]["call_id"]=="fc-test-owner"
        assert owners.keys["worker"]["input_id"]=="in-test-owner"
        assert cloud.exclusive("worker",lambda:pytest.fail("Second writer ran"))["status"]=="writer_busy"
        return "done"
    assert cloud.exclusive("worker",nested)=="done"
    assert owners.keys=={}
    def fail(): raise RuntimeError("failed input")
    with pytest.raises(RuntimeError): cloud.exclusive("worker",fail)
    assert owners.keys=={}


@pytest.mark.parametrize("has_archive",[False,True])
def test_normal_cloud_entrypoint_with_and_without_archive(monkeypatch,has_archive):
    import cloud
    from pathlib import Path
    class LocalPath(type(Path())):
        def exists(self): return has_archive
    class Volume:
        def reload(self): pass
        def commit(self): pass
    monkeypatch.setattr(cloud,"Path",LocalPath)
    monkeypatch.setattr(cloud,"volume",Volume())
    monkeypatch.setattr(cloud,"discovery_volume",Volume())
    monkeypatch.setenv("PAPERLAB_FLY","1")
    monkeypatch.setenv("PAPERLAB_UNIVERSE","1")
    recorded={}
    def reserve(*a,**kw):
        recorded.update(kw)
        return {"started":0}
    monkeypatch.setattr("paperlab.budget.reserve",reserve)
    monkeypatch.setattr("paperlab.budget.settle",lambda *a:{"estimated_compute_usd":0,"monthly_reserved_usd":0})
    written={}
    monkeypatch.setattr("paperlab.core.atomic_json",lambda p,v:written.update({str(p):v}))
    monkeypatch.setattr("paperlab.multi.cycle",lambda *a,**kw:{"status":"paper_research"})
    result=cloud._worker()
    assert result["status"]==("paper_research" if has_archive else "waiting_for_universe_collector")
    assert "/state/meme-pools-v1/latest.json" in written
    assert recorded["limit_override"]==25 and recorded["memory_gib"]==8
