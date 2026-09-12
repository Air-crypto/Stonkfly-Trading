from dataclasses import asdict, replace
import json
from types import SimpleNamespace
import time

import numpy as np
import pytest

from paperlab.core import Tick
from paperlab.fly_market_study import ARMS, REINFORCEMENT_ARMS, phase, quote_at, seal, seal_followup, signature, validate
from paperlab.universe import Pool, Store


def test_sealed_cohort_does_not_select_on_future_returns(tmp_path):
    store=Store(tmp_path/"archive.db")
    for i in range(180):
        stamp=1700000100+i*60
        store.add([Pool(f"synthetic:pool{k}","synthetic",f"pool{k}",f"token{k}",f"TEST{k}",
                        1 if i<100 else 1+k*10,100000,10000,20,20,stamp-3600,stamp,
                        source="synthetic") for k in range(3)])
    store.db.close()
    first=seal(tmp_path/"archive.db",tmp_path/"plan.json")
    assert first["sha256"]==signature(first["plan"])
    assert first["plan"]["cohort"]==["synthetic:pool0","synthetic:pool1"]
    assert all(t["ts"]<=first["plan"]["start"]+2700 for seq in first["plan"]["series"].values() for t in seq)
    with pytest.raises(ValueError): seal(tmp_path/"archive.db",tmp_path/"plan.json")
    bad=json.loads(json.dumps(first["plan"]));bad["costs"]["fee_bps"]=0
    with pytest.raises(ValueError):validate(bad)
    follow=seal_followup(tmp_path/"archive.db",first,tmp_path/"next.json",phase_steps=2)
    assert follow["plan"]["cohort"]==first["plan"]["cohort"]
    assert follow["plan"]["arms"]==REINFORCEMENT_ARMS
    assert follow["plan"]["start"]==first["plan"]["start"]+2700
    assert follow["plan"]["parent_plan_sha256"]==first["sha256"]
    bad=json.loads(json.dumps(follow["plan"]));bad["start"]-=300
    with pytest.raises(ValueError):validate(bad)
    bad=json.loads(json.dumps(first["plan"]));bad["series"][bad["cohort"][0]][0]["ts"]=bad["start"]+999999
    with pytest.raises(ValueError):validate(bad)


def test_quote_lookup_never_uses_future_or_optimistic_stale_prices():
    ticks=[Tick(1000,1,1.01),Tick(1600,9,9.01)]
    index,t=quote_at(ticks,1200)
    assert index==0 and t.bid==1 and not t.available
    index,t=quote_at(ticks,1600)
    assert index==1 and t.bid==9 and t.available


def test_phase_uses_next_receipt_fills_and_holds_missing_inventory(monkeypatch):
    import paperlab.fly_market_study as study
    seen=[]
    def render(ticks,index,news):
        seen.append(ticks[index].ts)
        return np.zeros((180,320,3),dtype=np.uint8)
    monkeypatch.setattr(study,"frame",render)
    b=SimpleNamespace(weight=np.array([1.]),circuit={"edges":np.array([0])},weights_frozen=False,eta=.001)
    from dataclasses import dataclass
    @dataclass
    class Settings:
        learning: bool=True
    controller=SimpleNamespace(s=Settings(),observe=lambda *a:{"side":"BUY"})
    lab=SimpleNamespace(brain=b,fly=SimpleNamespace(controller=controller))
    ticks=[Tick(1000,1,1.01),Tick(1300,1,1.01),Tick(1600,1,1.01,available=False)]
    result=phase(lab,ticks,1000,2,ARMS["pristine_frozen"],False,time.monotonic()+30)
    assert seen==[1000,1300]
    assert result["rows"][0]["fill"]["status"]=="hold"
    assert result["rows"][1]["fill"]["status"]=="filled"
    assert result["rows"][1]["fill"]["decision_ts"]==1000
    assert result["rows"][1]["fill"]["fill_ts"]==1300
    assert result["rows"][2]["fill"]["reason"]=="unavailable_market"
    assert float(result["rows"][-1]["broker"]["qty"])>0
    assert result["equity"]==float(result["rows"][-1]["broker"]["cash"])
    timeline=study.decision_timeline(result["rows"])
    assert len(timeline)==3  # The missing terminal quote has no neural capture.
    assert timeline[1]["fill_status"]=="filled"
    assert timeline[2]["observation"]=="terminal_mark"
    assert timeline[2]["fill_reason"]=="unavailable_market"
    assert timeline[2]["action"] is None


def test_reinforcement_gate_obeys_closed_loop_reward_without_substituting_actions(monkeypatch):
    import paperlab.fly_market_study as study
    from dataclasses import dataclass
    @dataclass
    class Settings:
        learning: bool=True
    monkeypatch.setattr(study,"frame",lambda *a:np.zeros((180,320,3),dtype=np.uint8))
    b=SimpleNamespace(weight=np.array([1.]),circuit={"edges":np.array([0])},weights_frozen=False,eta=.001)
    states=[]
    def observe(rgb,stimulus):
        states.append((controller.s.learning,b.weights_frozen,stimulus))
        if controller.s.learning: b.weight[0]+=.1
        return {"side":"BUY"}
    controller=SimpleNamespace(s=Settings(),observe=observe)
    lab=SimpleNamespace(brain=b,fly=SimpleNamespace(controller=controller))
    ticks=[Tick(1000+i*300,1,1.01) for i in range(3)]
    result=phase(lab,ticks,1000,2,REINFORCEMENT_ARMS["reinforcement_gated"],True,time.monotonic()+30)
    assert states==[(False,True,"none"),(True,False,"aversive")]
    assert result["rows"][0]["event"]["side"]==result["rows"][1]["event"]["side"]=="BUY"
    assert result["rows"][0]["event"]["weight_delta_l2"]==0
    assert result["rows"][1]["event"]["weight_delta_l2"]>0
