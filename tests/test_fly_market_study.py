from dataclasses import asdict, replace
import json
import os
from types import SimpleNamespace
import time

import numpy as np
import pytest

from paperlab.core import Tick
from paperlab.fly_market_study import ARMS, REINFORCEMENT_ARMS, VISUAL_ARMS, INFERENCE_ARMS, phase, quote_at, seal, seal_followup, signature, validate
from paperlab.universe import Pool, Store


def test_sealed_cohort_does_not_select_on_future_returns(tmp_path):
    store=Store(tmp_path/"archive.db")
    for i in range(200):
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
    visual=seal_followup(tmp_path/"archive.db",first,tmp_path/"visual.json",phase_steps=2,protocol="visual")
    assert visual["plan"]["arms"]==VISUAL_ARMS and visual["plan"]["schema"]==3
    assert visual["plan"]["start"]==follow["plan"]["start"]
    frozen=seal_followup(tmp_path/"archive.db",visual,tmp_path/"frozen-inference.json",phase_steps=2,protocol="frozen-inference")
    assert frozen["plan"]["arms"]==INFERENCE_ARMS and frozen["plan"]["schema"]==4
    assert "visual_encoding" not in frozen["plan"]
    bad=json.loads(json.dumps(frozen["plan"]));bad["visual_encoding"]={"view":"fixed_returns"}
    with pytest.raises(ValueError):validate(bad)
    assert frozen["plan"]["start"]==visual["plan"]["start"]+1800
    bad=json.loads(json.dumps(visual["plan"]));bad["visual_encoding"]["log_return_knee"] = .1
    with pytest.raises(ValueError):validate(bad)
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
    b=SimpleNamespace(memory_u=np.zeros(1),memory_w=np.zeros(1),weight=np.array([1.]),circuit={"edges":np.array([0])},weights_frozen=False,eta=.001)
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
    b=SimpleNamespace(memory_u=np.zeros(1),memory_w=np.zeros(1),weight=np.array([1.]),circuit={"edges":np.array([0])},weights_frozen=False,eta=.001)
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


def test_trained_frozen_phase_retains_memory_without_reinforcement(monkeypatch):
    import paperlab.fly_market_study as study
    from dataclasses import dataclass
    @dataclass
    class Settings:
        learning: bool=True
    monkeypatch.setattr(study,'frame',lambda *a:np.zeros((180,320,3),dtype=np.uint8))
    b=SimpleNamespace(weight=np.array([1.2]),circuit={'edges':np.array([0])},memory_u=np.array([.3]),memory_w=np.array([.2]),weights_frozen=False,eta=.001)
    states=[]
    def observe(rgb,stimulus):
        states.append((controller.s.learning,b.weights_frozen,stimulus))
        return {'side':'BUY'}
    controller=SimpleNamespace(s=Settings(),observe=observe)
    lab=SimpleNamespace(brain=b,fly=SimpleNamespace(controller=controller))
    ticks=[Tick(1000+i*300,1,1.01) for i in range(3)]
    result=phase(lab,ticks,1000,2,INFERENCE_ARMS['trained_frozen'],False,time.monotonic()+30)
    assert states==[(False,True,'none')]*2
    assert result['initial_memory_sha256']==result['final_memory_sha256']
    assert b.weight[0]==1.2 and b.memory_u[0]==.3 and b.memory_w[0]==.2
    def broken(rgb,stimulus):
        b.memory_u[0]+=.1
        return {'side':'BUY'}
    controller.observe=broken
    with pytest.raises(AssertionError,match='synaptic memory'):
        phase(lab,ticks,1000,2,INFERENCE_ARMS['trained_frozen'],False,time.monotonic()+30)


@pytest.mark.skipif(not os.environ.get('FLY_TRACE_DATA'),reason='requires prepared full retained graph')
def test_native_market_training_state_survives_frozen_inference(tmp_path):
    from pathlib import Path
    from paperlab.fly_trace import TraceLab,synthetic_ticks
    from paperlab.fly_market_study import learned_state,restore_learned,memory_signature
    lab=TraceLab(Path(os.environ['FLY_TRACE_DATA']))
    ticks=synthetic_ticks('fall',3)
    training=phase(lab,ticks,ticks[99].ts,2,INFERENCE_ARMS['trained_frozen'],True,time.monotonic()+90)
    state=learned_state(lab.brain)
    assert np.linalg.norm(state['weights']-lab.brain.baseline_plastic)>0
    restore_learned(lab.brain,state)
    inference=phase(lab,ticks,ticks[99].ts,2,INFERENCE_ARMS['trained_frozen'],False,time.monotonic()+90,tmp_path/'trace')
    assert training['final_memory_sha256']==inference['initial_memory_sha256']==inference['final_memory_sha256']==memory_signature(state)
    assert all(r['event']['stimulus']=='none' and not r['event']['plasticity_enabled'] for r in inference['rows'] if r['event'])
    with np.load(tmp_path/'trace/initial-memory.npz') as saved:
        assert all(np.array_equal(saved[k],v) for k,v in state.items())
