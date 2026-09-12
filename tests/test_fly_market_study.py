from dataclasses import asdict, replace
import json
import os
from pathlib import Path
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
    from paperlab.fly_market_study import RESTORATION_ARMS,RESTORATION_TIMING
    restored=seal_followup(tmp_path/"archive.db",first,tmp_path/"restoration.json",phase_steps=2,protocol="memory-restoration")
    assert restored["plan"]["arms"]==RESTORATION_ARMS and restored["plan"]["schema"]==5
    assert restored["plan"]["restoration_timing"]==RESTORATION_TIMING
    restored_back=seal_followup(tmp_path/"archive.db",restored,tmp_path/"back-frozen.json",phase_steps=2,protocol="frozen-inference")
    assert "restoration_timing" not in restored_back["plan"]
    invalid=json.loads(json.dumps(restored["plan"]));invalid["restoration_timing"]="restore after test"
    with pytest.raises(ValueError):validate(invalid)
    registration=json.loads(Path("reports/fly-market-study-06-preregistration.json").read_text())
    assert RESTORATION_ARMS==registration["arms"] and RESTORATION_TIMING==registration["restoration_timing"]
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


@pytest.mark.skipif(not os.environ.get('FLY_TRACE_DATA'),reason='requires prepared full retained graph')
def test_native_inference_restores_only_declared_inputs(tmp_path):
    from paperlab.fly_trace import TraceLab,synthetic_ticks
    from paperlab.fly_market_study import RESTORATION_ARMS,inference_memory,learned_state,restore_learned,memory_signature
    lab=TraceLab(Path(os.environ['FLY_TRACE_DATA']));b=lab.brain
    pristine=learned_state(b);ticks=synthetic_ticks('fall',5);arm=RESTORATION_ARMS['restore_10704']
    training=phase(lab,ticks,ticks[99].ts,2,arm,True,time.monotonic()+90)
    trained=learned_state(b);saved={k:v.copy() for k,v in trained.items()}
    initial,restoration=inference_memory(b,trained,pristine,arm)
    mask=b.ids[b.post[b.circuit['edges']]]==10704
    assert mask.sum()==restoration['edge_count']==2048
    for k in initial:
        np.testing.assert_array_equal(initial[k][mask],pristine[k][mask])
        np.testing.assert_array_equal(initial[k][~mask],trained[k][~mask])
        np.testing.assert_array_equal(trained[k],saved[k])
    assert memory_signature(trained)==training['final_memory_sha256']
    restore_learned(b,initial)
    result=phase(lab,ticks,ticks[101].ts,2,arm,False,time.monotonic()+90,tmp_path/'inference',restoration=restoration)
    assert result['initial_memory_sha256']==result['final_memory_sha256']==memory_signature(initial)
    view=json.loads((tmp_path/'inference/view.json').read_text())
    assert view['report']['restoration']==restoration
    assert view['report']['config']['restore_post_ids']==['10704']
    assert all(row['event']['stimulus']=='none' and not row['event']['plasticity_enabled'] for row in result['rows'] if row['event'])


def test_market_runner_uses_prepared_memory_and_never_development_state(tmp_path,monkeypatch):
    import paperlab.fly_market_study as study
    class Brain:
        n=3;ids=np.array([10704,11402,12859]);post=np.arange(3);circuit={'edges':np.arange(3)};build={'binary_sha256':'fixture'}
        def reset(self):self.weight=np.ones(3,dtype=np.float32);self.memory_u=np.zeros(3,dtype=np.float32);self.memory_w=np.zeros(3,dtype=np.float32)
    b=Brain();b.reset();lab=SimpleNamespace(brain=b)
    plan=json.loads(Path('reports/fly-market-study-05-plan.json').read_text())['plan']
    plan.update(schema=5,arms=study.RESTORATION_ARMS,restoration_timing=study.RESTORATION_TIMING)
    observed=[]
    def fake_phase(lab,ticks,start,steps,arm,learning,deadline,trace_output=None,restoration=None):
        observed.append((start,arm,study.learned_state(b)))
        if learning:b.weight+=1;b.memory_u+=2;b.memory_w+=3
        if start==plan['start']+900:
            # If the runner accidentally carries development into test, this marker leaks.
            b.weight[:]=999;b.memory_u[:]=888;b.memory_w[:]=777
        return {'equity':250,'return_pct':0,'rows':[],'restoration':restoration}
    monkeypatch.setattr(study,'TraceLab',lambda data:lab);monkeypatch.setattr(study,'phase',fake_phase)
    result=study.run({'plan':plan,'sha256':signature(plan)},tmp_path/'unused',tmp_path/'run')
    assert result['selection']['selected'] is None
    assert sum(start==plan['start'] for start,_,_ in observed)==2*len(plan['cohort'])
    records=json.loads((tmp_path/'run/results.json').read_text())
    for pool in records.values():
        assert pool['restore_10704']['training']['training_compute_source']=='trained_frozen'
        assert pool['restore_10704']['training']['training_compute_reused'] is True
        assert pool['trained_frozen']['training']['training_compute_reused'] is False
    for start,arm,state in observed:
        if start!=plan['start']+1800:continue
        mask=np.isin(b.ids.astype(str),arm.get('restore_post_ids',[]))
        expected=np.where(mask,1,2 if arm['train'] else 1)
        np.testing.assert_array_equal(state['weights'],expected)
        np.testing.assert_array_equal(state['u'],np.where(mask,0,2 if arm['train'] else 0))
        np.testing.assert_array_equal(state['w'],np.where(mask,0,3 if arm['train'] else 0))
    with np.load(tmp_path/'run/plastic-map.npz') as saved:np.testing.assert_array_equal(saved['post_ids'],b.ids)
