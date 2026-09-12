import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import pytest

from paperlab.core import Tick, atomic_json
from paperlab.fly_market_input import INPUT_ARMS, INPUT_TIMING, INPUT_PROMOTION, development_choice, seal_registered
from paperlab.fly_market_study import validate, signature, phase, learned_state, restore_learned
from paperlab.fly_market_cloud import validate_registration


def plan_fixture():
    p=json.loads(Path('reports/fly-market-study-07-plan.json').read_text())['plan']
    p.update(schema=6,arms=copy.deepcopy(INPUT_ARMS),activity_reset_timing=INPUT_TIMING,promotion_rule=INPUT_PROMOTION,
             recorded_at=p['start']-60,mechanism_report_sha256='a'*64,parent_report_sha256='b'*64)
    p.pop('restoration_timing');return p


def test_registered_input_comparison_has_fixed_arms_and_pretraining_boundary():
    p=plan_fixture();validate(p)
    registration={**p,'end':p['start']+2700}
    validate_registration({'plan':p,'sha256':signature(p)},registration)
    published=json.loads(Path('reports/fly-market-study-08-preregistration.json').read_text())
    assert published['arms']==INPUT_ARMS and published['activity_reset_timing']==INPUT_TIMING and published['promotion_rule']==INPUT_PROMOTION
    assert published['recorded_at']<published['start']


@pytest.mark.parametrize('bad',['late','timing','arms','promotion','length','parent','provenance'])
def test_changed_input_protocol_is_rejected(bad):
    p=plan_fixture()
    if bad=='late':p['recorded_at']=p['start']
    if bad=='timing':p['activity_reset_timing']='reset after a fill'
    if bad=='arms':p['arms']['trained_input_reset']['activity_reset']='voltage'
    if bad=='promotion':p['promotion_rule']='choose from test'
    if bad=='length':p['phase_steps']=2
    if bad=='parent':p['previous_test_end']=p['start']+300
    if bad=='provenance':p['mechanism_report_sha256']='unrecorded'
    with pytest.raises(ValueError):validate(p)


@pytest.mark.parametrize('cash,pristine,trained,reset,expected',[
    (1000,990,995,999,None),(1000,1001,1002,1003,'trained_input_reset'),
    (1000,1001,1004,1003,None),(1000,1004,1002,1003,None),(1000,1001,1002,1002,None)])
def test_candidate_must_beat_cash_and_every_control(cash,pristine,trained,reset,expected):
    equities={'pristine_frozen':pristine,'trained_frozen':trained,'pristine_input_reset':cash,'trained_input_reset':reset}
    assert development_choice(plan_fixture(),equities)==expected
    equities['pristine_input_reset']=reset
    assert development_choice(plan_fixture(),equities) is None


@dataclass
class Settings:
    learning: bool=False


class SmallBrain:
    def __init__(self):
        self.weight=np.ones(1,dtype=np.float32);self.circuit={'edges':np.array([0])};self.memory_u=np.zeros(1);self.memory_w=np.zeros(1)
        self.initial={'g':np.zeros(2,dtype=np.float32),'v':np.full(2,-52.,dtype=np.float32),'last':np.full(2,-1,dtype=np.int64),'counts':np.zeros(2,dtype=np.int32)}
        for k,v in self.initial.items():setattr(self,k,v.copy())
        self.cursor=0;self.sim_ms=0.;self.total_spikes=0


def test_resets_follow_observations_not_missing_market_slots(tmp_path,monkeypatch):
    import paperlab.fly_market_study as study
    monkeypatch.setattr(study,'frame',lambda *a:np.zeros((180,320,3),dtype=np.uint8))
    b=SmallBrain();seen=[]
    def observe(rgb,stimulus):
        seen.append((b.g.copy(),b.v.copy(),b.cursor,stimulus))
        b.g[:]=7;b.v[:]=-71;b.cursor+=5000;b.sim_ms+=500;b.last[:]=b.cursor-1;b.counts[:]=1;b.total_spikes+=2
        return {'side':'BUY','brain_ms':b.sim_ms,'total_spikes':2,'spike_sha256':hashlib.sha256(b.counts.tobytes()).hexdigest()}
    lab=SimpleNamespace(brain=b,fly=SimpleNamespace(controller=SimpleNamespace(s=Settings(),observe=observe)))
    ticks=[Tick(1000+i*300,1,1.01,available=i!=1) for i in range(4)]
    result=phase(lab,ticks,1000,3,INPUT_ARMS['trained_input_reset'],False,time.monotonic()+30,activity_output=tmp_path/'boundaries')
    assert len(seen)==2 and seen[1][2:]==(5000,'none')
    np.testing.assert_array_equal(seen[1][0],[0,0]);np.testing.assert_array_equal(seen[1][1],[-71,-71])
    assert sorted(p.name for p in (tmp_path/'boundaries').glob('*.npz'))==['boundary-01.npz','boundary-02.npz']
    events=[r['event'] for r in result['rows'] if r['event']]
    assert [e['activity_boundary']['mode'] for e in events]==['initial','conductance']
    assert result['rows'][1]['event'] is None and result['rows'][-1]['event'] is None
    assert result['rows'][1]['fill']['reason']=='unavailable_market'
    assert result['rows'][-1]['fill']['status']=='filled'


def test_schema_six_runner_keeps_training_and_phase_states_separate(tmp_path,monkeypatch):
    import paperlab.fly_market_study as study
    class Brain:
        n=2;ids=np.array([1,2]);post=np.array([1]);circuit={'edges':np.array([0])};build={'binary_sha256':'fixture'}
        initial={'g':np.zeros(2)}
        def reset(self):self.g=np.zeros(2);self.weight=np.ones(1);self.memory_u=np.zeros(1);self.memory_w=np.zeros(1)
    b=Brain();b.reset();lab=SimpleNamespace(brain=b);p=plan_fixture();observed=[]
    def fake_phase(lab,ticks,start,steps,arm,learning,deadline,trace_output=None,restoration=None,activity_output=None):
        observed.append((start,arm,learned_state(b),activity_output))
        if learning:b.weight+=1;b.memory_u+=2;b.memory_w+=3
        if start==p['start']+900:b.weight[:]=999
        return {'equity':250,'return_pct':0,'rows':[]}
    monkeypatch.setattr(study,'TraceLab',lambda _:lab);monkeypatch.setattr(study,'phase',fake_phase)
    study.run({'plan':p,'sha256':signature(p)},tmp_path/'graph',tmp_path/'run')
    assert sum(start==p['start'] for start,*_ in observed)==2*len(p['cohort'])
    for start,arm,state,path in observed:
        if start==p['start']:assert path is None
        else:
            assert path is not None;np.testing.assert_array_equal(state['weights'],[2 if arm['train'] else 1])
    records=json.loads((tmp_path/'run/results.json').read_text())
    for pool in records.values():
        assert pool['trained_input_reset']['training']['training_compute_source']=='trained_frozen'
        assert pool['pristine_input_reset']['training']['training_compute_source']=='pristine_frozen'


@pytest.mark.skipif(not os.environ.get('FLY_TRACE_DATA'),reason='requires prepared full graph')
def test_native_market_reset_exports_real_state_and_keeps_trained_memory(tmp_path):
    from paperlab.fly_trace import TraceLab,synthetic_ticks
    from paperlab.fly_market_activity import dynamic_state
    from paperlab.fly_market_activity_audit import audit_boundary,audit_view_boundaries
    lab=TraceLab(Path(os.environ['FLY_TRACE_DATA']));b=lab.brain;ticks=synthetic_ticks('fall',5)
    initial=dynamic_state(b);phase(lab,ticks,ticks[99].ts,2,INPUT_ARMS['trained_frozen'],True,time.monotonic()+90)
    trained=learned_state(b);restore_learned(b,trained)
    result=phase(lab,ticks,ticks[101].ts,3,INPUT_ARMS['trained_input_reset'],False,time.monotonic()+90,tmp_path/'trace',activity_output=tmp_path/'boundaries')
    view=json.loads((tmp_path/'trace/view.json').read_text());audit_view_boundaries(view,tmp_path/'boundaries',b.ids)
    assert len(view['frames'])==3 and view['report']['config']['activity_reset']=='conductance'
    for i,e in enumerate(view['report']['events'],1):audit_boundary(tmp_path/f'boundaries/boundary-{i:02}.npz',e['activity_boundary'],initial,trained,'conductance',i,b.ids)
    assert result['initial_memory_sha256']==result['final_memory_sha256']
    assert [e['brain_ms'] for e in view['report']['events']]==[500,1000,1500]


@pytest.mark.skipif(not os.environ.get('FLY_TRACE_DATA'),reason='requires prepared full graph')
def test_native_four_arm_market_run_passes_independent_audits(tmp_path):
    from dataclasses import replace
    from paperlab.fly_trace import synthetic_ticks
    from paperlab.fly_market_study import run
    from paperlab.fly_market_cloud import build_report
    from paperlab.fly_market_memory_audit import audit as memory_audit
    from paperlab.fly_market_input import audit_market_boundaries
    ticks=[replace(t,ts=t.ts+100,available=i not in (100,103,106)) for i,t in enumerate(synthetic_ticks('fall',9))]
    p=plan_fixture();p.update(start=ticks[99].ts,previous_test_end=ticks[99].ts,recorded_at=ticks[99].ts-60,
        snapshot_end=ticks[-1].ts,cohort=['SYNTHETIC'],series={'SYNTHETIC':[asdict(t) for t in ticks]})
    env={'plan':p,'sha256':signature(p)};root=tmp_path/'run'
    summary=run(env,Path(os.environ['FLY_TRACE_DATA']),root)
    raw=json.loads((root/'results.json').read_text());selection=json.loads((root/'selection.json').read_text())
    report=build_report(env,summary,raw,selection)
    memory_audit(env,report,root);audited=audit_market_boundaries(env,report,root)
    assert sum(len(a[phase]['boundaries']) for a in audited['pools']['SYNTHETIC'].values() for phase in ('development','test'))==16
    for bad in ('quote','input','clock','terminal','learning'):
        changed=copy.deepcopy(report);d=changed['phase_diagnostics']['SYNTHETIC']['trained_input_reset']['test']['decisions'][0]
        if bad=='quote':d['available']=False
        if bad=='input':d['neural']['input_sha256']='c'*64
        if bad=='clock':d['neural']['activity_boundary']['before_clock']['cursor']=1
        if bad=='terminal':d['terminal']=True
        if bad=='learning':d['neural']['plasticity_enabled']=True
        with pytest.raises(ValueError):audit_market_boundaries(env,changed,root)


def test_registered_sealer_keeps_cohort_and_excludes_post_window_data(tmp_path):
    from paperlab.universe import Pool,Store
    from paperlab.fly_market_study import seal
    store=Store(tmp_path/'archive.db');origin=1700000100
    for i in range(1000):
        stamp=origin+i*60
        store.add([Pool(f'synthetic:pool{k}','synthetic',f'pool{k}',f'token{k}',f'TEST{k}',
            1 if i<900 else 99999,100000 if i<800 else 0,10000,20,20,stamp-3600,stamp,source='synthetic') for k in range(2)])
    store.db.close()
    parent=seal(tmp_path/'archive.db',tmp_path/'parent.json')
    start=origin+780*60
    registration={**plan_fixture(),'start':start,'recorded_at':start-60,'end':start+2700,'cohort':parent['plan']['cohort'],
        'previous_test_end':parent['plan']['start']+2700,'parent_plan_sha256':parent['sha256']}
    registration.pop('series');registration.pop('snapshot_sha256');registration.pop('snapshot_end')
    atomic_json(tmp_path/'registration.json',registration)
    result=seal_registered(tmp_path/'archive.db',tmp_path/'parent.json',tmp_path/'registration.json',tmp_path/'next.json')
    assert result['plan']['cohort']==parent['plan']['cohort'] and result['plan']['start']>result['plan']['previous_test_end']
    for series in result['plan']['series'].values():
        assert len(series)==512 and max(t['ts'] for t in series)<=registration['end']
        assert any(not t['available'] for t in series if t['ts']>start)
        assert all(t['bid']<10 for t in series)
    with pytest.raises(ValueError,match='overwrite'):seal_registered(tmp_path/'archive.db',tmp_path/'parent.json',tmp_path/'registration.json',tmp_path/'next.json')
    registration['end']=origin+2000*60;atomic_json(tmp_path/'incomplete.json',registration)
    with pytest.raises(ValueError,match='endpoint'):seal_registered(tmp_path/'archive.db',tmp_path/'parent.json',tmp_path/'incomplete.json',tmp_path/'missing.json')
