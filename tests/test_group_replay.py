from dataclasses import asdict
import json
import sqlite3

import numpy as np
import pytest
import torch

from paperlab.core import Broker, Tick
from paperlab.group_replay import (Actor, COSTS, GROUP_SIZE, action_mask, advantages,
    branch_inputs, cache_fly, clipped_objective, extract, rollout, update)


def episode(prices=(1., 1.2, 1.1, 1.3)):
    return dict(mint='synthetic', frames=[dict(now=100+i*5,
        tick=asdict(Tick(100+i*5, p*.995, p*1.005, received_at=100+i*5)),
        base=[0.]*22) for i,p in enumerate(prices)])


def test_delayed_fill_costs_and_no_invented_terminal_sale():
    e = episode()
    result = rollout(Actor(), e, 1, 'buy_hold')
    assert result['rows'][0]['fill']['status']=='hold'
    fill = result['rows'][1]['fill']
    assert fill['status']=='filled' and float(fill['price']) > 1.2
    assert fill['fill_ts'] > fill['decision_ts']
    assert result['fees_usd'] > 0 and result['fills']==1
    assert float(result['terminal_inventory']['qty']) > 0
    e['frames'][-1]['tick']['available'] = False
    missing = rollout(Actor(), e, 1, 'buy_hold')
    assert missing['fills']==1
    assert missing['pnl_usd'] == pytest.approx(float(missing['terminal_inventory']['cash'])-1000)
    assert not missing['terminal_quote_available']


def test_same_receipt_cannot_fill_and_expired_order_is_not_resurrected():
    e = episode([1.]*6)
    for f in e['frames'][1:]:
        f['tick']['received_at'] = f['tick']['ts'] = 100
    e['frames'][-1]['tick']['received_at'] = e['frames'][-1]['tick']['ts'] = 125
    result = rollout(Actor(), e, 1, 'buy_hold')
    assert result['fills']==0
    assert any(r['fill']['status']=='expired' for r in result['rows'])


def test_branch_account_state_and_prefix_are_independent():
    e = episode()
    actor = Actor()
    cash = rollout(actor, e, 1, 'cash')
    invested = rollout(actor, e, 1, 'buy_hold')
    assert cash['rows'][1]['x'][8:11] != invested['rows'][1]['x'][8:11]
    assert cash['pnl_usd']==0
    repeated = rollout(actor, e, 1, 'buy_hold')
    assert repeated==invested
    changed = episode((1., 1.2, 1.1, 10.))
    later = rollout(actor, changed, 1, 'buy_hold')
    assert later['rows'][:3]==invested['rows'][:3]


def test_pending_and_missing_quote_masks():
    b = Broker(COSTS)
    t = Tick(100, .995, 1.005, received_at=100)
    assert action_mask(b,t,True)==[True,False,False,False]
    assert action_mask(b,Tick(100,.995,1.005,available=False),None)==[True,False,False,False]
    assert action_mask(b,t,None)==[True,False,True,True]
    x = branch_inputs(episode()['frames'][0],b,1)
    assert len(x)==23 and np.isfinite(x).all()


def test_group_normalization_and_clipping_math():
    assert advantages([1.]*12) is None
    with pytest.raises(ValueError):
        advantages([float('nan')])
    a = advantages([0.,1.,2.])
    assert a.mean()==pytest.approx(0) and a.std()==pytest.approx(1)
    logs = torch.tensor([np.log(1.5), np.log(.5)],requires_grad=True)
    loss = clipped_objective(logs,torch.tensor([1.,-1.]))
    assert loss.item()==pytest.approx(-.2)
    loss.backward()
    assert logs.grad.tolist()==[0.,0.]


def test_update_uses_frozen_probabilities_and_reports_real_gradients():
    actor = Actor()
    group = [rollout(actor,episode(),i) for i in range(GROUP_SIZE)]
    saved = json.dumps(group)
    old = [p.detach().clone() for p in actor.parameters()]
    result = update(actor,torch.optim.Adam(actor.parameters(),lr=.001),group)
    assert json.dumps(group)==saved
    assert result[0]['status']=='updated'
    assert result[0]['sequence_ratios']==pytest.approx([1.]*12,abs=1e-6)
    assert result[0]['gradient_norm']>0 and result[0]['weight_delta_l2']>0
    assert any(not torch.equal(p,q) for p,q in zip(old,actor.parameters()))
    assert all(np.isfinite(r['loss']) for r in result)
    ties=[rollout(actor,episode(),i,'cash') for i in range(GROUP_SIZE)]
    before = [p.detach().clone() for p in actor.parameters()]
    assert update(actor,torch.optim.Adam(actor.parameters()),ties)[0]['status']=='skipped_zero_reward_variance'
    assert all(torch.equal(p,q) for p,q in zip(before,actor.parameters()))


def test_native_encoder_refuses_local_execution(tmp_path):
    pytest.importorskip('modal')
    with pytest.raises(RuntimeError,match='must run in Modal'):
        cache_fly([],tmp_path,tmp_path)


def test_extract_freezes_selection_and_retains_unavailable_outcomes(tmp_path, monkeypatch):
    import paperlab.group_replay as module
    source=tmp_path/'source'; source.mkdir()
    (source/'completed.json').write_text('{}')
    (source/'fx.jsonl').write_text(json.dumps(dict(at=100,sol_usd=100)))
    db=sqlite3.connect(source/'events.db')
    db.executescript('CREATE TABLE events(received REAL,mint TEXT,body TEXT);CREATE TABLE health(at REAL,kind TEXT);')
    db.execute('INSERT INTO health VALUES(100,"connected")')
    for now in range(100,171,5):
        for mint in ('a','b'):
            db.execute('INSERT INTO events VALUES(?,?,?)',(now,mint,json.dumps(dict(received=now,mint=mint))))
    db.commit();db.close()
    class FakeFeed:
        def __init__(self,*args):self.tokens={};self.pinned_mints=set()
        def accept(self,event):self.tokens[event['mint']]=event
    monkeypatch.setattr(module,'Feed',FakeFeed)
    def tick(token,now,*args,**kwargs):
        available = now < 120
        return Tick(now,.995,1.005,product=token['mint'],received_at=now,available=available), None if available else 'stale'
    monkeypatch.setattr(module,'tick_for',tick)
    monkeypatch.setattr(module,'inputs',lambda *args:np.zeros(22))
    result,seen,bounds=extract(source,tmp_path/'scratch',excluded={'a'},limit=1)
    assert seen=={'a','b'} and result[0]['mint']=='b'
    assert len(result[0]['frames'])==13
    assert not result[0]['frames'][-1]['tick']['available']
    assert bounds==(100,170)


def test_experiment_writes_complete_auditable_outputs_without_native(tmp_path, monkeypatch):
    import paperlab.group_replay as module
    def synthetic_extract(source, *args, **kwargs):
        e=episode()
        split='train' if source.name==module.SOURCES['train'] else 'test'
        e.update(mint=split,start=100 if split=='train' else 200,end=115 if split=='train' else 215)
        return [e], {'train'}, (100,115) if split=='train' else (200,215)
    monkeypatch.setattr(module,'extract',synthetic_extract)
    monkeypatch.setattr(module,'cache_fly',lambda *args: {'native_observations':0,'synthetic_test':True})
    monkeypatch.setattr(module,'PASSES',1)
    for name in module.SOURCES.values():
        source=tmp_path/'solana-live'/name;source.mkdir(parents=True)
        for filename in ('events.db','fx.jsonl','completed.json'):
            (source/filename).write_text('synthetic')
    out=tmp_path/'out';out.mkdir()
    result=module.experiment(tmp_path,out)
    assert result['training_trajectories']==12 and result['gradient_updates']==2
    assert len(result['evaluation'])==8
    assert result['actor_weight_delta_l2']>0
    assert len((out/'training-trajectories.jsonl').read_text().splitlines())==12
    assert len((out/'evaluation-trajectories.jsonl').read_text().splitlines())==52
    assert (out/'manifest.json').exists() and (out/'actor-trained.pt').exists()
    (out/'completed.json').write_text(json.dumps(result))
    from paperlab.group_replay_audit import audit
    assert audit(out)['verified']


@pytest.mark.parametrize('failure',[False,True])
def test_real_cloud_entrypoint_reserves_budget_and_preserves_failures(tmp_path, monkeypatch, failure):
    pytest.importorskip('modal')
    import group_replay_cloud as cloud
    import paperlab.group_replay as module
    import paperlab.core as core
    from pathlib import Path
    from types import SimpleNamespace
    monkeypatch.setattr(cloud,'Path',lambda p: tmp_path if p=='/state' else Path(p))
    monkeypatch.setattr(cloud,'volume',SimpleNamespace(reload=lambda:None,commit=lambda:None))
    monkeypatch.setattr(cloud.modal,'current_function_call_id',lambda:'test-call')
    monkeypatch.setattr(cloud.modal,'current_input_id',lambda:'test-input')
    monkeypatch.setattr(core,'digest',lambda p:'test-source-hash')
    def fake_experiment(*args):
        if failure:raise RuntimeError('test failure')
        return {'status':'completed','paper_only':True}
    monkeypatch.setattr(module,'experiment',fake_experiment)
    if failure:
        with pytest.raises(RuntimeError,match='test failure'):cloud._run('group-replay-test')
    else:
        result=cloud._run('group-replay-test')
        assert result['budget']['monthly_limit_usd']==85
    root=tmp_path/'group-replay/group-replay-test'
    owner=json.loads((root/'owner.json').read_text())
    assert owner['reservation']['reserve']==pytest.approx(.3191496)
    assert (root/('failed.json' if failure else 'completed.json')).exists()
    if failure:
        budget=json.loads((tmp_path/'budget.json').read_text())
        assert next(iter(budget['months'].values()))==pytest.approx(.3191496)
    with pytest.raises(ValueError,match='Preserve'):cloud._run('group-replay-test')


@pytest.mark.parametrize('corruption',[None,'fee','receipt','cash','feature','reward'])
def test_independent_ledger_audit_rejects_corruption(corruption):
    from paperlab.group_replay_audit import verify_trajectory
    e=episode()
    trajectory=rollout(Actor(),e,1,'buy_hold')
    if corruption=='fee':trajectory['rows'][1]['fill']['fee']='0'
    elif corruption=='receipt':trajectory['rows'][1]['fill']['decision_ts']=105
    elif corruption=='cash':trajectory['rows'][2]['state']['cash']='1000'
    elif corruption=='feature':trajectory['rows'][1]['x'][8]=0
    elif corruption=='reward':trajectory['reward']+=1
    if corruption:
        with pytest.raises(ValueError):verify_trajectory(trajectory,e,asdict(COSTS))
    else:
        result=verify_trajectory(trajectory,e,asdict(COSTS))
        assert result==dict(rows=4,fills=1)
