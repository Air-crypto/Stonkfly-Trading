import json
from pathlib import Path
import pytest
from paperlab.checkpoint_eval import register,seal_plan,seal_tape,verify_checkpoint
from paperlab.core import atomic_json


def checkpoint(root,name,started,ended):
    p=root/'solana-live'/name;p.mkdir(parents=True)
    atomic_json(p/'completed.json',dict(status='completed',paper_only=True,account_mode='fresh_training_episode',started=started,ended=ended,readout_updates=10))
    for n in ['fly-final.npz','head-final.pt','events.db','fx.jsonl','decisions.jsonl']:(p/n).write_bytes(b'fixture')
    return p


def test_registry_preserves_original_hash_and_detects_mutation(tmp_path):
    p=checkpoint(tmp_path,'solana-online-20260914-010000',10,20)
    first=register(tmp_path);assert len(first)==1
    verify_checkpoint(first[0]);(p/'head-final.pt').write_bytes(b'changed')
    assert register(tmp_path)==first
    with pytest.raises(ValueError,match='changed'):verify_checkpoint(first[0])


def test_future_tape_excludes_overlapping_window_and_later_weights(tmp_path):
    checkpoint(tmp_path,'solana-online-20260914-010000',10,20)
    plan=seal_plan(tmp_path,register(tmp_path),30)
    checkpoint(tmp_path,'solana-online-20260914-020000',25,40)
    assert seal_tape(tmp_path,plan) is None
    future=checkpoint(tmp_path,'solana-online-20260914-030000',31,45)
    tape=seal_tape(tmp_path,plan);assert tape['id']==future.name
    assert len(plan['checkpoints'])==1
    assert plan['policies'][:3]==['untrained','cash','always_long']
    verify_checkpoint(tape)
    (future/'events.db').write_bytes(b'changed')
    with pytest.raises(ValueError):verify_checkpoint(tape)


def test_frozen_evaluation_delays_fills_and_audits_cash(tmp_path,monkeypatch):
    import sqlite3
    import numpy as np
    from types import SimpleNamespace
    from paperlab.core import Tick,digest
    from paperlab.checkpoint_eval import evaluate
    import paperlab.solana_events as events
    import paperlab.solana_paper as paper
    class Feed:
        def __init__(self,root):self.tokens={}
        def accept(self,e):
            t=self.tokens.setdefault('mint',dict(created={'timestamp':0},trades=[]))
            t['trades'].append(e)
        def snapshot(self):return self.tokens
    monkeypatch.setattr(events,'Feed',Feed)
    monkeypatch.setattr(paper,'tick_for',lambda t,now,fx,seen,**kw:(Tick(t['trades'][-1]['received'],.995,1.005,product='mint',received_at=t['trades'][-1]['received']),None))
    class Brain:
        def __init__(self):self.weight=np.array([1.,2.])
        def reset(self,keep_memory=False):
            if not keep_memory:self.weight[:]=0
    class Fly:
        def __init__(self,*args,**kw):
            assert kw['learning'] is False
            self.controller=SimpleNamespace(brain=Brain())
        def save(self,p):Path(p).write_bytes(b'fake native fixture')
        def observe(self,*args):
            return dict(left_hz=1,right_hz=1,gate_spikes=0,total_spikes=0,KC_spikes=0,reward_spikes=0,aversive_spikes=0,difference_hz=0,learning_diagnostics=dict(weight_delta_l2=0,kc_trace_mean_hz=0))
    db=sqlite3.connect(tmp_path/'events.db');db.execute('create table events(received real, log_index integer, body text)')
    rows=[]
    for i in range(20):
        now=100+i*5;e=dict(received=now-.1,is_buy=i%2==0,sol_amount=1,real_sol_reserves=100_000_000_000)
        db.execute('insert into events values(?,?,?)',(e['received'],i,json.dumps(e)))
        rows.append(dict(at=now,feed=dict(status='connected',last_message=now-.1),neural=dict(mint='mint') if i>=11 else None))
    db.commit();db.close()
    (tmp_path/'fx.jsonl').write_text(json.dumps(dict(at=99,sol_usd=100))+'\n')
    (tmp_path/'decisions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    plan=dict(cutoff=1,policies=['always_long','untrained'],checkpoints=[],scope='test',tape=dict(id='fixture',started=90,files={n:dict(path=str(tmp_path/n),sha256=digest(tmp_path/n)) for n in ['events.db','fx.jsonl','decisions.jsonl']}))
    result=evaluate(plan,'always_long',tmp_path/'long','unused')
    assert result['account_audit']['paper_fills']>0
    assert result['pnl_usd']<0 and result['account_audit']['cash_usd']>=0
    result=evaluate(plan,'untrained',tmp_path/'pristine','unused',fly_factory=Fly)
    assert result['weights_unchanged'] and result['native_observations']>0
    assert result['account_audit']['new_head_updates']==0
