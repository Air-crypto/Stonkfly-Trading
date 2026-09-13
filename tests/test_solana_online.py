from collections import deque
from decimal import Decimal
import json
from pathlib import Path
import threading
from copy import deepcopy

import numpy as np
import pytest
from paperlab.core import Tick, atomic_json
from paperlab.solana_online import Portfolio, choose, load_parent, run
from paperlab.solana_service import service
from paperlab.solana_paper import Readout, FEATURES
from paperlab.solana_events import Feed, SOL, TOKEN


def state():
    return dict(cash='1000', fees='0', positions={}, halted=False)


def tick(m='a', t=100):
    return Tick(t, .995, 1.005, product=m, received_at=t)


def test_single_cash_account_and_cashflow_rewards():
    p = Portfolio(state()); a = tick('a'); b = tick('b')
    assert p.execute('a',.025,99,a,{'a':a})['status']=='filled'
    before=p.cash; reward_a=p.contribution('a',a)
    assert p.execute('b',.025,99,b,{'a':a,'b':b})['status']=='filled'
    assert p.cash<before<1000
    assert p.contribution('a',a)==reward_a  # Another token cannot create this token's reward.
    assert sum(x['cash_flow'] for x in p.positions.values()) == p.cash-1000
    assert Portfolio(p.state()).state()==p.state()


def test_missing_position_keeps_inventory_and_risk_and_blocks_new_risk():
    p=Portfolio(state())
    for m in ('a','b','c','d'):
        t=tick(m);p.execute(m,.025,99,t,{k:tick(k) for k in ('a','b','c','d')})
    quantities={m:x['qty'] for m,x in p.positions.items()}
    before=p.state();t=tick('e')
    f=p.execute('e',.025,99,t,{'e':t})
    assert f['status']=='rejected' and f['reason']=='aggregate_acquisition_cost_cap'
    assert p.cash==Decimal(before['cash'])
    assert all(p.positions[m]['qty']==q for m,q in quantities.items())
    assert p.equity({})==float(p.cash)


def test_later_fill_and_fee_accounting_independent_of_broker():
    p=Portfolio(state());t=tick()
    assert p.execute('a',.025,100,t,{'a':t})['reason']=='not_after_decision'
    f=p.execute('a',.025,99,t,{'a':t});q=Decimal(f['quantity']);px=Decimal(f['price']);fee=Decimal(f['fee'])
    assert p.cash==1000-q*px-fee and p.positions['a']['qty']==q
    t2=tick(t=105);f=p.execute('a',0,104,t2,{'a':t2})
    assert f['side']=='SELL' and p.positions['a']['basis']<q*px+fee
    assert sum(x['cash_flow'] for x in p.positions.values())==p.cash-1000


def test_multi_pin_eviction():
    f=Feed('/tmp',max_tokens=3)
    for m in ('a','b','c'):f.accept(dict(kind='CreateEvent',mint=m,received=ord(m)))
    f.pinned_mints={'a','b'};f.accept(dict(kind='CreateEvent',mint='d',received=104))
    assert set(f.snapshot())=={'a','b','d'}


def test_burst_rotation_is_fair():
    current=None;burst=0;visits={};seen=[]
    for i in range(12):
        current,burst=choose(['a','b','c'],current,burst,visits);visits[current]=i+1;seen.append(current)
    assert seen==list('aaabbbcccaaa')


def make_parent(tmp_path):
    p=tmp_path/'parent';p.mkdir();h=Readout();h.updates=42;h.save(p/'head.pt')
    s=dict(portfolio=state(),active={},retired=[],native_checkpoint='native-cloud-only.npz',
           head_checkpoint=str(p/'head.pt'),native_observations_total=0)
    atomic_json(p/'online-state.json',s);atomic_json(p/'completed.json',dict(status='completed',paper_only=True))
    return p


class Clock:
    def __init__(self):self.now=1000.
    def __call__(self):return self.now
    def sleep(self,n):self.now+=n


class FakeBrain:
    def reset(self,keep_memory):assert keep_memory


class FakeFly:
    def __init__(self,*args,**kwargs):self.controller=type('Controller',(),{'brain':FakeBrain()})();self.weights=0
    def observe(self,ticks,i,news,delta,path):
        self.weights+=1
        return dict(left_hz=1.,right_hz=2.,gate_spikes=0,total_spikes=3,KC_spikes=2,reward_spikes=1,
            aversive_spikes=0,difference_hz=1,learning_diagnostics=dict(weight_delta_l2=.1,
            kc_trace_mean_hz=1,equity_reward_usd=delta),compute_seconds=.1)
    def save(self,path):Path(path).write_bytes(b'fake test checkpoint')


def test_cloud_entry_logic_rotates_launches_and_restores_account(tmp_path):
    clock=Clock();parent=make_parent(tmp_path)
    class FakeFeed:
        def __init__(self,root):self.lock=threading.Lock();self.pinned_mints=set();self.accepted={}
        def accept(self,created):self.accepted[created['mint']]=created
        def start(self):pass
        def close(self):pass
        def health(self):return dict(status='connected',last_message=clock(),events=20)
        def snapshot(self):
            out={}
            for m in ('a','b','c'):
                created=dict(kind='CreateEvent',mint=m,timestamp=970,received=971,quote_mint=SOL,
                    token_program=TOKEN,is_mayhem_mode=False)
                trades=[dict(received=clock()-6+i,timestamp=clock()-6+i,is_buy=bool(i%2),
                    mint=m,quote_mint=SOL,mayhem_mode=False,signature=str(clock())+m+str(i),log_index=i,
                    virtual_sol_reserves=100_000_000_000,virtual_token_reserves=1_000_000_000_000,
                    real_sol_reserves=50_000_000_000,sol_amount=1000000) for i in range(6)]
                out[m]=dict(created=created,trades=trades,complete=False)
            return out
    result=run(tmp_path/'run','unused',parent,seconds=150,fly_factory=FakeFly,feed_factory=FakeFeed,
        clock=clock,sleep=clock.sleep,fx_fetch=lambda:100)
    rows=[json.loads(x) for x in (tmp_path/'run/decisions.jsonl').read_text().splitlines()]
    assert result['status']=='completed' and result['active_tokens']==3
    assert {r['neural']['mint'] for r in rows if r['neural']}=={'a','b','c'}
    assert result['readout_updates']>42 and result['new_readout_updates']==result['readout_updates']-42
    assert all(r['neural']['learning_diagnostics']['equity_reward_usd']==0 for r in rows if r['neural'] and r['neural']['switched_token'])
    # Reconstruct all shared-account fills using Decimal rather than calling Portfolio/Broker.
    cash=Decimal(1000);qty={};fees=Decimal(0)
    for r in rows:
        for e in r['executions']:
            f=e['fill'];m=e['mint']
            if f['status']!='filled':continue
            assert 0<f['fill_ts']-f['decision_ts']<=15
            n=Decimal(f['quantity']);px=Decimal(f['price']);fee=Decimal(f['fee']);buy=f['side']=='BUY'
            cash+=-n*px-fee if buy else n*px-fee;qty[m]=qty.get(m,Decimal(0))+(n if buy else -n);fees+=fee
        assert abs(cash-Decimal(r['portfolio']['cash']))<Decimal('1e-20')
        assert fees==Decimal(r['portfolio']['fees'])
        for m,n in qty.items():assert n==Decimal(r['portfolio']['positions'][m]['qty'])
    atomic_json(tmp_path/'run/completed.json',result)
    restored=load_parent(tmp_path/'run');assert restored['portfolio']==result['portfolio']


def test_scheduler_single_flight_completion_pacing_and_failure(tmp_path):
    calls=[];outcomes={}
    def dispatch(run,parent):calls.append((run,parent));return 'call'+str(len(calls))
    def poll(call):return outcomes.get(call)
    def advance(now):return service(tmp_path,dispatch=dispatch,poll=poll,commit=lambda:None,now=now)
    a=advance(1000);assert a['status']=='dispatched' and len(calls)==1
    assert advance(1100)['status']=='running_or_queued' and len(calls)==1
    run_id=calls[0][0];atomic_json(tmp_path/'solana-live'/run_id/'completed.json',dict(status='completed',paper_only=True))
    outcomes['call1']=dict(status='completed')
    b=advance(2000);assert b['status']=='waiting_for_budget_paced_window' and b['parent']==run_id
    c=advance(44200);assert c['status']=='dispatched' and len(calls)==2
    outcomes['call2']=dict(status='failed')
    assert advance(44300)['enabled'] is False
    assert advance(44400)['status']=='disabled' and len(calls)==2


def test_dispatch_ambiguity_does_not_retry(tmp_path):
    def broken(*args):raise ConnectionError('unknown remote submission result')
    with pytest.raises(ConnectionError):service(tmp_path,dispatch=broken,poll=lambda _:None,commit=lambda:None,now=1000)
    result=service(tmp_path,dispatch=broken,poll=lambda _:None,commit=lambda:None,now=1100)
    assert result['status']=='ambiguous_dispatch_requires_review' and not result['enabled']


def test_budget_pause_preserves_parent(tmp_path):
    s=service(tmp_path,dispatch=lambda *_:'call',poll=lambda _:None,commit=lambda:None,now=1789323000)
    p=s['parent']
    s=service(tmp_path,dispatch=lambda *_:'unexpected',poll=lambda _:dict(status='budget_stopped'),commit=lambda:None,now=1789324000)
    assert s['status']=='budget_paused_until_next_month' and s['parent']==p and s['pending'] is None
    assert s['next_at']>1789324000
