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


def locked_state():
    return dict(cash='912.9515687621246',fees='5.3208408645',halted=False,
                positions={'old':dict(qty='100',basis='99.24406717085564',cash_flow='-87.0484312378754')})


def test_quarantine_preserves_loss_and_caps_new_entries_and_recovery():
    from paperlab.solana_online import update_active
    p=Portfolio(locked_state());before=p.state();t=tick('new')
    assert p.execute('new',.025,99,t,{'new':t})['reason']=='entry_risk_budget'
    created=dict(timestamp=900,received=901);active={'old':created};watched=dict(active);inactive={};retired=set()
    update_active(active,watched,retired,inactive,p,{}, {},1000)
    update_active(active,watched,retired,inactive,p,{}, {},1119)
    assert not p.quarantined
    update_active(active,watched,retired,inactive,p,{}, {},1120)
    assert p.quarantined=={'old'} and p.cash==Decimal(before['cash'])
    assert p.state()['positions']['old']==before['positions']['old']
    assert p.entry_budget()==p.cash-902
    for m in ('a','b','c','d','e'):
        t=tick(m,1121);cash=p.cash;f=p.execute(m,.025,1120,t,{m:t})
        assert p.cash>=902 and cash-p.cash<=Decimal('2.50')
    assert p.entry_budget()<Decimal('1.05') and p.allowed_actions('another')==(0,)
    assert p.positions['old']['qty']==100 and p.positions['old']['basis']==Decimal(before['positions']['old']['basis'])
    restored=Portfolio(p.state());assert restored.quarantined=={'old'} and restored.cash==p.cash
    # A recovered quote restores old acquisition risk; no inventory or loss is erased.
    update_active(active,watched,retired,inactive,p,{'old':{'created':created}},{'old':tick('old',1130)},1130)
    assert not p.quarantined and p.entry_budget()==0
    assert p.execute('old',0,1129,tick('old',1130),{'old':tick('old',1130)})['side']=='SELL'


def test_feed_outage_does_not_quarantine_and_loss_stop_cannot_be_reopened():
    from paperlab.solana_online import update_active
    p=Portfolio(locked_state());created=dict(timestamp=900,received=901)
    active={'old':created};watched=dict(active);inactive={'old':1000}
    update_active(active,watched,set(),inactive,p,{}, {},2000,connected=False)
    assert not p.quarantined and not inactive
    p.quarantined.add('old');p.cash=Decimal(899)
    assert p.execute('new',.025,2000,tick('new',2001),{'new':tick('new',2001)})['reason']=='loss_stop'
    assert p.halted and p.entry_budget()==0


def test_masked_head_cannot_choose_or_bootstrap_blocked_buy():
    h=Readout();x=np.zeros(FEATURES,dtype=np.float32)
    for _ in range(30):assert h.choose(x,allowed_actions=(0,))[0]==0
    q=h.values(x)
    u=h.update(dict(x=x,action=0),x,0,5,allowed_actions=(0,))
    assert u['target']==pytest.approx(.95*q[0],abs=1e-7)


def test_terminal_head_uses_only_final_reward_not_next_account_value():
    h=Readout();x=np.zeros(FEATURES,dtype=np.float32)
    # The next state deliberately has no finite value: terminal targets must not inspect it.
    u=h.update(dict(x=x,action=1),np.full(FEATURES,np.nan,dtype=np.float32),-10,5,terminal=True)
    assert u['target']==pytest.approx(-.4) and u['discount']==0 and u['terminal']
    for reward,elapsed in [(float('nan'),5),(1,-1),(1,float('inf'))]:
        with pytest.raises(ValueError,match='transition'):h.update(dict(x=x,action=0),x,reward,elapsed)


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


def test_snapshot_records_applied_cursor_not_merely_receipt_time(tmp_path):
    f=Feed(tmp_path)
    f.accept(dict(kind='CreateEvent',mint='a',received=999),event_cursor=1)
    before,at,health=f.snapshot_at(lambda:1000.)
    # Both events were received before the snapshot, but the second was still being decoded.
    f.accept(dict(kind='CreateEvent',mint='b',received=999),event_cursor=2)
    assert set(before)=={'a'} and at==1000 and health['event_cursor']==1
    after,_,health2=f.snapshot_at(lambda:1000.)
    assert set(after)=={'a','b'} and health2['event_cursor']==2
    with pytest.raises(ValueError,match='cursor'):
        f.accept(dict(kind='CreateEvent',mint='c',received=999),event_cursor=2)


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


@pytest.mark.parametrize('locked,reject_all,episodic',[(False,False,False),(True,False,False),(True,True,False),(True,False,True)])
def test_cloud_entry_logic_rotates_launches_and_restores_account(tmp_path,monkeypatch,locked,reject_all,episodic):
    clock=Clock();parent=make_parent(tmp_path)
    if locked:
        saved=json.loads((parent/'online-state.json').read_text());saved['portfolio']=locked_state()
        saved['active']={'old':dict(mint='old',timestamp=900,received=901)}
        atomic_json(parent/'online-state.json',saved)
    if episodic:
        saved=json.loads((parent/'online-state.json').read_text())
        (parent/'native.npz').write_bytes(b'fake immutable native weights')
        saved['native_checkpoint']=str(parent/'native.npz')
        atomic_json(parent/'online-state.json',saved)
    calls=[0]
    def choose_actions(self,x,allowed_actions=(0,1)):
        action=int(calls[0]%12<6 and 1 in allowed_actions);calls[0]+=1
        return action,dict(test_policy=True,allowed_actions=list(allowed_actions))
    monkeypatch.setattr(Readout,'choose',choose_actions)
    if reject_all:
        monkeypatch.setattr(Portfolio,'execute',lambda *args:dict(status='rejected',reason='entry_risk_budget'))
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
                    virtual_sol_reserves=int(100_000_000_000*(1+.03*np.sin((clock()-1000)/15))),virtual_token_reserves=1_000_000_000_000,
                    real_sol_reserves=50_000_000_000,sol_amount=1000000) for i in range(6)]
                out[m]=dict(created=created,trades=trades,complete=False)
            return out
    result=run(tmp_path/'run','unused',parent,seconds=300,fly_factory=FakeFly,feed_factory=FakeFeed,
        clock=clock,sleep=clock.sleep,fx_fetch=lambda:100,
        account_mode='fresh_training_episode' if episodic else 'continuous')
    rows=[json.loads(x) for x in (tmp_path/'run/decisions.jsonl').read_text().splitlines()]
    assert result['status']=='completed' and result['active_tokens']==3
    assert {r['neural']['mint'] for r in rows if r['neural']}=={'a','b','c'}
    assert result['new_readout_updates']==result['readout_updates']-42
    if reject_all:
        assert result['new_readout_updates']==0 and result['rejected_order_credit_dropped']>0
    else:
        assert result['new_readout_updates']>0 and result['nonzero_reward_updates']>0 and result['fills']>0
        assert {e['fill']['side'] for r in rows for e in r['executions'] if e['fill']['status']=='filled'}=={'BUY','SELL'}
    if locked and not episodic:
        assert result['portfolio']['positions']['old']==locked_state()['positions']['old']
        assert 'old' in result['portfolio']['quarantined']
    if episodic:
        assert result['account_mode']=='fresh_training_episode' and 'old' not in result['portfolio']['positions']
        assert result['episode']['parent_portfolio']==locked_state()
        assert result['episode_pnl_usd']==pytest.approx(result['equity_stress_usd']-1000)
    assert all(r['neural']['learning_diagnostics']['equity_reward_usd']==0 for r in rows if r['neural'] and r['neural']['switched_token'])
    # Reconstruct all shared-account fills using Decimal rather than calling Portfolio/Broker.
    opening=json.loads((tmp_path/'run/opening.json').read_text())['portfolio']
    cash=Decimal(opening['cash']);qty={m:Decimal(p['qty']) for m,p in opening['positions'].items()};fees=Decimal(opening['fees'])
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
    from paperlab.solana_online_audit import audit
    a=audit(json.loads((tmp_path/'run/opening.json').read_text()),rows)
    assert a['equity_marks_verified']==len(rows) and a['distinct_tokens_with_native_inference']==3
    tampered=deepcopy(rows);tampered[-1]['portfolio']['cash']='9999'
    with pytest.raises(ValueError,match='cash'):audit(json.loads((tmp_path/'run/opening.json').read_text()),tampered)


@pytest.mark.parametrize('condition',['permanent_gap','recovered_gap','last_fill','one_sided_exit'])
def test_reward_credit_reconciles_gaps_terminal_fills_and_separate_exit_quotes(tmp_path,monkeypatch,condition):
    clock=Clock();parent=make_parent(tmp_path)
    saved=json.loads((parent/'online-state.json').read_text())
    (parent/'native.npz').write_bytes(b'fake immutable native weights')
    saved['native_checkpoint']=str(parent/'native.npz');atomic_json(parent/'online-state.json',saved)
    def choose_action(self,x,allowed_actions=(0,1)):
        if condition=='last_fill':action=int(clock()>=1265)
        elif condition=='one_sided_exit':action=int(clock()<1200)
        else:action=1
        return action if action in allowed_actions else 0,dict(test_policy=True)
    monkeypatch.setattr(Readout,'choose',choose_action)
    class Market:
        def __init__(self,root):self.lock=threading.Lock();self.pinned_mints=set()
        def accept(self,created):pass
        def start(self):pass
        def close(self):pass
        def health(self):return dict(status='connected',last_message=clock(),events=20)
        def snapshot(self):
            now=clock()
            gap=now>=1140 and (condition=='permanent_gap' or (condition=='recovered_gap' and now<1180))
            stamp=1139. if gap else now
            one_sided=condition=='one_sided_exit' and now>=1200
            created=dict(kind='CreateEvent',mint='a',timestamp=970,received=971,quote_mint=SOL,
                token_program=TOKEN,is_mayhem_mode=False)
            trades=[dict(received=stamp-6+i,timestamp=stamp-6+i,is_buy=False if one_sided else bool(i%2),
                mint='a',quote_mint=SOL,mayhem_mode=False,signature=str(stamp)+str(i),log_index=i,
                virtual_sol_reserves=100_000_000_000,virtual_token_reserves=1_000_000_000_000,
                real_sol_reserves=50_000_000_000,sol_amount=1000000) for i in range(6)]
            return {'a':dict(created=created,trades=trades,complete=False)}
    root=tmp_path/'run'
    result=run(root,'unused',parent,seconds=300,fly_factory=FakeFly,feed_factory=Market,
        clock=clock,sleep=clock.sleep,fx_fetch=lambda:100,account_mode='fresh_training_episode')
    rows=[json.loads(x) for x in (root/'decisions.jsonl').read_text().splitlines()]
    metrics=[r['neural']['head_training'] for r in rows if r.get('neural') and r['neural'].get('head_training')]
    metrics+=rows[-1]['reward_settlements']
    assert rows[-1]['terminal'] and all(h['terminal'] for h in rows[-1]['reward_settlements'])
    assert result['reward_reconciliation']['residual_usd']==pytest.approx(0,abs=1e-7)
    assert sum(h['reward_usd'] for h in metrics)==pytest.approx(result['episode_pnl_usd'],abs=1e-7)
    assert all(h['reward_usd']<1e-6 for h in metrics)  # A constant market cannot produce gap/recovery profits.
    if condition=='permanent_gap':
        assert result['episode_pnl_usd']<-200
        assert rows[-1]['reward_settlements'][0]['valuation']=='terminal_missing_quote_stress'
        assert rows[-1]['reward_settlements'][0]['reward_usd']<-200
    elif condition=='recovered_gap':
        assert any(h['quote_gap'] and not h['terminal'] for h in metrics)
        assert result['episode_pnl_usd']>-30
    elif condition=='last_fill':
        assert result['fills']==1 and rows[-1]['reward_settlements'][0]['reward_usd']<0
        assert next(e['fill']['fill_ts'] for r in rows for e in r['executions'] if e['fill']['status']=='filled')>1265
    else:
        assert any(e['fill'].get('side')=='SELL' for r in rows if r['observation_at']>=1200 for e in r['executions'])
        assert rows[-2]['quote_diagnostics']['a']['observation_reason'] is None
        assert rows[-2]['quote_diagnostics']['a']['entry_reason']=='one_sided'
    from paperlab.solana_online_audit import audit
    opening=json.loads((root/'opening.json').read_text());a=audit(opening,rows)
    assert a['reward_reconciliation_verified'] and a['new_head_updates']==result['new_readout_updates']
    bad=deepcopy(rows);bad[-1]['reward_settlements'][0]['reward_usd']+=1
    with pytest.raises(ValueError,match='Reward credit'):audit(opening,bad)


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


@pytest.mark.parametrize('mode,delay',[('six_hour',21600),('hourly',3600)])
def test_cadence_change_preserves_pending_and_uses_requested_spacing(tmp_path,mode,delay):
    from paperlab.solana_service import configure_cadence
    c=service(tmp_path,dispatch=lambda *_:'call',poll=lambda _:None,commit=lambda:None,now=1000)
    pending=deepcopy(c['pending']);parent=c['parent']
    changed=configure_cadence(tmp_path,mode,commit=lambda:None,now=1100,start_now=True)
    assert changed['pending']==pending and changed['parent']==parent
    atomic_json(tmp_path/'solana-live'/pending['run_id']/'completed.json',dict(status='completed',paper_only=True))
    c=service(tmp_path,dispatch=lambda *_:pytest.fail('Duplicate dispatch'),
              poll=lambda _:dict(status='completed'),commit=lambda:None,now=2000)
    assert c['next_at']==1000+delay and c['pending'] is None
    changed=configure_cadence(tmp_path,mode,commit=lambda:None,now=2100,start_now=True)
    assert changed['next_at']==2100 and changed['parent']==pending['run_id']
    assert changed['completed_windows']==1 and len(changed['cadence_changes'])==2
    c=service(tmp_path,dispatch=lambda *_:'call2',poll=lambda _:None,commit=lambda:None,now=2100)
    assert c['status']=='dispatched' and c['pending']['call_id']=='call2'


def test_cadence_change_cannot_bypass_budget_or_disabled_state(tmp_path):
    from paperlab.solana_service import configure_cadence
    service(tmp_path,dispatch=lambda *_:'call',poll=lambda _:None,commit=lambda:None,now=1789323000)
    service(tmp_path,dispatch=lambda *_:'unexpected',poll=lambda _:dict(status='budget_stopped'),
            commit=lambda:None,now=1789324000)
    # The next scheduled tick changes the display status but must preserve the pause.
    c=service(tmp_path,dispatch=lambda *_:pytest.fail('Budget pause bypassed'),
              poll=lambda _:None,commit=lambda:None,now=1789324100)
    path=tmp_path/'solana-online/control.json';before=path.read_bytes()
    with pytest.raises(ValueError,match='budget pause'):
        configure_cadence(tmp_path,'six_hour',commit=lambda:None,now=1789324200,start_now=True)
    assert path.read_bytes()==before
    c['enabled']=False;atomic_json(path,c)
    with pytest.raises(ValueError,match='Disabled'):
        configure_cadence(tmp_path,'six_hour',commit=lambda:None,now=1789324300,start_now=True)
    with pytest.raises(ValueError,match='Unknown cadence'):
        configure_cadence(tmp_path,'unknown',commit=lambda:None)


def test_hourly_worker_budget_uses_revised_authorization_and_preserves_spend(tmp_path):
    from paperlab.budget import reserve
    from datetime import datetime,timezone
    path=tmp_path/'budget.json'
    now=datetime(2026,10,1,tzinfo=timezone.utc).timestamp()
    atomic_json(path,{'first_month':'2026-09','months':{'2026-09':6.74,'2026-10':63.5}})
    args=dict(now=now,seconds=3600,startup_seconds=30,memory_gib=8,
              limit_override=85,authorized_monthly_limit=100)
    assert reserve(path,True,**args) is None
    saved=json.loads(path.read_text());assert saved['months']['2026-09']==6.74
    saved['months']['2026-10']=6.74;atomic_json(path,saved)
    r=reserve(path,True,**args)
    assert r['limit']==85 and r['reserve']==pytest.approx(.3191496)
    assert json.loads(path.read_text())['months']['2026-10']==pytest.approx(6.74+r['reserve'])
    for invalid in [101,0,-1,float('nan'),float('inf')]:
        with pytest.raises(ValueError,match='Monthly authorization'):
            reserve(path,True,authorized_monthly_limit=invalid)
    with pytest.raises(ValueError,match='Override'):
        reserve(path,True,authorized_monthly_limit=100,limit_override=101)


def test_unavailable_and_dust_holdings_park_without_resetting_losses():
    from paperlab.solana_online import update_active
    p=Portfolio(state());p.execute('a',.025,99,tick('a'),{'a':tick('a')})
    original=p.state();created={'timestamp':900,'received':901}
    active={'a':created};watched=dict(active);retired=set();inactive={'a':1000}
    snapshots={'b':{'created':created}}
    added,removed=update_active(active,watched,retired,inactive,p,snapshots,{'b':tick('b',1121)},1121)
    assert removed==['a'] and added==['b'] and 'a' in watched and 'a' not in retired
    assert p.state()['positions']['a']==original['positions']['a'] and p.cash==Decimal(original['cash'])
    # Fresh recovery is re-admitted with the original inventory, not a new account.
    snapshots['a']={'created':created}
    added,_=update_active(active,watched,retired,inactive,p,snapshots,{'a':tick('a',1126),'b':tick('b',1126)},1126)
    assert 'a' in added and p.state()['positions']['a']==original['positions']['a']
    # Unsellable sub-dollar dust retains its risk allocation but frees an inference slot.
    dust=Tick(1130,.0001,.000101,product='a',received_at=1130)
    _,removed=update_active(active,watched,retired,inactive,p,snapshots,{'a':dust,'b':tick('b',1130)},1130)
    assert 'a' in removed and 'a' not in active and 'a' in watched
    assert p.state()['positions']['a']==original['positions']['a']


def test_scheduler_pauses_on_paper_loss_stop(tmp_path):
    c=service(tmp_path,dispatch=lambda *_:'call',poll=lambda _:None,commit=lambda:None,now=1000)
    atomic_json(tmp_path/'solana-live'/c['pending']['run_id']/'completed.json',
                dict(status='completed',paper_only=True,portfolio={'halted':True}))
    c=service(tmp_path,dispatch=lambda *_:'must-not-dispatch',poll=lambda _:dict(status='completed'),
              commit=lambda:None,now=2000)
    assert not c['enabled'] and c['status']=='paper_loss_stop_requires_review'


def test_scheduler_does_not_burn_compute_when_no_risk_capacity_remains(tmp_path):
    c=service(tmp_path,dispatch=lambda *_:'call',poll=lambda _:None,commit=lambda:None,now=1000)
    atomic_json(tmp_path/'solana-live'/c['pending']['run_id']/'completed.json',
        dict(status='completed',paper_only=True,training_health='risk_capacity_exhausted',portfolio={'halted':False}))
    c=service(tmp_path,dispatch=lambda *_:pytest.fail('Risk stop bypassed'),poll=lambda _:dict(status='completed'),
              commit=lambda:None,now=2000)
    assert not c['enabled'] and c['status']=='training_risk_capacity_requires_review'


def test_fresh_episode_preserves_models_and_parent_but_resets_account(tmp_path):
    from paperlab.solana_online import opening_state,EPISODE_POLICY
    from paperlab.core import digest
    parent=make_parent(tmp_path)
    s=load_parent(parent);s['portfolio']=locked_state();s['portfolio']['halted']=True
    (parent/'native.npz').write_bytes(b'original weights');s['native_checkpoint']=str(parent/'native.npz')
    atomic_json(parent/'online-state.json',s)
    before={p.name:p.read_bytes() for p in parent.iterdir()}
    fresh=opening_state(parent,'fresh_training_episode','episode-1')
    assert fresh['portfolio']==dict(cash='1000',fees='0',halted=False,positions={},quarantined=[],risk_policy=EPISODE_POLICY)
    assert fresh['active']==fresh['watched']=={} and fresh['retired']==[]
    assert fresh['episode']['parent_portfolio']==s['portfolio']
    assert fresh['episode']['native_checkpoint_sha256']==digest(s['native_checkpoint'])
    assert fresh['head_checkpoint']==s['head_checkpoint']
    head=Readout();head.restore(fresh['head_checkpoint']);assert head.updates==42
    assert all(p.read_bytes()==before[p.name] for p in parent.iterdir())
    assert opening_state(parent,'continuous','x')==s
    with pytest.raises(ValueError,match='Unknown'):opening_state(parent,'typo','x')


def test_training_can_allocate_entire_balance_with_fee_and_liquidity_caps():
    from paperlab.solana_online import EPISODE_POLICY
    p=Portfolio({**state(),'risk_policy':EPISODE_POLICY})
    for mint in ('a','b','c','d'):
        t=tick(mint);before=p.cash
        fill=p.execute(mint,p.target(mint,1,t),99,t,{mint:t},liquidity_notional_usd=1000.)
        assert fill['status']=='filled' and before-p.cash<=250 and p.cash>=0
    assert p.cash<1 and not p.halted  # No inherited $902 floor or $100 allocation cap.
    q=Portfolio({**state(),'risk_policy':EPISODE_POLICY})
    t=tick()
    assert q.execute('a',1.,99,t,{'a':t})['reason']=='insufficient_observed_liquidity'
    f=q.execute('a',1.,99,t,{'a':t},liquidity_notional_usd=10.)
    assert Decimal(f['quantity'])*Decimal(f['price'])<=Decimal('10.000000001')
    assert Portfolio(q.state()).entry_budget()==q.cash


def paused_control(tmp_path):
    run='solana-online-20260914-003217'
    result=dict(status='completed',paper_only=True,training_health='risk_capacity_exhausted',
                portfolio=dict(halted=False),equity_stress_usd=902.60)
    atomic_json(tmp_path/'solana-live'/run/'completed.json',result)
    control=dict(enabled=False,mode='hourly',parent=run,pending=None,next_at=0,completed_windows=5,
                 status='disabled',last_result=result)
    atomic_json(tmp_path/'solana-online/control.json',control)
    return run,control


def test_operator_conversion_archives_account_and_preserves_budget(tmp_path):
    from paperlab.solana_service import enable_training_episodes
    run,control=paused_control(tmp_path)
    atomic_json(tmp_path/'budget.json',dict(months={'2026-09':7.57}))
    before=(tmp_path/'budget.json').read_bytes()
    changed=enable_training_episodes(tmp_path,run,commit=lambda:None,now=1000)
    assert changed['enabled'] and changed['account_mode']=='fresh_training_episode'
    assert changed['completed_windows']==5 and changed['episode_totals']['episodes']==0
    assert json.loads(Path(changed['continuous_account_archive']['control_archive']).read_text())==control
    assert (tmp_path/'budget.json').read_bytes()==before
    with pytest.raises(ValueError,match='already'):enable_training_episodes(tmp_path,run,commit=lambda:None)


@pytest.mark.parametrize('block',['pending','parent','budget','failed'])
def test_episode_conversion_does_not_bypass_other_guards(tmp_path,block):
    from paperlab.solana_service import enable_training_episodes
    run,c=paused_control(tmp_path)
    if block=='pending':c['pending']={'call_id':'running'}
    elif block=='parent':c['parent']='different'
    elif block=='budget':c['budget_paused_until']=2000
    elif block=='failed':c['status']='failed_requires_review'
    path=tmp_path/'solana-online/control.json';atomic_json(path,c);before=path.read_bytes()
    with pytest.raises(ValueError):enable_training_episodes(tmp_path,run,commit=lambda:None,now=1000)
    assert path.read_bytes()==before


def test_episode_losses_survive_reset_and_next_window_keeps_cadence(tmp_path):
    from paperlab.solana_service import enable_training_episodes
    parent,_=paused_control(tmp_path);enable_training_episodes(tmp_path,parent,commit=lambda:None,now=1000)
    calls=[]
    def dispatch(*args):calls.append(args);return 'call'
    c=service(tmp_path,dispatch=dispatch,poll=lambda _:None,commit=lambda:None,now=1000)
    assert calls[0][2]=='fresh_training_episode'
    run=c['pending']['run_id']
    result=dict(status='completed',paper_only=True,account_mode='fresh_training_episode',
        episode=dict(episode_id=run,training_only=True,initial_cash_usd=1000),episode_pnl_usd=-1000.,
        ended=1900,equity_stress_usd=0.,portfolio=dict(cash='0',fees='10',halted=True),
        training_health='risk_capacity_exhausted')
    atomic_json(tmp_path/'solana-live'/run/'completed.json',result)
    c=service(tmp_path,dispatch=dispatch,poll=lambda _:dict(status='completed'),commit=lambda:None,now=2000)
    assert c['enabled'] and c['next_at']==4600
    assert c['episode_totals']['sum_episode_pnl_usd']==-1000
    assert c['episode_totals']['sum_episode_fees_usd']==10
    assert (tmp_path/'solana-online/episode-results'/f'{run}.json').exists()
    c=service(tmp_path,dispatch=dispatch,poll=lambda _:None,commit=lambda:None,now=4600)
    assert c['pending'] and calls[-1][1]==run and c['episode_totals']['episodes']==1


def test_episode_dispatch_requires_matching_completion_mode(tmp_path):
    from paperlab.solana_service import enable_training_episodes
    parent,_=paused_control(tmp_path);enable_training_episodes(tmp_path,parent,commit=lambda:None,now=1000)
    c=service(tmp_path,dispatch=lambda *_:'call',poll=lambda _:None,commit=lambda:None,now=1000)
    atomic_json(tmp_path/'solana-live'/c['pending']['run_id']/'completed.json',dict(status='completed',paper_only=True))
    c=service(tmp_path,dispatch=lambda *_:'no',poll=lambda _:dict(status='completed'),commit=lambda:None,now=2000)
    assert not c['enabled'] and c['status']=='invalid_episode_completion'
