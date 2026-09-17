import json
from copy import deepcopy
from pathlib import Path
import threading
from unittest.mock import Mock

import numpy as np
import pytest
from paperlab.core import atomic_json, digest
from paperlab.solana_events import SOL, TOKEN
from paperlab.solana_paper import Readout
from frozen_forward import run, initial_state, validate_state, audit_fills, verify_frozen
from forward_service import fund, coordinate, CHECKPOINT, WORKER_RESERVATION

class Clock:
    def __init__(self):self.now=1000.
    def __call__(self):return self.now
    def sleep(self,n):self.now+=n

class Brain:
    def __init__(self):self.weight=np.array([1.,2.])
    def reset(self,keep_memory):pass

class Fly:
    def __init__(self,*args,learning,**kwargs):
        assert not learning
        self.controller=type('Controller',(),{'brain':Brain()})()
    def observe(self,ticks,i,news,delta):
        return dict(left_hz=1.,right_hz=2.,gate_spikes=0,total_spikes=3,KC_spikes=2,reward_spikes=1,
            aversive_spikes=0,difference_hz=1,learning_diagnostics=dict(weight_delta_l2=0.,
            kc_trace_mean_hz=1,equity_reward_usd=delta),compute_seconds=.1)

@pytest.fixture
def fixture(tmp_path):
    clock=Clock()
    native=tmp_path/'native.npz';native.write_bytes(b'fake native checkpoint')
    head=tmp_path/'head.pt';h=Readout();h.updates=4853;h.save(head)
    manifest=dict(checkpoint=CHECKPOINT,ends_at=10000,
                  native=dict(path=str(native),sha256=digest(native)),head=dict(path=str(head),sha256=digest(head)))
    class Market:
        def __init__(self,root):self.lock=threading.RLock();self.pinned_mints=set()
        def accept(self,event):pass
        def start(self):pass
        def close(self):pass
        def health(self):return dict(status='connected',last_message=clock(),event_cursor=int(clock()))
        def snapshot(self):
            return {m:dict(created=dict(kind='CreateEvent',mint=m,timestamp=970,received=971,quote_mint=SOL,
                          token_program=TOKEN,is_mayhem_mode=False),complete=False,
                trades=[dict(received=clock()-1,timestamp=clock()-1,is_buy=True,mint=m,quote_mint=SOL,
                    mayhem_mode=False,signature=str(clock())+m,log_index=0,
                    virtual_sol_reserves=int(100_000_000_000*(1+.03*np.sin((clock()-1000)/15))),
                    virtual_token_reserves=1_000_000_000_000,real_sol_reserves=50_000_000_000,sol_amount=1000000)])
                    for m in ('a','b')}
    def execute(name,saved,head_factory=Readout,fly_factory=Fly):
        return run(tmp_path/name,'unused',manifest,saved,seconds=180,clock=clock,sleep=clock.sleep,
            feed_factory=Market,fly_factory=fly_factory,fx_fetch=lambda symbol:100 if symbol=='SOL' else 1,
            head_factory=head_factory)
    return tmp_path,clock,manifest,execute


def test_frozen_forward_buys_sells_audits_and_carries_account(fixture,monkeypatch):
    root,clock,manifest,execute=fixture
    calls=[0]
    def values(self,x):
        calls[0]+=1
        return np.array([0.,1.]) if calls[0]%12<6 else np.array([1.,0.])
    monkeypatch.setattr(Readout,'values',values)
    monkeypatch.setattr(Readout,'update',lambda *a,**kw:pytest.fail('Must not train'))
    result=execute('one',initial_state())
    assert result['weights_unchanged'] and result['new_backprop_updates']==0
    assert result['inherited_backprop_updates']==4853
    rows=[json.loads(x) for x in (root/'one/decisions.jsonl').read_text().splitlines()]
    assert {e['fill']['side'] for r in rows for e in r['executions']['trained'] if e['fill']['status']=='filled'}=={'BUY','SELL'}
    saved=json.loads((root/'one/continuation.json').read_text())
    assert float(saved['portfolios']['trained']['fees'])>0
    cash=saved['portfolios']['trained']['cash'];clock.sleep(60)
    execute('two',saved)
    opening=json.loads((root/'two/opening.json').read_text())
    assert opening['portfolios']==saved['portfolios']
    assert opening['portfolios']['trained']['cash']==cash!='1000'
    assert json.loads((root/'two/started.json').read_text())['gap_seconds']>=60
    assert manifest['head']['sha256']==digest(manifest['head']['path'])


def test_indefinite_long_does_not_reset_holdings_or_fees(fixture,monkeypatch):
    root,clock,manifest,execute=fixture
    monkeypatch.setattr(Readout,'values',lambda *a:np.array([0.,1.]))
    execute('one',initial_state());saved=json.loads((root/'one/continuation.json').read_text())
    assert any(float(p['qty'])>0 for p in saved['portfolios']['trained']['positions'].values())
    result=execute('two',saved)
    assert result['account_audits']['trained']['cash_flows_verified']
    assert result['metrics']['trained']['fees_usd']>=float(saved['portfolios']['trained']['fees'])
    assert abs(result['metrics']['trained']['marked_pnl_usd']-
        result['metrics']['trained']['realized_pnl_usd']-result['metrics']['trained']['unrealized_pnl_usd'])<1e-7


def test_rejects_modified_checkpoint_and_missing_capital(fixture):
    root,clock,manifest,execute=fixture
    Path(manifest['native']['path']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash'):execute('one',initial_state())
    s=initial_state();s['portfolios']['trained']['cash']='2000'
    with pytest.raises(ValueError,match='cash flows'):validate_state(s)


def test_detects_native_drift(fixture):
    root,clock,manifest,execute=fixture
    class CorruptFly(Fly):
        def observe(self,*args):
            self.controller.brain.weight[0]+=1
            return super().observe(*args)
    with pytest.raises(ValueError,match='native weights'):execute('one',initial_state(),fly_factory=CorruptFly)


def test_funding_is_once_and_respects_global_limit(tmp_path):
    shared=tmp_path/'shared';dest=tmp_path/'forward';p=shared/'solana-live'/CHECKPOINT;p.mkdir(parents=True)
    (p/'native').write_bytes(b'native');(p/'head').write_bytes(b'head')
    atomic_json(p/'online-state.json',dict(native_checkpoint=str(p/'native'),head_checkpoint=str(p/'head')))
    atomic_json(p/'completed.json',dict(status='completed',paper_only=True))
    atomic_json(shared/'budget.json',dict(months={'1970-01':44.4}))
    m=fund(shared,dest,now=1000)
    assert m['ends_at']==1000+3*86400
    assert json.loads((shared/'budget.json').read_text())['months']['1970-01']==61.9
    assert fund(shared,dest,now=2000)==m
    (dest/'manifest.json').unlink()
    with pytest.raises(ValueError,match='receipt'):fund(shared,dest,now=2000)


def control(root,**updates):
    c=dict(enabled=True,pending=None,sequence=0,spent_usd=0,allowance_usd=17.5,ends_at=5000,
           completed_sessions=0,archive_limit_bytes=64*1024**3)
    c.update(updates);atomic_json(root/'control.json',c);return c


def test_scheduler_waits_promotes_exact_account_and_stops_on_deadline(tmp_path):
    control(tmp_path);spawn=Mock(return_value='fc1');poll=Mock(return_value=None)
    c=coordinate(tmp_path,spawn,poll,now=1000)
    assert c['pending']['session']=='session-00001' and c['spent_usd']>=WORKER_RESERVATION
    coordinate(tmp_path,spawn,poll,now=1100);assert spawn.call_count==1
    root=tmp_path/'sessions/session-00001';root.mkdir(parents=True)
    s=initial_state();atomic_json(root/'continuation.json',s)
    result=dict(status='completed',ended=1900,continuation_sha256=digest(root/'continuation.json'),
                weights_unchanged=True,new_backprop_updates=0,checkpoint=CHECKPOINT)
    atomic_json(root/'completed.json',result);poll.return_value=result
    c=coordinate(tmp_path,spawn,poll,now=4990)
    assert c['status']=='completed_three_day_test' and not c['enabled']
    assert json.loads((tmp_path/'state.json').read_text())==s
    assert spawn.call_count==1


@pytest.mark.parametrize('result',[dict(status='failed'),dict(status='completed')])
def test_failure_never_resets_cash_or_retries(tmp_path,result):
    control(tmp_path,pending=dict(call_id='fc1',session='session-00001',reserved_at=1000))
    s=initial_state();atomic_json(tmp_path/'state.json',s);spawn=Mock()
    c=coordinate(tmp_path,spawn,lambda _:result,now=1100)
    assert not c['enabled'] and c['status']=='paused_worker_failure'
    assert json.loads((tmp_path/'state.json').read_text())==s
    spawn.assert_not_called()


def test_stops_before_budget_and_unknown_dispatch_not_retried(tmp_path):
    control(tmp_path,spent_usd=17.49);spawn=Mock()
    assert coordinate(tmp_path,spawn,lambda _:None,now=1000)['status']=='budget_stopped'
    spawn.assert_not_called()
    control(tmp_path);spawn.side_effect=RuntimeError('uncertain')
    with pytest.raises(RuntimeError):coordinate(tmp_path,spawn,lambda _:None,now=1000)
    assert json.loads((tmp_path/'control.json').read_text())['status']=='paused_dispatch_uncertain'
    coordinate(tmp_path,spawn,lambda _:None,now=1100);assert spawn.call_count==1


def test_funding_refuses_to_exceed_shared_guard(tmp_path):
    shared=tmp_path/'shared';shared.mkdir()
    atomic_json(shared/'budget.json',dict(months={'1970-01':47.}))
    with pytest.raises(ValueError,match='guard'):
        fund(shared,tmp_path/'forward',now=1000)
    assert json.loads((shared/'budget.json').read_text())['months']['1970-01']==47.
    assert not (tmp_path/'forward/manifest.json').exists()


def test_readout_weight_change_is_detected():
    h=Readout();initial={k:v.clone() for k,v in h.model.state_dict().items()}
    with h.torch.no_grad():next(h.model.parameters()).add_(.1)
    with pytest.raises(ValueError,match='readout'):
        verify_frozen(None,None,h,initial,0)


def test_portfolio_tampering_is_detected(fixture,monkeypatch):
    root,clock,manifest,execute=fixture
    monkeypatch.setattr(Readout,'values',lambda *a:np.array([0.,1.]))
    execute('one',initial_state())
    rows=[json.loads(x) for x in (root/'one/decisions.jsonl').read_text().splitlines()]
    rows[-1]['portfolios']['trained']['cash']='1000'
    with pytest.raises(ValueError,match='reconstruction'):
        audit_fills(initial_state(),rows,'trained')


def test_pending_deadline_and_budget_are_bounded(tmp_path):
    control(tmp_path,pending=dict(call_id='fc1',session='session-00001',reserved_at=1000))
    spawn=Mock()
    c=coordinate(tmp_path,spawn,lambda _:None,now=2600)
    assert not c['enabled'] and c['status']=='paused_worker_deadline'
    control(tmp_path,spent_usd=17.5,pending=dict(call_id='fc1',session='session-00001',reserved_at=1000))
    c=coordinate(tmp_path,spawn,lambda _:None,now=1100)
    assert not c['enabled'] and c['status']=='budget_stopped'
    spawn.assert_not_called()


def test_result_mismatch_pauses_without_changing_account(tmp_path):
    control(tmp_path,pending=dict(call_id='fc1',session='session-00001',reserved_at=1000))
    original=initial_state();atomic_json(tmp_path/'state.json',original)
    root=tmp_path/'sessions/session-00001';root.mkdir(parents=True)
    atomic_json(root/'completed.json',dict(status='completed',wrong=True))
    spawn=Mock();c=coordinate(tmp_path,spawn,lambda _:dict(status='completed'),now=1100)
    assert c['status']=='paused_result_validation' and not c['enabled']
    assert json.loads((tmp_path/'state.json').read_text())==original
    spawn.assert_not_called()
