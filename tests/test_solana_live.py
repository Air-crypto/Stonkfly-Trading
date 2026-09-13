import base64
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from paperlab.solana_events import (Feed, PROGRAM, Reader, SCHEMA, SOL, TOKEN,
                                    base58, decode, eligible, parse_notification)
from paperlab.solana_paper import COSTS, FEATURES, Readout, tick_for
from paperlab.core import Broker

FIXTURE=Path(__file__).parent/'fixtures/solana/confirmed-pump-layouts.json'


def test_real_confirmed_event_layouts():
    samples=json.loads(FIXTURE.read_text())
    for name,message in samples.items():
        rows,errors=parse_notification(message,1789317352.)
        assert errors==0 and name in [r['kind'] for r in rows]
        assert all(r['slot']==message['params']['result']['context']['slot'] for r in rows)
        assert all(r['received']==1789317352 for r in rows)


def test_event_provenance_excludes_other_program_and_failed_tx():
    message=json.loads(FIXTURE.read_text())['TradeEvent']
    mutated=deepcopy(message)
    mutated['params']['result']['value']['err']={'InstructionError':[0,'Custom']}
    assert parse_notification(mutated,1)==([],0)
    mutated=deepcopy(message)
    mutated['params']['result']['value']['logs']=[l.replace(PROGRAM,SOL) for l in mutated['params']['result']['value']['logs']]
    assert parse_notification(mutated,1)==([],0)


def test_unknown_and_truncated_layout_are_not_prices():
    assert decode(base64.b64encode(b'unknown!').decode()) is None
    event=next(e for e in SCHEMA['events'] if e['name']=='TradeEvent')
    with pytest.raises(ValueError,match='Truncated'): decode(base64.b64encode(bytes(event['discriminator'])).decode())
    message=json.loads(FIXTURE.read_text())['TradeEvent']
    lines=message['params']['result']['value']['logs']
    for line in lines:
        if line.startswith('Program data: '):
            raw=base64.b64decode(line[14:])
            if raw[:8]==bytes(event['discriminator']):
                with pytest.raises(ValueError,match='layout changed'): decode(base64.b64encode(raw+b'\0').decode())


def test_reader_limits():
    assert base58(bytes(32))=='1'*32
    with pytest.raises(ValueError): Reader(b'\2').read('bool')
    with pytest.raises(ValueError): Reader((5000).to_bytes(4,'little')).read('string')


def token(now=1000):
    return {'complete':False,'created':{'timestamp':now-100,'received':now-99,
        'quote_mint':SOL,'token_program':TOKEN,'is_mayhem_mode':False},
        'trades':[{'received':now-6+i,'timestamp':now-6+i,'is_buy':bool(i%2),
            'mint':'fixture','quote_mint':SOL,'mayhem_mode':False,
            'virtual_sol_reserves':100_000_000_000,'virtual_token_reserves':1_000_000_000_000,
            'real_sol_reserves':50_000_000_000} for i in range(6)]}


@pytest.mark.parametrize('change,reason',[
    ('complete','curve_completed'),('old','launch_age'),('late','late_creation'),
    ('quote','unsupported_quote'),('one_sided','one_sided'),('stale','stale_trade'),('thin','thin_real_reserves')])
def test_admission(change,reason):
    t=token()
    if change=='complete':t['complete']=True
    if change=='old':t['created']['timestamp']=1
    if change=='late':t['created']['received']=950
    if change=='quote':t['created']['quote_mint']='notSOL'
    if change=='one_sided':
        for r in t['trades']:r['is_buy']=True
    if change=='stale':t['trades'][-1]['received']=980
    if change=='thin':t['trades'][-1]['real_sol_reserves']=1
    if change=='old':assert eligible(t,4000)==reason
    else:assert eligible(t,1000)==reason


def test_reserve_price_base_units_and_delayed_paper_fills():
    t=token();tick,reason=tick_for(t,1000,100,999)
    assert reason is None
    assert tick.mid==pytest.approx(1e-8)
    b=Broker(COSTS)
    assert b.execute(.025,1000,tick)['reason']=='not_after_decision'
    fill=b.execute(.025,990,tick)
    assert fill['status']=='filled' and float(b.cash)>=974
    assert b.equity(tick)<1000  # Entry and future exit costs are charged to the mark.
    assert tick_for(t,1000,100,900)[1]=='stale_sol_usd'
    t['trades'][-1]['real_sol_reserves']=10_000_000_000
    assert tick_for(t,1000,100,999)[1]=='order_exceeds_one_percent_real_sol_reserves'


def test_registry_keeps_pinned_launch_when_capacity_full(tmp_path):
    f=Feed(tmp_path,max_tokens=2)
    for i in range(2):f.accept({'kind':'CreateEvent','mint':str(i),'received':i})
    f.pinned='0';f.accept({'kind':'CreateEvent','mint':'2','received':2})
    assert set(f.snapshot())=={'0','2'} and f.health()['overflow']==1


def test_readout_gradient_and_directional_update(tmp_path):
    h=Readout();x=np.zeros(FEATURES,dtype=np.float32)
    before=h.values(x).copy()
    metric=h.update({'x':x,'action':1},x,10,5)
    assert metric['loss']>0 and metric['gradient_l2_before_clip']>0
    assert metric['weight_delta_l2']>0 and h.values(x)[1]>before[1]
    assert metric['reward_scaled']==pytest.approx(.4)
    h.save(tmp_path/'head.pt')


def test_native_constructor_forbidden_on_laptop(tmp_path):
    from paperlab.solana_paper import run
    with pytest.raises(RuntimeError,match='never on the laptop'):run(tmp_path,'unused',seconds=60)


def test_independent_ledger_audit_rejects_tampered_cash_and_fill():
    from dataclasses import asdict
    from paperlab.solana_audit import audit
    t,_=tick_for(token(),1000,100,999);b=Broker(COSTS);f=b.execute(.025,990,t)
    row={'at':1000.,'tick':asdict(t),'quote_available':True,'fill':f,'broker':b.state(),
         'equity_stress_usd':b.equity(t),'neural':None}
    assert audit([row])['fills']==1
    bad=deepcopy(row);bad['broker']['cash']='1000'
    with pytest.raises(ValueError,match='Account'):audit([bad])
    bad=deepcopy(row);bad['fill']['decision_ts']=1000
    with pytest.raises(ValueError,match='Noncausal'):audit([bad])


def test_native_array_audit_rejects_false_update(tmp_path):
    import hashlib
    from paperlab.solana_audit import audit
    counts=np.array([1,2],dtype=np.int32);before=np.array([1.,2.],dtype=np.float32)
    after=before+.1;delta=after-before
    np.savez(tmp_path/'00000.npz',counts=counts,before=before,after=after,delta=delta)
    n={'compute_seconds':1.,'spike_sha256':hashlib.sha256(counts.tobytes()).hexdigest(),'total_spikes':3,
       'learning_diagnostics':{'changed_this_step':2,'weight_delta_l2':float(np.linalg.norm(delta))}}
    row={'at':1000.,'tick':None,'quote_available':False,'fill':{'status':'hold'},'broker':Broker(COSTS).state(),
         'equity_stress_usd':1000.,'neural':n}
    assert audit([row],native_root=tmp_path)['native_arrays_verified']==1
    row['neural']['learning_diagnostics']['weight_delta_l2']=0
    with pytest.raises(ValueError,match='Update norm'):audit([row],native_root=tmp_path)


def test_canonical_pool_and_migrated_quote(tmp_path):
    from paperlab.solana_events import canonical_pool
    payload=json.loads((FIXTURE.parent/'pumpswap-selected-events.json').read_text())
    mint=payload['created']['mint']
    assert canonical_pool(mint)=='5jLYKxisLsLxTeCNGd5QvQShfCP1jMqaVokjpNb8BndC'  # Independently matched to official JS SDK.
    f=Feed(tmp_path);f.accept(payload['created']);f.accept({'kind':'CompleteEvent','mint':mint})
    for e in payload['events']:f.accept(e)
    t=f.snapshot()[mint];assert t['migrated']
    last=t['trades'][-1];raw=payload['events'][-1]
    assert last['virtual_sol_reserves']==raw['pool_quote_token_reserves']+raw['virtual_quote_reserves']
    assert last['real_sol_reserves']==raw['pool_quote_token_reserves']
    assert last['venue']=='pumpswap'
    bad=deepcopy(raw);bad['pool']=SOL;before=len(t['trades']);f.accept(bad)
    assert len(f.snapshot()[mint]['trades'])==before
    assert Reader((-42).to_bytes(16,'little',signed=True)).read('i128')==-42


def test_readout_restores_predictions_optimizer_and_rng(tmp_path):
    h=Readout();x=np.ones(FEATURES,dtype=np.float32)
    h.update({'x':x,'action':0},x,-1.,5);path=tmp_path/'head.pt';h.save(path)
    restored=Readout();restored.restore(path)
    assert restored.updates==h.updates
    assert np.array_equal(restored.values(x),h.values(x))
    assert restored.choose(x)==h.choose(x)
    a=h.update({'x':x,'action':1},x,2.,5);b=restored.update({'x':x,'action':1},x,2.,5)
    assert a==b


def test_opening_balance_is_preserved_in_audit():
    from paperlab.solana_audit import audit
    opening={'cash':'985.6','qty':'5','fees':'0.5','halted':False}
    row={'at':1000.,'tick':None,'quote_available':False,'fill':{'status':'hold'},
         'broker':opening,'equity_stress_usd':985.6,'neural':None}
    assert audit([row],opening=opening)['end_equity_stress_usd']==985.6
    with pytest.raises(ValueError,match='Account'):audit([row])


def test_resume_refuses_incomplete_or_failed_window(tmp_path):
    from paperlab.solana_paper import prior_state
    (tmp_path/'completed.json').write_text(json.dumps({'status':'failed','paper_only':True}))
    with pytest.raises(ValueError,match='completed paper'):prior_state(tmp_path)


def test_cloud_continuation_is_bounded_and_does_not_retry_failures(monkeypatch):
    from types import SimpleNamespace
    import solana_cloud
    import paperlab.core
    calls=[];writes=[]
    monkeypatch.setattr(solana_cloud,'_run',lambda *a:{'status':'completed'})
    monkeypatch.setattr(solana_cloud,'worker',SimpleNamespace(spawn=lambda *a,**kw:(calls.append((a,kw)) or SimpleNamespace(object_id='fixture-call'))))
    monkeypatch.setattr(solana_cloud,'volume',SimpleNamespace(commit=lambda:None))
    monkeypatch.setattr(paperlab.core,'atomic_json',lambda *a:writes.append(a))
    result=solana_cloud._window('solana-fixture',60,False,None,2)
    assert result['next_call_id']=='fixture-call' and calls[0][1]['remaining_windows']==1
    assert calls[0][1]['resume_id']=='solana-fixture'
    calls.clear();monkeypatch.setattr(solana_cloud,'_run',lambda *a:{'status':'budget_stopped'})
    solana_cloud._window('solana-fixture',60,False,None,2)
    assert not calls
    with pytest.raises(ValueError,match='At most three'):solana_cloud._window('solana-fixture',60,False,None,3)
