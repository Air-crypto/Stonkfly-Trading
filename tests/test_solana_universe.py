import asyncio
import base64
from copy import deepcopy
import json
import sqlite3
import struct

import pytest
from solders.pubkey import Pubkey
from paperlab.solana_events import SOL,ZERO,TOKEN,AMM_PROGRAM
from paperlab.solana_quotes import quote_for
from paperlab.solana_universe import (AllObservedFeed,POOL_SCHEMA,QUOTE_PROTOCOL,USDC,
    pool_metadata,prepare_snapshots,update_all_active)

def address(n):return str(Pubkey(bytes([n])*32))

def trade(mint=None,now=100,quote=SOL):
    return dict(kind='TradeEvent',mint=mint or address(1),quote_mint=quote,received=now,
        timestamp=int(now),slot=int(now),signature=str(now),log_index=1,is_buy=True,mayhem_mode=True,
        virtual_sol_reserves=100_000_000_000,virtual_token_reserves=1_000_000_000_000,
        real_sol_reserves=2_000_000_000,sol_amount=100_000_000)

def token(event):
    return dict(created=dict(kind='CreateEvent',mint=event['mint'],timestamp=0,received=99,
        quote_mint=event['quote_mint'],token_program=TOKEN,is_mayhem_mode=True),
        trades=[event],complete=False)

def test_old_one_sided_mayhem_thin_token_can_enter_but_stale_cannot():
    t=token(trade());old=quote_for(t,100,100,99)
    assert old.entry_tick is None
    q=quote_for(t,100,100,99,unrestricted=True)
    assert q.entry_tick and q.liquidity_notional_usd==2 and q.protocol==QUOTE_PROTOCOL
    assert quote_for(t,111,100,99,unrestricted=True).observed_tick is None
    t['trades'][0]['real_sol_reserves']=0
    q=quote_for(t,100,100,99,unrestricted=True)
    assert q.observed_tick and q.entry_tick is None

def test_unknown_curve_discovered_without_launch_and_no_eviction(tmp_path):
    f=AllObservedFeed(tmp_path)
    for i in range(600):
        m=str(Pubkey(i.to_bytes(32,'little')));f.accept(trade(m),event_cursor=i+1)
    assert len(f.tokens)==600 and f.health()['overflow']==0 and f.health()['event_cursor']==600
    assert not f.tokens[address(0)]['created']['creation_time_known']

def test_all_active_queue_has_no_eight_or_128_cap_and_readmits():
    snapshots={str(i):token(trade()) for i in range(600)}
    quotes={m:object() for m in snapshots};a={};w={};inactive={};retired=set()
    admissions,_=update_all_active(a,w,retired,inactive,None,snapshots,quotes,100)
    assert len(admissions)==len(a)==600
    update_all_active(a,w,retired,inactive,None,snapshots,{},101)
    _,removed=update_all_active(a,w,retired,inactive,None,snapshots,{},222)
    assert len(removed)==600 and not retired
    admissions,_=update_all_active(a,w,retired,inactive,None,snapshots,quotes,223)
    assert len(admissions)==600

def test_pool_identity_owner_and_discriminator():
    raw=bytes(POOL_SCHEMA['account']['discriminator'])+struct.pack('<BH',1,3)
    raw+=bytes(Pubkey.from_string(address(2)))+bytes(Pubkey.from_string(address(3)))+bytes(Pubkey.from_string(USDC))
    account=dict(owner=AMM_PROGRAM,executable=False,data=[base64.b64encode(raw).decode(),'base64'])
    meta=pool_metadata(address(4),account,110,300)
    assert meta['base_mint']==address(3) and meta['quote_mint']==USDC and meta['received']==110
    for invalid in [{**account,'owner':TOKEN},{**account,'data':['AAAA','base64']},{**account,'executable':True}]:
        with pytest.raises(ValueError):pool_metadata(address(4),invalid,110,300)

def test_unknown_pool_not_retroactively_replayed_and_arbitrary_pool_supported(tmp_path):
    f=AllObservedFeed(tmp_path);pool=address(4);mint=address(3)
    e=dict(kind='BuyEvent',pool=pool,received=100,timestamp=100,slot=100,signature='buy',log_index=1,
        quote_amount_in=1_000_000,pool_quote_token_reserves=1_000_000_000,
        virtual_quote_reserves=500_000_000,pool_base_token_reserves=10_000_000_000)
    f.accept(e,event_cursor=1)
    assert not f.tokens and f.health()['unresolved_pool_trade_events']==1
    meta=dict(kind='PoolMetadata',pool=pool,base_mint=mint,quote_mint=USDC,
        received=105,timestamp=105,slot=105)
    f.accept(meta,event_cursor=2)
    assert not f.tokens[mint]['trades']  # Do not inject an earlier event after metadata arrives.
    f.accept({**e,'received':106,'timestamp':106,'slot':106},event_cursor=3)
    t=f.tokens[mint]['trades'][-1]
    assert t['quote_mint']==USDC and t['virtual_sol_reserves']==1_500_000_000
    assert f.health()['unresolved_pools']==0 and f.health()['event_cursor']==3

def test_usdc_conversion_is_priced_not_assumed_and_preserves_raw():
    e={**trade(quote=USDC),'virtual_quote_reserves':2_000_000_000,
       'real_quote_reserves':1_000_000_000,'quote_amount':1_000_000}
    s={e['mint']:token(e)};before=deepcopy(s)
    result=prepare_snapshots(s,100,100,99,{USDC:.98})
    q=quote_for(result[e['mint']],100,100,99,unrestricted=True)
    assert q.observed_tick.mid==pytest.approx(1960/1e12)
    assert q.liquidity_notional_usd==pytest.approx(9.8)
    assert s==before
    missing=prepare_snapshots(s,100,100,99,{})
    assert quote_for(missing[e['mint']],100,100,99,unrestricted=True).observed_tick is None

def test_fresh_observed_quote_asset_path_and_stale_path_rejection():
    bridge=address(5);base=address(6)
    e1={**trade(bridge),'virtual_token_reserves':1_000_000_000} # $0.00001/base unit
    e2={**trade(base,quote=bridge),'virtual_quote_reserves':1_000_000,
        'real_quote_reserves':1_000_000,'quote_amount':100,'virtual_token_reserves':1000}
    s={base:token(e2),bridge:token(e1)}
    prepared=prepare_snapshots(s,100,100,99)
    assert quote_for(prepared[base],100,100,99,unrestricted=True).observed_tick.mid==pytest.approx(.01)
    e1['received']=80
    prepared=prepare_snapshots(s,100,100,99)
    assert quote_for(prepared[base],100,100,99,unrestricted=True).observed_tick is None

def test_unknown_pool_raw_event_is_archived(tmp_path,monkeypatch):
    import paperlab.solana_events as module
    import websockets
    f=AllObservedFeed(tmp_path)
    event=dict(kind='BuyEvent',pool=address(9),signature='raw',log_index=1,received=100,timestamp=100,slot=1)
    monkeypatch.setattr(module,'parse_notification',lambda *_:([event],0))
    class WS:
        count=0
        async def send(self,_):pass
        async def recv(self):
            self.count+=1
            if self.count==1:return json.dumps(dict(result=1))
            f.stop.set();return '{}'
        async def __aenter__(self):return self
        async def __aexit__(self,*_):pass
    monkeypatch.setattr(websockets,'connect',lambda *a,**k:WS())
    asyncio.run(f._run())
    db=sqlite3.connect(tmp_path/'events.db')
    assert json.loads(db.execute('select body from events').fetchone()[0])==event
    db.close()

def test_broad_usdc_replay_uses_same_conversion_and_rejects_changed_fx(tmp_path):
    from paperlab.checkpoint_eval import evaluate,PROTOCOL
    from paperlab.core import digest
    mint=address(1)
    events=[dict(kind='CreateEvent',mint=mint,quote_mint=USDC,token_program=TOKEN,
        is_mayhem_mode=True,timestamp=1,received=999,signature='create',log_index=0,slot=1)]
    rows=[]
    for i in range(16):
        now=1000+i*5
        e={**trade(mint,now-.1,USDC),'virtual_quote_reserves':2_000_000_000,
           'real_quote_reserves':1_000_000_000,'quote_amount':1_000_000}
        events.append(e)
        rows.append(dict(at=now+.5,observation_at=now,event_cursor=i+2,quote_protocol=QUOTE_PROTOCOL,
            fx_state=dict(sol_usd=100,seen_at=999,quote_usd={USDC:.98}),
            feed=dict(status='connected',last_message=now-.1),admissions=[mint] if i==0 else [],retired=[],
            neural=dict(mint=mint) if i>=12 else None,
            decision=dict(mint=mint,issued=now+.25) if i>=12 else None))
    db=sqlite3.connect(tmp_path/'events.db');db.execute('create table events(body text)')
    db.executemany('insert into events values(?)',[(json.dumps(e),) for e in events]);db.commit();db.close()
    (tmp_path/'fx.jsonl').write_text(json.dumps(dict(at=999,sol_usd=100,quote_usd={USDC:.98}))+'\n')
    def plan():
        (tmp_path/'decisions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        return dict(protocol=PROTOCOL,quote_protocol=QUOTE_PROTOCOL,all_observed=True,cutoff=900,
            policies=['always_long'],checkpoints=[],scope='Broad USD conversion test',tape=dict(id='broad',started=950,
            files={n:dict(path=str(tmp_path/n),sha256=digest(tmp_path/n))
                   for n in ('events.db','fx.jsonl','decisions.jsonl')}))
    r=evaluate(plan(),'always_long',tmp_path/'evaluation','unused')
    assert r['account_audit']['paper_fills']==3 and r['coverage']['policy_decisions']==4
    ledger=[json.loads(x) for x in (tmp_path/'evaluation/decisions.jsonl').read_text().splitlines()]
    fill=next(e for row in ledger for e in row['executions'] if e['fill']['status']=='filled')
    assert float(fill['fill']['price'])==pytest.approx(1960/1e12*1.005*1.01)
    rows[-1]['fx_state']['quote_usd']={USDC:1.}
    with pytest.raises(ValueError,match='Quote FX'):evaluate(plan(),'always_long',tmp_path/'bad','unused')
