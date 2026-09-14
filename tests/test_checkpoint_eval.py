import json
from pathlib import Path
import pytest
from paperlab.checkpoint_eval import register,seal_plan,seal_tape,verify_checkpoint,PROTOCOL,source_fingerprint
from paperlab.core import atomic_json
from paperlab.solana_quotes import QUOTE_PROTOCOL


def checkpoint(root,name,started,ended):
    p=root/'solana-live'/name;p.mkdir(parents=True)
    atomic_json(p/'completed.json',dict(status='completed',paper_only=True,account_mode='fresh_training_episode',started=started,ended=ended,readout_updates=10))
    for n in ['fly-final.npz','head-final.pt','events.db','fx.jsonl','decisions.jsonl']:(p/n).write_bytes(b'fixture')
    (p/'decisions.jsonl').write_text(json.dumps(dict(at=ended,observation_at=started,event_cursor=0,
        fx_state=dict(sol_usd=0,seen_at=0),quote_protocol=QUOTE_PROTOCOL,neural=None,decision=None))+'\n')
    return p


def test_registry_preserves_original_hash_and_detects_mutation(tmp_path):
    p=checkpoint(tmp_path,'solana-online-20260914-010000',10,20)
    first=register(tmp_path);assert len(first)==1
    verify_checkpoint(first[0]);(p/'head-final.pt').write_bytes(b'changed')
    assert register(tmp_path)==first
    with pytest.raises(ValueError,match='changed'):verify_checkpoint(first[0])


def test_registry_retains_reused_checkpoint_for_healthy_empty_window(tmp_path):
    parent=checkpoint(tmp_path,'solana-online-20260914-010000',10,20)
    empty=checkpoint(tmp_path,'solana-online-20260914-020000',30,40)
    for name in ('fly-final.npz','head-final.pt'):(empty/name).unlink()
    atomic_json(empty/'online-state.json',dict(native_checkpoint=str(parent/'fly-final.npz'),
                                              head_checkpoint=str(parent/'head-final.pt')))
    checkpoints=register(tmp_path)
    assert len(checkpoints)==2 and checkpoints[1]['checkpoint_reused']
    assert checkpoints[1]['files']['fly-final.npz']['path']==str(parent/'fly-final.npz')
    verify_checkpoint(checkpoints[1])
    (empty/'online-state.json').write_text('{}')
    with pytest.raises(ValueError,match='changed'):verify_checkpoint(checkpoints[1])


def test_source_fingerprint_covers_native_schema_and_added_import(tmp_path):
    for name in ('cloud.py','solana_cloud.py','checkpoint_eval_cloud.py'):(tmp_path/name).write_text('# image')
    native=tmp_path/'vendor/stonkfly/stonkfly/neural/kernel.cpp';native.parent.mkdir(parents=True);native.write_text('original')
    schema=tmp_path/'paperlab/solana_schema/pump.json';schema.parent.mkdir(parents=True);schema.write_text('{}')
    before=source_fingerprint(tmp_path)
    native.write_text('changed')
    assert source_fingerprint(tmp_path)!=before
    native.write_text('original');schema.write_text('{"version":2}')
    assert source_fingerprint(tmp_path)!=before
    schema.write_text('{}');(tmp_path/'paperlab/new_module.py').write_text('# imported helper')
    assert source_fingerprint(tmp_path)!=before


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
    import paperlab.solana_quotes as quote_module
    class Feed:
        def __init__(self,root):self.tokens={}
        def accept(self,e):
            t=self.tokens.setdefault('mint',dict(created={'timestamp':0},trades=[]))
            t['trades'].append(e)
        def snapshot(self):return self.tokens
    monkeypatch.setattr(events,'Feed',Feed)
    def fake_quote(t,now,fx,seen,**kw):
        tick=Tick(t['trades'][-1]['received'],.995,1.005,product='mint',received_at=t['trades'][-1]['received'])
        assert t['trades'][-1]['received']<=now
        return quote_module.QuoteSnapshot(tick,tick,tick,None,None,None,100.)
    monkeypatch.setattr(quote_module,'quote_for',fake_quote)
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
        rows.append(dict(at=now+.5,observation_at=now,event_cursor=i*2+1,fx_state=dict(sol_usd=100,seen_at=99),
                         quote_protocol=QUOTE_PROTOCOL,admissions=['mint'] if i==0 else [],retired=[],
                         feed=dict(status='connected',last_message=now-.1),neural=dict(mint='mint') if i>=11 else None,
                         decision=dict(mint='mint',issued=now+.25) if i>=11 else None))
        # Received before the snapshot, but applied after it. The cursor must exclude it.
        later={**e,'received':now-.05}
        db.execute('insert into events values(?,?,?)',(later['received'],i+100,json.dumps(later)))
    rows.append({**rows[-1],'at':rows[-1]['at']+1,'terminal':True,'neural':None,'decision':None,'executions':[]})
    db.commit();db.close()
    (tmp_path/'fx.jsonl').write_text(json.dumps(dict(at=99,sol_usd=100))+'\n')
    (tmp_path/'decisions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    plan=dict(protocol=PROTOCOL,quote_protocol=QUOTE_PROTOCOL,cutoff=1,policies=['always_long','untrained'],checkpoints=[],scope='test',tape=dict(id='fixture',started=90,files={n:dict(path=str(tmp_path/n),sha256=digest(tmp_path/n)) for n in ['events.db','fx.jsonl','decisions.jsonl']}))
    result=evaluate(plan,'always_long',tmp_path/'long','unused')
    assert result['account_audit']['paper_fills']>0
    assert result['pnl_usd']<0 and result['account_audit']['cash_usd']>=0
    actual=[json.loads(line) for line in (tmp_path/'long/decisions.jsonl').read_text().splitlines()]
    assert actual[11]['decision']['issued']==155.25
    assert result['coverage']['recorded_issued_opportunities']==9
    assert actual[-1]['terminal'] and actual[-1]['executions']==[]
    assert actual[-1]['portfolio']==actual[-2]['portfolio']
    result=evaluate(plan,'untrained',tmp_path/'pristine','unused',fly_factory=Fly)
    assert result['weights_unchanged'] and result['native_observations']>0
    assert result['account_audit']['new_head_updates']==0
    pristine=[json.loads(line) for line in (tmp_path/'pristine/decisions.jsonl').read_text().splitlines()]
    assert pristine[11]['policy_diagnostics']['input_features'][11]==pytest.approx(.01)

    class MutatingFly(Fly):
        def observe(self,*args):
            self.controller.brain.weight[0]+=1
            return super().observe(*args)  # A false zero diagnostic must not hide mutation.
    with pytest.raises(ValueError,match='Frozen native checkpoint changed'):
        evaluate(plan,'untrained',tmp_path/'mutating','unused',fly_factory=MutatingFly)

    rows[14]['fx_state']['sol_usd']=0
    rows[14]['decision']=None;rows[14]['neural']=None
    (tmp_path/'decisions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    plan['tape']['files']['decisions.jsonl']['sha256']=digest(tmp_path/'decisions.jsonl')
    evaluate(plan,'always_long',tmp_path/'failed-fx','unused')
    failed_fx=[json.loads(line) for line in (tmp_path/'failed-fx/decisions.jsonl').read_text().splitlines()]
    assert failed_fx[14]['executions']==[] and failed_fx[14]['marks']=={}

    (tmp_path/'fx.jsonl').write_text(json.dumps(dict(at=99,sol_usd=100))+'\n'+json.dumps(dict(at=98,sol_usd=100))+'\n')
    plan['tape']['files']['fx.jsonl']['sha256']=digest(tmp_path/'fx.jsonl')
    with pytest.raises(ValueError,match='unordered contemporaneous FX'):
        evaluate(plan,'untrained',tmp_path/'bad-fx','unused',fly_factory=Fly)


@pytest.mark.parametrize('change', [
    lambda rows: rows[0].pop('observation_at'),
    lambda rows: rows[1].update(observation_at=rows[0]['observation_at']),
    lambda rows: rows[1].update(decision=dict(mint='mint',issued=101)),
    lambda rows: rows[1].update(neural=dict(mint='mint',tick=dict(received_at=111))),
])
def test_rejects_missing_or_noncausal_observation_cutoff(change):
    from paperlab.checkpoint_eval import validate_observation_rows
    rows=[dict(at=101,observation_at=100,event_cursor=0,fx_state=dict(sol_usd=0,seen_at=0),quote_protocol=QUOTE_PROTOCOL,decision=None,neural=None),
          dict(at=111,observation_at=110,event_cursor=1,fx_state=dict(sol_usd=0,seen_at=0),quote_protocol=QUOTE_PROTOCOL,decision=None,neural=dict(mint='mint'))]
    change(rows)
    with pytest.raises(ValueError):validate_observation_rows(rows)


def test_unchanged_receipt_keeps_pending_order_when_flow_guard_ages(tmp_path):
    """Aging entry eligibility alone must not cancel an unprocessed live intent."""
    import sqlite3
    from paperlab.checkpoint_eval import evaluate
    from paperlab.core import digest
    from paperlab.solana_events import SOL,TOKEN
    mint='CeSFzoAqSMXMgodeLzrTMe3V5MdV5Nhinehedxohpump'
    events=[dict(kind='CreateEvent',mint=mint,quote_mint=SOL,token_program=TOKEN,is_mayhem_mode=False,
                 received=10,timestamp=10,slot=1,signature='create',log_index=0)]
    receipts=[(35,True),(36,False),(37,True),(38,False),(39,True),(44,True),
              (49,False),(54,True),(59,False),(64,True),(69,False),(71,True),
              (72,False),(73,False),(74,True),(79,True),(84,True),(89,True),
              (94,True),(99,True),(108,False),(109,False)]
    for index,(at,buy) in enumerate(receipts):
        events.append(dict(kind='TradeEvent',mint=mint,received=at,timestamp=at,slot=index+2,
            signature='trade-'+str(index),log_index=index+1,is_buy=buy,sol_amount=1_000_000_000,
            quote_mint=SOL,mayhem_mode=False,virtual_sol_reserves=60_000_000_000,
            virtual_token_reserves=500_000_000_000_000,real_sol_reserves=30_000_000_000))
    db=sqlite3.connect(tmp_path/'events.db');db.execute('create table events(body text)')
    db.executemany('insert into events values(?)',[(json.dumps(e),) for e in events]);db.commit();db.close()
    rows=[]
    for at in range(45,111,5):
        prefix=[e for e in events if e['received']<=at]
        opportunity=at==100
        rows.append(dict(at=at+.5,observation_at=at,event_cursor=len(prefix),
            quote_protocol=QUOTE_PROTOCOL,fx_state=dict(sol_usd=100,seen_at=40),
            admissions=[mint] if at==45 else [],retired=[],
            feed=dict(status='connected',last_message=prefix[-1]['received']),
            neural=dict(mint=mint,tick=dict(received_at=99)) if opportunity else None,
            decision=dict(mint=mint,issued=100.25) if opportunity else None))
    (tmp_path/'fx.jsonl').write_text(json.dumps(dict(at=40,sol_usd=100))+'\n')
    (tmp_path/'decisions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    plan=dict(protocol=PROTOCOL,quote_protocol=QUOTE_PROTOCOL,cutoff=1,policies=['always_long'],
        checkpoints=[],scope='Receipt processing regression',tape=dict(id='receipt-fixture',started=5,
        files={n:dict(path=str(tmp_path/n),sha256=digest(tmp_path/n))
               for n in ('events.db','fx.jsonl','decisions.jsonl')}))
    result=evaluate(plan,'always_long',tmp_path/'result','unused')
    ledger=[json.loads(line) for line in (tmp_path/'result/decisions.jsonl').read_text().splitlines()]
    assert ledger[-2]['observation_at']==105 and ledger[-2]['executions']==[]
    assert len(ledger[-1]['executions'])==1
    fill=ledger[-1]['executions'][0]['fill']
    assert fill['status']=='filled' and fill['fill_ts']==109 and fill['decision_ts']==100.25
    assert result['account_audit']['paper_fills']==1


def test_simultaneous_pending_fills_share_cash_in_active_admission_order(tmp_path):
    import sqlite3
    from solders.pubkey import Pubkey
    from paperlab.checkpoint_eval import evaluate
    from paperlab.core import digest
    from paperlab.solana_events import SOL,TOKEN
    mints={label:str(Pubkey(bytes([index+1])*32)) for index,label in enumerate('ABCDEF')}
    events=[]
    for label,mint in mints.items():
        events.append(dict(kind='CreateEvent',mint=mint,quote_mint=SOL,token_program=TOKEN,is_mayhem_mode=False,
                           received=950,timestamp=950,slot=1,signature='create-'+label,log_index=0))
        times=[990,991,992,993,994]+[999.9+step*5 for step in range(19)
                   if not (label=='B' and step in (16,17)) and not (label=='A' and step==17)]
        for index,received in enumerate(times):
            # Prior C/D/E entries nearly exhaust their $250 caps, F consumes
            # $202.50; the remaining ~$47.50 cannot fund both A and B fully.
            reserves=30 if label in 'AB' else 200 if label=='F' else 1000
            events.append(dict(kind='TradeEvent',mint=mint,received=received,timestamp=int(received),
                slot=int(received),signature=label+'-'+str(index),log_index=index+1,
                is_buy=index%2==0,sol_amount=1_000_000_000,quote_mint=SOL,mayhem_mode=False,
                virtual_sol_reserves=1_500_000_000_000,virtual_token_reserves=500_000_000_000_000,
                real_sol_reserves=reserves*1_000_000_000))
    events.sort(key=lambda event:(event['received'],event['signature']))
    db=sqlite3.connect(tmp_path/'events.db');db.execute('create table events(body text)')
    db.executemany('insert into events values(?)',[(json.dumps(event),) for event in events]);db.commit();db.close()
    offers={11:'C',12:'D',13:'E',14:'F',15:'B',16:'A'}
    rows=[]
    for step in range(19):
        at=1000+step*5;prefix=[event for event in events if event['received']<=at]
        mint=mints[offers[step]] if step in offers else None
        rows.append(dict(at=at+.5,observation_at=at,event_cursor=len(prefix),quote_protocol=QUOTE_PROTOCOL,
            fx_state=dict(sol_usd=100,seen_at=999),admissions=list(mints.values()) if step==0 else [],retired=[],
            feed=dict(status='connected',last_message=prefix[-1]['received']),
            neural=dict(mint=mint,tick=dict(received_at=at-.1)) if mint else None,
            decision=dict(mint=mint,issued=at+.25) if mint else None))
    # Keep the final FX observation fresh without changing the conversion.
    for row in rows:
        if row['observation_at']>=1080:row['fx_state']['seen_at']=1080
    (tmp_path/'fx.jsonl').write_text(''.join(json.dumps(dict(at=at,sol_usd=100))+'\n' for at in (999,1080)))
    (tmp_path/'decisions.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    plan=dict(protocol=PROTOCOL,quote_protocol=QUOTE_PROTOCOL,cutoff=900,policies=['always_long'],
        checkpoints=[],scope='Shared-cash execution order regression',tape=dict(id='order-fixture',started=901,
        files={name:dict(path=str(tmp_path/name),sha256=digest(tmp_path/name))
               for name in ('events.db','fx.jsonl','decisions.jsonl')}))
    result=evaluate(plan,'always_long',tmp_path/'result','unused')
    ledger=[json.loads(line) for line in (tmp_path/'result/decisions.jsonl').read_text().splitlines()]
    final_fills=[entry for entry in ledger[-1]['executions'] if entry['fill']['status']=='filled']
    assert [entry['mint'] for entry in final_fills]==[mints['A'],mints['B']]
    assert float(result['portfolio']['positions'][mints['A']]['basis'])==pytest.approx(30.375)
    assert float(result['portfolio']['positions'][mints['B']]['basis'])==pytest.approx(17.125732874,abs=1e-7)
    assert result['account_audit']['paper_fills']==6
