"""Bounded live Solana paper-training pilot. Native fly propagation is cloud-only.

Five-second observation deadlines are targets, never fabricated samples. A single
newly observed launch is selected without future outcomes and retained to the end.
"""
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import numpy as np
import requests

from .core import Broker, Costs, Tick, atomic_json
from .solana_events import Feed, eligible

INTERVAL=5.
COSTS=Costs(capital=1000,fee_bps=125,slippage_bps=100,max_exposure=.025,
            max_order=25,max_spread_bps=110,max_delay=15,loss_stop=.10)
FEATURES=22


def tick_for(token, now, sol_usd, fx_seen):
    reason=eligible(token,now)
    if not math.isfinite(sol_usd) or sol_usd<=0 or not 0<=now-fx_seen<=90:
        return None,'stale_sol_usd'
    if not token['trades']: return None,reason or 'missing_trade'
    t=token['trades'][-1]
    # USD per token BASE UNIT: no assumed token decimals. Quantities use base units.
    if min(t['virtual_sol_reserves'],t['virtual_token_reserves'])<=0: return None,'invalid_reserves'
    mid=t['virtual_sol_reserves']/t['virtual_token_reserves']/1e9*sol_usd
    if t['real_sol_reserves']/1e9*sol_usd < COSTS.max_order*100:
        reason=reason or 'order_exceeds_one_percent_real_sol_reserves'
    tick=Tick(t['received'],mid*.995,mid*1.005,product=t['mint'],
              source='solana_confirmed_pump_reserve_indicative',received_at=t['received'],available=reason is None)
    return tick,reason


def inputs(ticks, token, broker, event, now):
    p=np.array([t.mid for t in ticks]); last=token['trades'][-1]
    recent=[v for v in token['trades'] if now-v['received']<=30]
    returns=[np.clip(math.log(p[-1]/p[max(0,len(p)-1-lag)])*10,-5,5) for lag in (1,3,6,12)]
    flow=(sum(t['sol_amount']*(1 if t['is_buy'] else -1) for t in recent)/
          max(1,sum(t['sol_amount'] for t in recent)))
    market=returns+[min(1,(now-token['created']['timestamp'])/3600),flow,
        math.log1p(len(recent))/5,math.log1p(last['real_sol_reserves']/1e9)/10,
        min(1,float(broker.qty)*ticks[-1].mid/25),float(broker.cash)/1000,
        (broker.equity(ticks[-1])-1000)/25, min(1,(now-last['received'])/10)]
    neural=[event['left_hz']/100,event['right_hz']/100,event['gate_spikes']/10,
            math.log1p(event['total_spikes'])/15,math.log1p(event['KC_spikes'])/15,
            math.log1p(event['reward_spikes'])/15,math.log1p(event['aversive_spikes'])/15,
            event['difference_hz']/100,event['learning_diagnostics']['weight_delta_l2']/100,
            event['learning_diagnostics']['kc_trace_mean_hz']/100]
    x=np.clip(np.array(market+neural,dtype=np.float32),-5,5)
    if x.shape!=(FEATURES,) or not np.isfinite(x).all(): raise ValueError('Invalid head inputs')
    return x


class Readout:
    """Small online Q head; the fly itself also receives realized equity feedback."""
    def __init__(self, seed=13):
        import torch
        torch.manual_seed(seed)
        self.torch=torch
        self.model=torch.nn.Sequential(torch.nn.Linear(FEATURES,32),torch.nn.Tanh(),torch.nn.Linear(32,2))
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=.0003)
        self.rng=np.random.default_rng(seed); self.updates=0
    def values(self,x):
        with self.torch.no_grad(): return self.model(self.torch.tensor(x)).numpy().astype(float)
    def choose(self,x):
        q=self.values(x); explore=bool(self.rng.random()<.10)
        action=int(self.rng.integers(2)) if explore else int(np.argmax(q))
        return action,{'q_values':q.tolist(),'exploration':explore,'epsilon':.10}
    def update(self,previous,x,reward,elapsed):
        torch=self.torch; before=torch.cat([p.detach().flatten() for p in self.model.parameters()]).clone()
        clipped=float(np.clip(reward/25,-1,1)); discount=.95**(elapsed/5)
        with torch.no_grad(): target=clipped+discount*self.model(torch.tensor(x)).max()
        estimate=self.model(torch.tensor(previous['x']))[previous['action']]
        loss=torch.nn.functional.smooth_l1_loss(estimate,target)
        self.optimizer.zero_grad(); loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.)
        self.optimizer.step(); self.updates+=1
        after=torch.cat([p.detach().flatten() for p in self.model.parameters()])
        return dict(algorithm='online_one_step_Q_learning',updates=self.updates,loss=float(loss.detach()),
                    gradient_l2_before_clip=float(norm),weight_delta_l2=float(torch.linalg.vector_norm(after-before)),
                    reward_usd=reward,reward_scaled=clipped,discount=discount,target=float(target),
                    td_error=float(target-estimate.detach()),transition_seconds=elapsed)
    def save(self,path):
        self.torch.save({'model':self.model.state_dict(),'optimizer':self.optimizer.state_dict(),
                         'updates':self.updates,'rng':self.rng.bit_generator.state},path)


def run(root,data,*,seconds=900,commit=lambda:None,fly_factory=None,feed_factory=Feed,clock=time.time):
    if not 60<=seconds<=900: raise ValueError('Pilot is bounded to 60–900 seconds')
    if fly_factory is None:
        import modal
        if modal.is_local(): raise RuntimeError('Native fly must run in Modal, never on the laptop')
        from .fly import Fly
        fly_factory=Fly
    root=Path(root)
    if (root/'started.json').exists(): raise ValueError('Preserve pilot state; do not restart this run id')
    root.mkdir(parents=True,exist_ok=True)
    started=clock(); deadline=started+seconds
    atomic_json(root/'started.json',dict(at=started,deadline=deadline,paper_only=True,interval_seconds=5,
        native_learning=True,readout_learning=True,capital=1000,costs=asdict(COSTS),max_selected_tokens=1,
        cohort='first causally eligible newly observed Pump.fun Solana launch; no replacement',
        evaluation='online training pilot, not a held-out profitability test',news_enabled=False))
    commit()
    feed=feed_factory(root); feed.start()
    fly=None; head=None; token_id=None; ticks=[]; previous=None; pending=None
    broker=Broker(COSTS); holding=Broker(COSTS); holding_pending=None; holding_entered=False
    sol_usd=0.; fx_seen=0.; next_fx=0.; next_step=clock(); anchor=1000.; last_usable=None
    rows=0; trained=0; skipped=0; overruns=0; last_signature=None; status='collecting'; peak=1000.
    ledger=open(root/'decisions.jsonl','a'); fxlog=open(root/'fx.jsonl','a')
    try:
        while clock()<deadline:
            now=clock()
            if now<next_step: time.sleep(min(next_step-now,1)); continue
            step_start=now; next_step=now+INTERVAL
            if now>=next_fx:
                next_fx=now+60
                try:
                    response=requests.get('https://api.coinbase.com/v2/prices/SOL-USD/spot',timeout=4)
                    response.raise_for_status(); value=float(response.json()['data']['amount'])
                    if not math.isfinite(value) or value<=0: raise ValueError('Invalid SOL/USD')
                    sol_usd=value; fx_seen=clock()
                    fxlog.write(json.dumps({'received':fx_seen,'sol_usd':value,'source':'coinbase_spot_indicative'})+'\n');fxlog.flush()
                except Exception as exc:
                    print(json.dumps({'event':'solana_fx_error','error_type':type(exc).__name__}),flush=True)
            now=clock(); snapshots=feed.snapshot(); health=feed.health()
            if health['status'] in ('failed','capacity_stopped'): status=health['status']; break
            if token_id is None:
                candidates=[]
                for mint,token in snapshots.items():
                    tick,reason=tick_for(token,now,sol_usd,fx_seen)
                    if tick is not None and reason is None: candidates.append((token['created']['received'],mint))
                if candidates:
                    token_id=min(candidates)[1]
                    with feed.lock: feed.pinned=token_id
                    atomic_json(root/'selection.json',{'at':now,'mint':token_id,'created':snapshots[token_id]['created'],
                        'eligible_candidates':len(candidates),'rule':'earliest received eligible launch, tie by mint'})
                    print(json.dumps({'event':'solana_selected','mint':token_id,'age_seconds':now-snapshots[token_id]['created']['timestamp']}),flush=True)
            neural=None; learning=None; fill={'status':'hold'}; reason='waiting_for_eligible_launch'
            t=None; available=False
            if token_id and token_id in snapshots:
                token=snapshots[token_id]; t,reason=tick_for(token,now,sol_usd,fx_seen)
                if now-health['last_message']>10: reason='disconnected_feed'
                signature=(token['trades'][-1]['signature'],token['trades'][-1]['log_index']) if token['trades'] else None
                fresh=t is not None and reason is None and signature!=last_signature
                available=t is not None and reason is None
                if fresh:
                    last_signature=signature
                    # A fill requires a new provider receipt AFTER model computation and decision publication.
                    if pending:
                        fill=broker.execute(pending['target'],pending['issued'],t)
                        if fill.get('reason')!='not_after_decision': pending=None
                    if holding_pending and not holding_entered:
                        reference_fill=holding.execute(COSTS.max_exposure,holding_pending,t)
                        holding_entered=reference_fill['status']=='filled'
                    else: reference_fill={'status':'hold'}
                    equity=broker.equity(t); delta=equity-anchor
                    continuous=last_usable is not None and 0<t.ts-last_usable<=15
                    if not continuous: previous=None; delta=0.
                    ticks.append(t);ticks=ticks[-100:]
                    if len(ticks)>=12 and deadline-clock()>30:
                        if fly is None:
                            from .news import News
                            news=News(enabled=False);fly=fly_factory(data,learning=True);head=Readout()
                        # No stale decision following a long initial graph load.
                        if clock()-t.received_at<=10:
                            neural=fly.observe(ticks,len(ticks)-1,news,delta,
                                root/'neural'/f'{trained:05d}')
                            x=inputs(ticks,token,broker,neural,now)
                            if previous is not None:
                                learning=head.update(previous,x,delta,t.ts-previous['ts'])
                            action,prediction=head.choose(x)
                            issued=clock()
                            if issued-t.received_at<=15 and pending is None:
                                pending={'target':[0,COSTS.max_exposure][action],'issued':issued}
                                previous={'x':x,'action':action,'ts':t.ts}
                            else:
                                previous=None;prediction['decision_skipped']='stale_after_compute_or_pending'
                            trained+=1; status='training'
                            if not holding_pending: holding_pending=issued
                            neural.update(readout_action=['FLAT','LONG_2.5_PERCENT'][action],readout=prediction,
                                          input_features=x.tolist(),head_training=learning)
                        else: skipped+=1;reason='stale_after_initialization';previous=None
                    anchor=equity;last_usable=t.ts
                elif not available:
                    previous=None;last_usable=None;skipped+=1
                    if pending and now-pending['issued']>COSTS.max_delay: pending=None
                else: reason='no_new_trade';skipped+=1
            mark=broker.equity(t) if t is not None and available else float(broker.cash)
            hold_mark=holding.equity(t) if t is not None and available else float(holding.cash)
            peak=max(peak,mark)
            row={'event':'solana_paper_step','at':clock(),'step':rows,'mint':token_id,'status':status,
                 'reason':reason,'quote_available':available,'tick':asdict(t) if t else None,
                 'equity_stress_usd':mark,'cash_baseline_usd':1000,'one_entry_hold_equity_usd':hold_mark,
                 'broker':broker.state(),'fill':fill,'pending':pending,'neural':neural,
                 'feed':health,'tracked_launches':len(snapshots),'neural_observations':trained,
                 'readout_updates':head.updates if head else 0,'skipped_steps':skipped,
                 'step_seconds':clock()-step_start,'target_interval_seconds':5,'paper_only':True}
            if row['step_seconds']>5: overruns+=1
            row['overruns']=overruns
            ledger.write(json.dumps(row,allow_nan=False)+'\n');ledger.flush(); rows+=1
            atomic_json(root/'latest.json',row)
            print(json.dumps({k:v for k,v in row.items() if k not in ('tick','neural','feed','broker')},allow_nan=False),flush=True)
            if neural: print(json.dumps({'event':'solana_fly_learning','step':rows,'native':neural['learning_diagnostics'],
                                         'readout':learning,'action':neural['readout_action']}),flush=True)
            if rows%12==0: commit()
        status='completed' if clock()>=deadline else status
    except BaseException:
        status='failed'
        raise
    finally:
        feed.close();ledger.close();fxlog.close()
        if fly is not None:
            fly.save(root/'fly-final.npz');head.save(root/'head-final.pt')
        result=dict(status=status,started=started,ended=clock(),selected_mint=token_id,
                    steps=rows,neural_observations=trained,readout_updates=head.updates if head else 0,
                    skipped_steps=skipped,overruns=overruns,feed=feed.health(),broker=broker.state(),
                    paper_only=True,profitable_learning_proven=False,
                    note='Training trajectory with exploratory paper trades; no independent holdout or executable-price validation.')
        atomic_json(root/'result.json',result);commit()
    return result
