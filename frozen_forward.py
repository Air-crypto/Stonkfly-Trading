"""Prospective frozen paper trader. Kept outside the sealed replay source tree."""
from collections import Counter
from decimal import Decimal
import json
from pathlib import Path
import time

import numpy as np
import requests
from paperlab.core import Tick, atomic_json, digest
from paperlab.solana_online import Portfolio, EPISODE_POLICY, choose
from paperlab.solana_paper import Readout, inputs
from paperlab.solana_quotes import quote_for
from paperlab.solana_universe import AllObservedFeed, prepare_snapshots, update_all_active, USDC
from paperlab.news import News

PROTOCOL = 'frozen_continuous_forward_v1'


def fresh_account():
    return dict(cash='1000', fees='0', halted=False, positions={}, quarantined=[], risk_policy=EPISODE_POLICY)


def initial_state():
    return dict(portfolios={name:fresh_account() for name in ('trained','always_long')},
                watched={}, pools=[], contexts={}, visits={}, steps=0, observations=0,
                fills={'trained':0,'always_long':0}, last_at=None)


def validate_state(saved):
    if set(saved['portfolios']) != {'trained','always_long'}:
        raise ValueError('Missing continuous account')
    for value in saved['portfolios'].values():
        p = Portfolio(value)
        if not p.episodic or abs(p.cash-Decimal(1000)-sum(x['cash_flow'] for x in p.positions.values())) > Decimal('1e-7'):
            raise ValueError('Continuous cash flows do not reconcile')


def account_values(portfolio, quotes, views):
    basis = sum(float(p['basis']) for p in portfolio.positions.values())
    marked = portfolio.equity(quotes)
    next_fill = sum(views[m].position_values(p['qty'])['next_fill_proceeds_usd']
                    for m,p in portfolio.positions.items() if p['qty'] and m in views)
    return dict(cash_usd=float(portfolio.cash), marked_equity_usd=marked,
        marked_pnl_usd=marked-1000, realized_pnl_usd=float(portfolio.cash)-1000+basis,
        unrealized_pnl_usd=marked-float(portfolio.cash)-basis, fees_usd=float(portfolio.fees),
        capacity_stress_equity_usd=float(portfolio.cash)+next_fill,
        capacity_stress_pnl_usd=float(portfolio.cash)+next_fill-1000,
        open_positions=sum(bool(p['qty']) for p in portfolio.positions.values()),
        unavailable_positions=[m for m,p in portfolio.positions.items() if p['qty'] and m not in quotes])


def verify_frozen(fly, weights, head, initial_head, updates):
    if head.updates != updates or any(not head.torch.equal(v,initial_head[k]) for k,v in head.model.state_dict().items()):
        raise ValueError('Frozen readout changed')
    if fly is not None and not np.array_equal(weights,fly.controller.brain.weight):
        raise ValueError('Frozen native weights changed')


def audit_fills(opening, rows, name):
    """Reconstruct every fill against its preceding intent and available cash."""
    p = Portfolio(opening['portfolios'][name]); pending = {}; fills = 0
    for row in rows:
        if row.get('decision'):
            # Intents in this row are issued AFTER its executions.
            new = row['decision'][name]
        else:new = None
        for e in row['executions'][name]:
            f=e['fill'];m=e['mint']
            if f['status']!='filled':continue
            intent=pending.pop((m,f['decision_ts']))
            t=Tick(**e['tick'])
            quotes={k:Tick(**v) for k,v in row['marks'].items()}
            actual=p.execute(m,intent['target'],intent['issued'],t,quotes,
                             liquidity_notional_usd=e['liquidity_notional_usd'])
            if actual != f:raise ValueError('Forward fill reconstruction differs')
            fills+=1
        if new:pending[(new['mint'],new['issued'])]=new
        reported=Portfolio(row['portfolios'][name])
        # Quote/feature construction may create zero positions with no fill.
        for m in reported.positions:p.position(m)
        if p.state()!=reported.state():raise ValueError('Continuous portfolio reconstruction differs')
        equity=p.equity({m:Tick(**v) for m,v in row['marks'].items()})
        if abs(equity-row['metrics'][name]['marked_equity_usd'])>1e-7:
            raise ValueError('Forward mark reconstruction differs')
    return dict(fills=fills,rows=len(rows),cash_flows_verified=True,paper_only=True)


def run(root, data, manifest, saved, *, seconds=900, commit=lambda:None,
        clock=time.time, sleep=time.sleep, feed_factory=AllObservedFeed, fly_factory=None,
        fx_fetch=None, head_factory=Readout):
    if not 60 <= seconds <= 900:raise ValueError('Forward sessions are bounded to 15 minutes')
    validate_state(saved)
    for key in ('native','head'):
        if digest(manifest[key]['path']) != manifest[key]['sha256']:
            raise ValueError('Selected checkpoint hash changed')
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if (root/'started.json').exists():raise ValueError('Never repeat a forward session')
    started=clock();deadline=min(started+seconds,manifest['ends_at'])
    if deadline-started<60:raise ValueError('Experiment has ended')
    atomic_json(root/'opening.json',saved)
    atomic_json(root/'started.json',dict(at=started,deadline=deadline,protocol=PROTOCOL,
        checkpoint=manifest['checkpoint'],paper_only=True,native_learning=False,readout_learning=False,
        epsilon=0,interval_seconds=5,reset_cash=False,news_enabled=False,
        previous_observation_at=saved['last_at'],gap_seconds=started-saved['last_at'] if saved['last_at'] else None,
        transient_neural_state='reset at session boundary; learned weights preserved',
        pending_orders='expire at session boundary; never filled retrospectively'))
    head=head_factory();head.restore(manifest['head']['path']);updates=head.updates
    initial_head={k:v.clone() for k,v in head.model.state_dict().items()}
    accounts={k:Portfolio(v) for k,v in saved['portfolios'].items()}
    watched=dict(saved['watched']);active=dict(watched);visits=dict(saved['visits']);inactive={}
    contexts={m:dict(ticks=[Tick(**t) for t in c['ticks']],signature=c['signature']) for m,c in saved['contexts'].items()}
    pending={k:{} for k in accounts};anchors={};current=None;burst=0
    fly=None;weights=None;rows=[];observations=0;fills=dict(saved['fills']);status='completed'
    feed=feed_factory(root)
    for c in watched.values():feed.accept(c)
    for c in saved['pools']:feed.accept(c)
    feed.pinned_mints={m for p in accounts.values() for m,v in p.positions.items() if v['qty']}
    feed.start();commit()
    next_step=started;next_fx=0;fx=0.;fx_seen=0.;quote_usd={};quotes={};views={};last_at=started
    def fetch(symbol):
        r=requests.get('https://api.coinbase.com/v2/prices/'+symbol+'-USD/spot',timeout=4)
        r.raise_for_status();return float(r.json()['data']['amount'])
    fx_fetch=fx_fetch or fetch
    ledger=open(root/'decisions.jsonl','a')
    try:
        while clock()<deadline:
            now=clock()
            if now<next_step:sleep(min(1,next_step-now));continue
            next_step=now+5
            if now>=next_fx:
                next_fx=now+60
                try:
                    fx=fx_fetch('SOL');usdc=fx_fetch('USDC');fx_seen=clock()
                    if not np.isfinite([fx,usdc]).all() or min(fx,usdc)<=0:raise ValueError('Invalid FX')
                    quote_usd={USDC:usdc}
                except Exception:fx=0.;quote_usd={}
            if hasattr(feed,'snapshot_at'):snapshots,now,health=feed.snapshot_at(clock)
            else:snapshots=feed.snapshot();now=clock();health=feed.health()
            if health['status'] in ('failed','capacity_stopped'):
                status=health['status'];break
            connected=health['status']=='connected' and 0<=now-health['last_message']<=10
            snapshots=prepare_snapshots(snapshots,now,fx,fx_seen,quote_usd)
            views={m:quote_for(t,now,fx,fx_seen,selected=m in watched,connected=connected,unrestricted=True)
                   for m,t in snapshots.items()}
            quotes={m:q.observed_tick for m,q in views.items() if q.observed_tick}
            admissions,removed=update_all_active(active,watched,set(),inactive,accounts['trained'],
                                                snapshots,quotes,now,connected=connected)
            for m in removed:
                for orders in pending.values():orders.pop(m,None)
                anchors.pop(m,None)
            executions={k:[] for k in accounts};fresh=[]
            for m in active:
                c=contexts.setdefault(m,dict(ticks=[],signature=None));t=quotes.get(m)
                for orders in pending.values():
                    if m in orders and now-orders[m]['issued']>15:orders.pop(m)
                if not t:anchors.pop(m,None);continue
                trade=snapshots[m]['trades'][-1];signature=[trade['signature'],trade['log_index']]
                if c['signature']==signature:continue
                c['signature']=signature
                for name,p in accounts.items():
                    order=pending[name].get(m)
                    if not order:continue
                    et,reason=views[m].for_target(order['target'],p.cash,p.position(m)['qty'])
                    liquidity=trade['real_sol_reserves']/1e9*fx*.01
                    fill=(p.execute(m,order['target'],order['issued'],et,quotes,liquidity_notional_usd=liquidity)
                          if et else dict(status='rejected',reason=reason))
                    executions[name].append(dict(mint=m,tick=(et or t).__dict__,fill=fill,liquidity_notional_usd=liquidity))
                    fills[name]+=fill['status']=='filled'
                    if fill.get('reason')!='not_after_decision':pending[name].pop(m,None)
                c['ticks'].append(t);c['ticks']=c['ticks'][-100:]
                if len(c['ticks'])>=12 and m not in pending['trained']:fresh.append(m)
            picked,next_burst=choose(fresh,current,burst,visits);decision=None;diagnostics=None
            if picked and deadline-clock()>20:
                m=picked;c=contexts[m];t=quotes[m];p=accounts['trained']
                if fly is None:
                    if fly_factory is None:
                        import modal
                        if modal.is_local():raise RuntimeError('Full native fly must run in Modal')
                        from paperlab.fly import Fly
                        fly_factory=Fly
                    fly=fly_factory(data,learning=False,checkpoint=manifest['native']['path'])
                    weights=fly.controller.brain.weight.copy()
                    fly.controller.brain.reset(keep_memory=False);fly.controller.brain.weight[:]=weights
                if clock()-t.received_at<=10:
                    contribution=p.contribution(m,t);prev=anchors.get(m)
                    continuous=prev is not None and 0<t.ts-prev[0]<=60
                    if current!=m or not continuous:fly.controller.brain.reset(keep_memory=True)
                    feedback=contribution-prev[1] if continuous and m==current else 0.
                    neural=fly.observe(c['ticks'],len(c['ticks'])-1,News(enabled=False),feedback)
                    if neural['learning_diagnostics']['weight_delta_l2']!=0:raise ValueError('Native weights changed')
                    x=inputs(c['ticks'],snapshots[m],p.view(m),neural,now);x[10]=np.clip(contribution/25,-5,5)
                    q=head.values(x)
                    if not np.isfinite(q).all():raise ValueError('Nonfinite prediction')
                    action=max(p.allowed_actions(m),key=lambda a:q[a]);issued=clock()
                    if issued-t.received_at<=15:
                        decision={}
                        for name,account in accounts.items():
                            a=action if name=='trained' else 1
                            d=dict(mint=m,action=a,issued=issued,target=account.target(m,a,t))
                            decision[name]=d
                            if m not in pending[name]:pending[name][m]=d
                        diagnostics=dict(q_values=q.tolist(),input_features=x.tolist(),epsilon=0,
                            native_weight_delta_l2=0.,native_learning=False,readout_learning=False,
                            inherited_backprop_updates=updates,neural=neural)
                    anchors[m]=(t.ts,contribution);visits[m]=saved['steps']+len(rows)+1
                    current=m;burst=next_burst;observations+=1
            last_at=clock()
            row=dict(at=last_at,observation_at=now,paper_only=True,protocol=PROTOCOL,
                event_cursor=health.get('event_cursor'),feed=health,fx_state=dict(sol_usd=fx,seen_at=fx_seen,quote_usd=quote_usd),
                decision=decision,diagnostics=diagnostics,executions=executions,
                portfolios={k:p.state() for k,p in accounts.items()},
                metrics={k:account_values(p,quotes,views) for k,p in accounts.items()},
                marks={m:t.__dict__ for m,t in quotes.items() if any(p.positions.get(m,{}).get('qty') for p in accounts.values())},
                coverage=dict(observed_tokens=len(snapshots),observable_tokens=len(quotes),active_tokens=len(active),
                    ready_for_native=len(fresh),all_tokens_guaranteed=False,
                    unavailable_reasons=dict(Counter(q.observation_reason for q in views.values() if not q.observed_tick))),
                observations=saved['observations']+observations,fills=fills.copy(),cash_baseline_usd=1000.)
            rows.append(row);ledger.write(json.dumps(row,allow_nan=False)+'\n');ledger.flush()
            atomic_json(root/'latest.json',row)
            if len(rows)%12==0:
                print(json.dumps(dict(event='frozen_forward',at=last_at,observations=row['observations'],
                    fills=fills,metrics=row['metrics'],coverage=row['coverage'],weights_frozen=True)),flush=True)
                commit()
    finally:
        feed.close();ledger.close()
    if not rows:raise ValueError('No forward observations')
    verify_frozen(fly,weights,head,initial_head,updates)
    audits={name:audit_fills(saved,rows,name) for name in accounts}
    held={m for p in accounts.values() for m,v in p.positions.items() if v['qty']}
    # Retain holdings plus currently served markets. Raw archives retain other discoveries.
    keep=set(active)|held
    continuation=dict(portfolios={k:p.state() for k,p in accounts.items()},
        watched={m:watched[m] for m in keep if m in watched},
        pools=[v for v in getattr(feed,'pool_info',{}).values() if v.get('base_mint') in keep],
        contexts={m:dict(ticks=[t.__dict__ for t in c['ticks']],signature=c['signature']) for m,c in contexts.items() if m in keep},
        visits={m:v for m,v in visits.items() if m in keep},steps=saved['steps']+len(rows),
        observations=saved['observations']+observations,fills=fills,last_at=last_at)
    validate_state(continuation)
    atomic_json(root/'continuation.json',continuation)
    result=dict(status=status,started=started,ended=clock(),checkpoint=manifest['checkpoint'],
        protocol=PROTOCOL,paper_only=True,weights_unchanged=True,new_backprop_updates=0,
        inherited_backprop_updates=updates,observations=observations,cumulative_observations=continuation['observations'],
        fills=fills,metrics=rows[-1]['metrics'],account_audits=audits,
        pending_orders_expired_at_boundary=sum(len(p) for p in pending.values()),
        continuation_sha256=digest(root/'continuation.json'),profitable_learning_proven=False)
    atomic_json(root/'result.json',result);commit();return result
