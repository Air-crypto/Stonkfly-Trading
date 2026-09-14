"""Immutable checkpoints and prospective, frozen-policy paper comparisons."""
from pathlib import Path
import json
import time
from .core import atomic_json, digest


def read(path):
    return json.loads(Path(path).read_text())


def register(state):
    state=Path(state); dest=state/'checkpoint-eval'/'checkpoints';dest.mkdir(parents=True,exist_ok=True)
    for completed in sorted((state/'solana-live').glob('solana-online-*/completed.json')):
        run=completed.parent; result=read(completed)
        if result.get('status')!='completed' or result.get('account_mode')!='fresh_training_episode':continue
        out=dest/(run.name+'.json')
        if out.exists():continue
        files={n:dict(path=str(run/n),sha256=digest(run/n)) for n in ['fly-final.npz','head-final.pt','completed.json']}
        atomic_json(out,dict(id=run.name,ended=result['ended'],updates=result['readout_updates'],files=files,
                             retain=True,paper_only=True))
    return [read(p) for p in sorted(dest.glob('*.json'))]


def verify_checkpoint(checkpoint):
    for entry in checkpoint['files'].values():
        if digest(entry['path'])!=entry['sha256']:raise ValueError('Retained checkpoint changed')


def seal_plan(state,checkpoints,now):
    eligible=[c for c in checkpoints if c['ended']<now]
    if not eligible:raise ValueError('No completed checkpoints')
    return dict(id='evaluation-'+str(int(now)),created=now,cutoff=now,
                checkpoints=eligible,policies=['untrained','cash','always_long']+[c['id'] for c in eligible],
                protocol='frozen_common_opportunities_v1',tape=None,
                scope='Conditional buy/sell evaluation at identical recorded attention opportunities; not universe-selection evaluation.',
                initial_cash=1000,epsilon=0,learning=False,paper_only=True)


def seal_tape(state,plan):
    state=Path(state)
    candidates=[]
    for p in (state/'solana-live').glob('solana-online-*/completed.json'):
        r=read(p)
        if r.get('status')=='completed' and r.get('started',0)>plan['cutoff']:
            candidates.append((r['started'],p.parent))
    if not candidates:return None
    _,root=min(candidates)
    return dict(id=root.name,started=read(root/'completed.json')['started'],
                files={n:dict(path=str(root/n),sha256=digest(root/n)) for n in ['events.db','fx.jsonl','decisions.jsonl','completed.json']})


def evaluate(plan,policy,output,data,commit=lambda:None,fly_factory=None):
    import bisect,sqlite3,numpy as np
    from .solana_events import Feed
    from .solana_online import Portfolio,EPISODE_POLICY
    from .solana_paper import Readout,inputs,tick_for
    from .news import News
    if not plan['tape'] or plan['tape']['started']<=plan['cutoff']:raise ValueError('Not a future evaluation tape')
    if policy not in plan['policies']:raise ValueError('Unsealed policy')
    tape=plan['tape'];verify_checkpoint(tape)
    cp=next((c for c in plan['checkpoints'] if c['id']==policy),None)
    if cp:
        if cp['ended']>=tape['started']:raise ValueError('Training overlaps evaluation')
        verify_checkpoint(cp)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    head=Readout()
    if cp:head.restore(cp['files']['head-final.pt']['path'])
    initial_head={k:v.clone() for k,v in head.model.state_dict().items()}
    initial_updates=head.updates
    fly=None
    if policy not in ('cash','always_long'):
        if fly_factory is None:
            import modal
            if modal.is_local():raise RuntimeError('Native inference must run in Modal')
            from .fly import Fly
            fly_factory=Fly
        fly=fly_factory(data,learning=False,checkpoint=cp['files']['fly-final.npz']['path'] if cp else None)
        # Standardize transient neural state without erasing learned synaptic weights.
        weights=fly.controller.brain.weight.copy()
        fly.controller.brain.reset(keep_memory=False);fly.controller.brain.weight[:]=weights
    else:weights=None
    if policy=='untrained':
        head.save(output/'head-untrained.pt');fly.save(output/'fly-untrained.npz')
    opening=dict(portfolio=dict(cash='1000',fees='0',halted=False,positions={},quarantined=[],risk_policy=EPISODE_POLICY))
    portfolio=Portfolio(opening['portfolio']);pending={};ticks={};seen={};current=None;ledger=[];anchors={}
    source_rows=[json.loads(x) for x in Path(tape['files']['decisions.jsonl']['path']).read_text().splitlines()]
    fx=[json.loads(x) for x in Path(tape['files']['fx.jsonl']['path']).read_text().splitlines()];fxt=[x['at'] for x in fx]
    db=sqlite3.connect('file:'+tape['files']['events.db']['path']+'?mode=ro',uri=True)
    iterator=iter(db.execute('select body from events order by received,log_index'));event=next(iterator,None)
    feed=Feed(output);last_quotes={};peak=1000.;drawdown=0;count=0
    for source in source_rows:
        now=source['at']
        while event:
            body=json.loads(event[0])
            if body['received']>now:break
            feed.accept(body);event=next(iterator,None)
        i=bisect.bisect_right(fxt,now)-1;quotes={};snap=feed.snapshot()
        health=source['feed'];connected=health['status']=='connected' and 0<=now-health['last_message']<=10
        if i>=0 and connected:
            for m,t in snap.items():
                q,reason=tick_for(t,now,fx[i]['sol_usd'],fx[i]['at'],selected=m in ticks or m in pending)
                if q and reason is None:quotes[m]=q
        executions=[]
        for m,order in list(pending.items()):
            if now-order['issued']>15:del pending[m];continue
            q=quotes.get(m)
            if not q:continue
            trade=snap[m]['trades'][-1];liquidity=trade['real_sol_reserves']/1e9*fx[i]['sol_usd']*.01
            fill=portfolio.execute(m,order['target'],order['issued'],q,quotes,liquidity_notional_usd=liquidity)
            executions.append(dict(mint=m,tick=q.__dict__,fill=fill,liquidity_notional_usd=liquidity,real_sol_reserves=trade['real_sol_reserves'],sol_usd=fx[i]['sol_usd']))
            if fill.get('reason')!='not_after_decision':del pending[m]
        for m,q in quotes.items():
            if q.ts!=seen.get(m):ticks.setdefault(m,[]).append(q);ticks[m]=ticks[m][-100:];seen[m]=q.ts
        opportunity=source.get('neural');decision=None
        if opportunity:
            m=opportunity['mint'];q=quotes.get(m)
            if q and len(ticks.get(m,[]))>=12 and m not in pending:
                if fly:
                    if m!=current:
                        fly.controller.brain.reset(keep_memory=True)
                    contribution=portfolio.contribution(m,q);previous=anchors.get(m)
                    delta=contribution-previous[1] if previous and 0<q.ts-previous[0]<=60 and m==current else 0.
                    neural=fly.observe(ticks[m],len(ticks[m])-1,News(enabled=False),delta)
                    if neural['learning_diagnostics']['weight_delta_l2']!=0:raise ValueError('Frozen native weights changed')
                    x=inputs(ticks[m],snap[m],portfolio.view(m),neural,now);x[10]=max(-5,min(5,contribution/25))
                    values=head.values(x)
                    if not np.isfinite(values).all():raise ValueError('Nonfinite Q values')
                    action=max(portfolio.allowed_actions(m),key=lambda a:values[a]);count+=1
                    anchors[m]=(q.ts,contribution);current=m
                else:action=0 if policy=='cash' else 1
                decision=dict(mint=m,action=action,issued=now,target=portfolio.target(m,action,q))
                pending[m]=decision
        equity=portfolio.equity(quotes);peak=max(peak,equity);drawdown=max(drawdown,peak-equity)
        ledger.append(dict(at=now,paper_only=True,account_mode='fresh_training_episode',risk_policy=EPISODE_POLICY,executions=executions,decision=decision,neural=None,
                           portfolio=portfolio.state(),equity_stress_usd=equity,marks={m:q.__dict__ for m,q in quotes.items() if portfolio.positions.get(m,{}).get('qty')},retired=[]))
        last_quotes=quotes
        if len(ledger)%30==0:atomic_json(output/'progress.json',dict(rows=len(ledger),total=len(source_rows),policy=policy));commit()
    db.close()
    if head.updates!=initial_updates or any(not head.torch.equal(v,initial_head[k]) for k,v in head.model.state_dict().items()):raise ValueError('Frozen Q changed')
    if fly and not np.array_equal(weights,fly.controller.brain.weight):raise ValueError('Frozen native checkpoint changed')
    from .solana_online_audit import audit
    opening.update(account_mode='fresh_training_episode',episode=dict(initial_cash_usd=1000,max_token_acquisition_usd=250))
    result_audit=audit(opening,ledger)
    (output/'decisions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in ledger))
    return dict(status='completed',policy=policy,tape=tape['id'],pnl_usd=portfolio.equity(last_quotes)-1000,fees_usd=float(portfolio.fees),
                max_drawdown_usd=drawdown,native_observations=count,account_audit=result_audit,weights_unchanged=True,
                scope=plan['scope'],portfolio=portfolio.state(),paper_only=True,profitable_learning_proven=False)
