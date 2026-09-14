"""Immutable checkpoints and prospective, frozen-policy paper comparisons."""
from pathlib import Path
import json
import time
from .core import atomic_json, digest

PROTOCOL = 'frozen_common_opportunities_v2'


def source_fingerprint(root):
    """Seal every imported experiment/native source, schemas and image recipe."""
    root = Path(root)
    paths = {root / name for name in ('cloud.py', 'solana_cloud.py', 'checkpoint_eval_cloud.py')}
    for directory in ('paperlab', 'vendor/stonkfly/stonkfly'):
        paths.update(p for p in (root / directory).rglob('*')
                     if p.is_file() and p.suffix in ('.py', '.cpp', '.h', '.hpp', '.json'))
    return {str(p.relative_to(root)): digest(p) for p in sorted(paths)}


def read(path):
    return json.loads(Path(path).read_text())


def register(state):
    state=Path(state); dest=state/'checkpoint-eval'/'checkpoints';dest.mkdir(parents=True,exist_ok=True)
    for completed in sorted((state/'solana-live').glob('solana-online-*/completed.json')):
        run=completed.parent; result=read(completed)
        if result.get('status')!='completed' or result.get('account_mode')!='fresh_training_episode':continue
        out=dest/(run.name+'.json')
        if out.exists():continue
        paths={n:run/n for n in ['fly-final.npz','head-final.pt','completed.json']}
        reused = not all(paths[n].exists() for n in ('fly-final.npz', 'head-final.pt'))
        if reused:
            # A healthy no-observation window carries its parent's checkpoints.
            # Preserve that provenance instead of crashing all future registration.
            saved=read(run/'online-state.json')
            if result.get('neural_observations', 0):
                raise ValueError('A trained episode is missing its final checkpoint')
            paths.update({'fly-final.npz':Path(saved['native_checkpoint']),
                          'head-final.pt':Path(saved['head_checkpoint']),
                          'online-state.json':run/'online-state.json'})
        files={n:dict(path=str(path),sha256=digest(path)) for n,path in paths.items()}
        atomic_json(out,dict(id=run.name,ended=result['ended'],updates=result['readout_updates'],files=files,
                             checkpoint_reused=reused,retain=True,paper_only=True))
    return [read(p) for p in sorted(dest.glob('*.json'))]


def verify_checkpoint(checkpoint):
    for entry in checkpoint['files'].values():
        if digest(entry['path'])!=entry['sha256']:raise ValueError('Retained checkpoint changed')


def seal_plan(state,checkpoints,now):
    from .solana_quotes import QUOTE_PROTOCOL
    eligible=[c for c in checkpoints if c['ended']<now]
    if not eligible:raise ValueError('No completed checkpoints')
    return dict(id='evaluation-'+str(int(now)),created=now,cutoff=now,
                checkpoints=eligible,policies=['untrained','cash','always_long']+[c['id'] for c in eligible],
                protocol=PROTOCOL,quote_protocol=QUOTE_PROTOCOL,tape=None,
                scope='Conditional buy/sell evaluation at identical recorded attention opportunities; not universe-selection evaluation.',
                feature_scope='Frozen deployment mode: native weight-change features are zero, unlike online training. No claim of identical training feature distribution.',
                timing='Inputs stop at the recorded applied-event cursor and observation_at; orders use the recorded later issuance time, preserving common inference latency.',
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
    if plan.get('protocol')==PROTOCOL:
        rows=[json.loads(line) for line in (root/'decisions.jsonl').read_text().splitlines()]
        validate_observation_rows(rows)
    return dict(id=root.name,started=read(root/'completed.json')['started'],
                files={n:dict(path=str(root/n),sha256=digest(root/n)) for n in ['events.db','fx.jsonl','decisions.jsonl','completed.json']})


def validate_observation_rows(rows):
    import math
    from .solana_quotes import QUOTE_PROTOCOL
    if not rows:raise ValueError('Evaluation tape has no observations')
    previous_at=previous_observed=-float('inf');previous_cursor=0
    for index,row in enumerate(rows):
        at=row['at'];observed=row.get('observation_at')
        if not isinstance(observed,(int,float)) or not math.isfinite(observed):
            raise ValueError('Evaluation requires ordered recorded observation cutoffs')
        cursor=row.get('event_cursor');fx_state=row.get('fx_state',{})
        if type(cursor) is not int or cursor<previous_cursor:
            raise ValueError('Evaluation requires an ordered applied-event cursor')
        if (not all(isinstance(fx_state.get(k),(float,int)) and math.isfinite(fx_state[k])
                    for k in ('sol_usd','seen_at')) or not 0<=fx_state['sol_usd']<1e7
                or not 0<=fx_state['seen_at']<=observed):
            raise ValueError('Evaluation requires recorded FX availability')
        if row.get('quote_protocol')!=QUOTE_PROTOCOL:
            raise ValueError('Evaluation requires the sealed observation/entry/exit quote protocol')
        terminal=row.get('terminal',False)
        if terminal and (index!=len(rows)-1 or index==0 or observed!=previous_observed or cursor!=previous_cursor
                         or row.get('decision') or row.get('neural') or row.get('executions')):
            raise ValueError('Invalid terminal settlement row')
        if (not isinstance(observed,(int,float)) or not math.isfinite(observed)
                or not math.isfinite(at) or not previous_at<at
                or not (previous_observed<=observed<=at if terminal else previous_observed<observed<=at)):
            raise ValueError('Evaluation requires ordered recorded observation cutoffs')
        decision=row.get('decision');neural=row.get('neural')
        if decision and (not neural or decision['mint']!=neural['mint']
                         or not observed<=decision['issued']<=at):
            raise ValueError('Invalid common decision timing')
        if neural and neural.get('tick') and neural['tick']['received_at']>observed:
            raise ValueError('Source observed a receipt after its observation cutoff')
        previous_at,previous_observed,previous_cursor=at,observed,cursor


def evaluate(plan,policy,output,data,commit=lambda:None,fly_factory=None):
    import bisect,sqlite3,math,numpy as np
    from .solana_events import Feed
    from .solana_online import Portfolio,EPISODE_POLICY
    from .solana_paper import Readout,inputs
    from .solana_quotes import quote_for,QUOTE_PROTOCOL
    from .news import News
    if not plan['tape'] or plan['tape']['started']<=plan['cutoff']:raise ValueError('Not a future evaluation tape')
    if plan.get('protocol')!=PROTOCOL:raise ValueError('Reseal evaluation under the current protocol; never rewrite an old plan')
    if plan.get('quote_protocol')!=QUOTE_PROTOCOL:raise ValueError('Unsealed quote protocol')
    if policy not in plan['policies']:raise ValueError('Unsealed policy')
    tape=plan['tape'];verify_checkpoint(tape)
    source_rows=[json.loads(x) for x in Path(tape['files']['decisions.jsonl']['path']).read_text().splitlines()]
    validate_observation_rows(source_rows)
    fx=[json.loads(x) for x in Path(tape['files']['fx.jsonl']['path']).read_text().splitlines()]
    fxt=[x['at'] for x in fx]
    if (any(not math.isfinite(x['at']) or not math.isfinite(x['sol_usd']) or x['sol_usd']<=0 for x in fx)
            or any(a>=b for a,b in zip(fxt,fxt[1:]))):
        raise ValueError('Invalid or unordered contemporaneous FX')
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
    portfolio=Portfolio(opening['portfolio']);pending={};ticks={};seen={};current=None;ledger=[];anchors={};active=set()
    coverage=dict(recorded_neural_opportunities=0,recorded_issued_opportunities=0,policy_decisions=0,
                  skipped_no_source_decision=0,skipped_unavailable_quote=0,skipped_insufficient_history=0,
                  skipped_pending_order=0,expired_pending_orders=0)
    db=sqlite3.connect('file:'+tape['files']['events.db']['path']+'?mode=ro',uri=True)
    maximum_cursor=db.execute('select coalesce(max(rowid),0) from events').fetchone()[0]
    if source_rows[-1]['event_cursor']>maximum_cursor:raise ValueError('Recorded event cursor exceeds the archived tape')
    iterator=iter(db.execute('select rowid,body from events order by rowid'));event=next(iterator,None)
    feed=Feed(output);last_quotes={};peak=1000.;drawdown=0;count=0
    for source in source_rows:
        now=source['observation_at']
        if source.get('terminal'):
            # This is learning settlement metadata, not an additional market receipt.
            ledger.append({**ledger[-1], 'at':source['at'], 'terminal':True,
                           'decision':None,'policy_diagnostics':None,'executions':[]})
            continue
        while event:
            if event[0]>source['event_cursor']:break
            body=json.loads(event[1])
            if body['received']>now:raise ValueError('Applied event was received after its observation cutoff')
            feed.accept(body);event=next(iterator,None)
        i=bisect.bisect_right(fxt,now)-1;quotes={};quote_views={};snap=feed.snapshot()
        for m in source.get('retired',[]):
            active.discard(m)
            pending.pop(m,None)
        active.update(source.get('admissions',[]))
        health=source['feed'];connected=health['status']=='connected' and 0<=now-health['last_message']<=10
        live_fx=source['fx_state']
        if live_fx['sol_usd'] and (i<0 or fx[i]['at']!=live_fx['seen_at'] or fx[i]['sol_usd']!=live_fx['sol_usd']):
            raise ValueError('Recorded FX state does not match contemporaneous receipts')
        if i>=0 and live_fx['sol_usd'] and connected:
            for m,t in snap.items():
                view=quote_for(t,now,fx[i]['sol_usd'],fx[i]['at'],selected=m in active or m in pending)
                quote_views[m]=view
                if view.observed_tick:quotes[m]=view.observed_tick
        for m in active:
            if m not in quotes:anchors.pop(m,None)
        executions=[]
        for m,order in list(pending.items()):
            q=quotes.get(m)
            if not q:
                if now-order['issued']>15:
                    del pending[m];coverage['expired_pending_orders']+=1
                continue
            view=quote_views[m]
            execution_tick,execution_reason=view.for_target(order['target'],portfolio.cash,portfolio.position(m)['qty'])
            if execution_tick is None:
                fill=dict(status='rejected',reason=execution_reason)
                executions.append(dict(mint=m,tick=q.__dict__,fill=fill))
                del pending[m]
                continue
            trade=snap[m]['trades'][-1];liquidity=trade['real_sol_reserves']/1e9*fx[i]['sol_usd']*.01
            fill=portfolio.execute(m,order['target'],order['issued'],execution_tick,quotes,liquidity_notional_usd=liquidity)
            executions.append(dict(mint=m,tick=execution_tick.__dict__,fill=fill,liquidity_notional_usd=liquidity,real_sol_reserves=trade['real_sol_reserves'],sol_usd=fx[i]['sol_usd']))
            if fill.get('reason')!='not_after_decision':del pending[m]
        for m,q in quotes.items():
            if m not in active:continue
            trade=snap[m]['trades'][-1]
            signature=(trade.get('signature'),trade.get('log_index'),trade['received'])
            if signature!=seen.get(m):ticks.setdefault(m,[]).append(q);ticks[m]=ticks[m][-100:];seen[m]=signature
        opportunity=source.get('neural');decision=None
        diagnostics=None
        if opportunity:
            coverage['recorded_neural_opportunities']+=1
            m=opportunity['mint'];q=quotes.get(m)
            source_decision=source.get('decision')
            if source_decision:coverage['recorded_issued_opportunities']+=1
            if not source_decision:coverage['skipped_no_source_decision']+=1
            elif not q:coverage['skipped_unavailable_quote']+=1
            elif len(ticks.get(m,[]))<12:coverage['skipped_insufficient_history']+=1
            elif m in pending:coverage['skipped_pending_order']+=1
            if source_decision and q and len(ticks.get(m,[]))>=12 and m not in pending:
                if fly:
                    contribution=portfolio.contribution(m,q);previous=anchors.get(m)
                    continuous=previous is not None and 0<q.ts-previous[0]<=60
                    if m!=current or not continuous:fly.controller.brain.reset(keep_memory=True)
                    delta=contribution-previous[1] if continuous and m==current else 0.
                    neural=fly.observe(ticks[m],len(ticks[m])-1,News(enabled=False),delta)
                    if neural['learning_diagnostics']['weight_delta_l2']!=0:raise ValueError('Frozen native weights changed')
                    x=inputs(ticks[m],snap[m],portfolio.view(m),neural,now);x[10]=max(-5,min(5,contribution/25))
                    values=head.values(x)
                    if not np.isfinite(values).all():raise ValueError('Nonfinite Q values')
                    action=max(portfolio.allowed_actions(m),key=lambda a:values[a]);count+=1
                    anchors[m]=(q.ts,contribution);current=m
                    diagnostics=dict(input_features=x.tolist(),q_values=values.tolist(),epsilon=0,
                                     native_learning=False,readout_learning=False,reward_feedback_usd=delta,
                                     native_weight_delta_l2=neural['learning_diagnostics']['weight_delta_l2'])
                else:action=0 if policy=='cash' else 1
                decision=dict(mint=m,action=action,issued=source_decision['issued'],target=portfolio.target(m,action,q))
                pending[m]=decision
                coverage['policy_decisions']+=1
        equity=portfolio.equity(quotes);peak=max(peak,equity);drawdown=max(drawdown,peak-equity)
        valuations={m:quote_views[m].position_values(p['qty']) for m,p in portfolio.positions.items()
                    if p['qty'] and m in quote_views}
        capacity_equity=float(portfolio.cash)+sum(v['capacity_stress_value_usd'] for v in valuations.values())
        ledger.append(dict(at=source['at'],observation_at=now,paper_only=True,account_mode='fresh_training_episode',risk_policy=EPISODE_POLICY,executions=executions,decision=decision,neural=None,
                           event_cursor=source['event_cursor'],fx_state=live_fx,
                           policy_diagnostics=diagnostics,
                           quote_protocol=QUOTE_PROTOCOL,position_values=valuations,capacity_stress_equity_usd=capacity_equity,
                           portfolio=portfolio.state(),equity_stress_usd=equity,marks={m:q.__dict__ for m,q in quotes.items() if portfolio.positions.get(m,{}).get('qty')},retired=[]))
        last_quotes=quotes
        feed.pinned_mints=active|{m for m,p in source.get('portfolio',{}).get('positions',{}).items() if float(p['qty'])>0}
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
                scope=plan['scope'],feature_scope=plan.get('feature_scope'),protocol=PROTOCOL,quote_protocol=QUOTE_PROTOCOL,coverage=coverage,
                capacity_stress_equity_usd=capacity_equity,capacity_stress_pnl_usd=capacity_equity-1000,
                pending_orders_at_end=len(pending),portfolio=portfolio.state(),paper_only=True,profitable_learning_proven=False)
