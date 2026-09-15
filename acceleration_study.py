"""Audited training-only replay comparison and full-native synthetic speed study."""
from pathlib import Path
import json
import math
import time
import statistics
import copy
from paperlab.core import atomic_json,digest
from paperlab.checkpoint_eval import read,verify_checkpoint
from paperlab.solana_online import Portfolio
from paperlab.solana_online_audit import audit
from replay_readout import ReplayReadout


def extract_transitions(opening,rows):
    """Recover exact logged credit transitions; each reward is admitted once."""
    verified=audit(opening,rows)
    if not verified['reward_reconciliation_verified']:raise ValueError('Require audited terminal reward conservation')
    previous={};out=[]
    for row in rows:
        n=row.get('neural');metrics=list(row.get('reward_settlements',[]))
        if n and n.get('head_training'):metrics.append(n['head_training'])
        for h in metrics:
            m=h['mint']
            if m not in previous:raise ValueError('Missing preceding decision')
            old=previous.pop(m)
            x=old['x'] if h['terminal'] else n['input_features']
            allowed=Portfolio(row['portfolio']).allowed_actions(m)
            out.append(dict(previous=old,next_x=x,reward_usd=h['reward_usd'],elapsed=h['transition_seconds'],
                            allowed=list(allowed),terminal=h['terminal'],mint=m,observed_at=row['observation_at']))
        if row.get('decision'):
            d=row['decision'];previous[d['mint']]=dict(x=n['input_features'],action=d['action'],ts=n['tick']['ts'])
    if len(out)!=verified['new_head_updates'] or abs(sum(t['reward_usd'] for t in out)-verified['raw_q_reward_usd'])>1e-6:
        raise ValueError('Transition extraction lost or repeated economic reward')
    return out


def select_data(state,now):
    """Exclude every sealed evaluation tape and any active prospective window."""
    state=Path(state);plans=[read(p) for p in (state/'checkpoint-eval').glob('evaluation-*/plan.json')]
    protected={p['tape']['id'] for p in plans if p.get('tape')}
    control=read(state/'checkpoint-eval/control.json')
    current=read(state/'checkpoint-eval'/control['batch']/'plan.json') if control.get('batch') else None
    cutoff=min(now,current['cutoff']) if current else now
    chosen=[]
    for path in sorted((state/'solana-live').glob('solana-online-*/completed.json')):
        d=read(path)
        if (path.parent.name not in protected and d.get('status')=='completed' and d.get('all_observed')
                and d.get('ended',math.inf)<cutoff and d.get('account_audit',{}).get('reward_reconciliation_verified')):
            chosen.append(dict(id=path.parent.name,started=d['started'],ended=d['ended'],
                               files={n:dict(path=str(path.parent/n),sha256=digest(path.parent/n))
                                      for n in ('opening.json','decisions.jsonl','completed.json')}))
    chosen.sort(key=lambda x:x['started'])
    if len(chosen)<3:raise ValueError('Need at least two training episodes and one later development episode')
    return dict(cutoff=cutoff,protected_tapes=sorted(protected),training=chosen[:-1],development=chosen[-1])


def replay_comparison(state,out,plan,commit):
    state=Path(state);out=Path(out);train=[];dev=[]
    for subset,episodes in [(train,plan['training']),(dev,[plan['development']])]:
        for episode in episodes:
            verify_checkpoint(episode)
            rows=[json.loads(s) for s in Path(episode['files']['decisions.jsonl']['path']).read_text().splitlines()]
            subset.extend(extract_transitions(read(episode['files']['opening.json']['path']),rows))
    summaries=[]
    for ratio in (1,4):
        arm=out/f'replay-{ratio}x';arm.mkdir()
        head=ReplayReadout(ratio=ratio);head.restore(plan['head_checkpoint']['path'])
        reference_target=copy.deepcopy(head.target)
        initial_steps=head.optimizer_steps;start=time.perf_counter();metrics=[]
        for i,t in enumerate(train):
            metrics.append(head.update(t['previous'],t['next_x'],t['reward_usd'],t['elapsed'],tuple(t['allowed']),terminal=t['terminal']))
        train_seconds=time.perf_counter()-start
        before_steps=head.optimizer_steps
        dev_transitions=[head.transition(t['previous'],t['next_x'],t['reward_usd'],t['elapsed'],tuple(t['allowed']),t['terminal']) for t in dev]
        development_loss=head.td_loss(dev_transitions,reference_target)
        if head.optimizer_steps!=before_steps:raise ValueError('Development scoring updated weights')
        head.save(arm/'head-final.pt')
        (arm/'optimizer.jsonl').write_text(''.join(json.dumps(m)+'\n' for m in metrics))
        result=dict(ratio=ratio,training_transitions=len(train),development_transitions=len(dev),
                    optimizer_steps=head.optimizer_steps-initial_steps,target_syncs=head.target_syncs,
                    replay_size=len(head.buffer),training_seconds=train_seconds,development_td_loss=development_loss,
                    final_training_loss=metrics[-1]['loss'],training_reward_usd=sum(t['reward_usd'] for t in train),
                    checkpoint_sha256=digest(arm/'head-final.pt'),development_updates=0,
                    scope='Chronological logged-transition TD diagnostic against a common frozen initial target; not a policy PnL or profitability test')
        atomic_json(arm/'completed.json',result);summaries.append(result);commit()
    return summaries


class SyntheticFeed:
    """Matched 24-token stream for compute/correctness only; no market claims."""
    def __init__(self,root,clock=time.time):
        import threading
        self.root=Path(root);self.clock=clock;self.base=clock();self.pinned_mints=set();self.lock=threading.Lock()
    def accept(self,event):pass
    def start(self):pass
    def close(self):pass
    def health(self):return dict(status='connected',event_cursor=int((self.clock()-self.base)*100),synthetic=True,last_message=self.clock())
    def snapshot_at(self,clock=time.time):
        from paperlab.solana_events import SOL,TOKEN
        now=clock();step=int((now-self.base)/2.5);receipt=self.base+step*2.5
        snapshots={}
        for i in range(24):
            mint=f'synthetic-{i:02d}'
            created=dict(mint=mint,quote_mint=SOL,token_program=TOKEN,is_mayhem_mode=False,
                         timestamp=self.base-120,received=self.base-120,creation_time_known=True)
            t=dict(kind='TradeEvent',mint=mint,quote_mint=SOL,mayhem_mode=False,venue='pump',
                   received=receipt,timestamp=int(receipt),slot=step,signature=f'{mint}-{step}',log_index=0,
                   virtual_sol_reserves=60_000_000_000,real_sol_reserves=30_000_000_000,
                   virtual_token_reserves=500_000_000_000_000,sol_amount=10_000_000,is_buy=step%2==0)
            snapshots[mint]=dict(created=created,trades=[t],complete=False)
        return snapshots,now,self.health()


def native_benchmark(state,out,plan,commit):
    import modal
    if modal.is_local():raise RuntimeError('Native fly benchmark must run in Modal')
    from paperlab.fly import Fly
    from accelerated_online import run
    state=Path(state);out=Path(out);results=[]
    for ratio,interval in [(1,5),(4,5),(4,2.5)]:
        # Identical restored native state, loaded before the timed window.
        start=time.perf_counter();fly=Fly('/state/fly-data',learning=True,checkpoint=plan['native_checkpoint']['path'])
        init_seconds=time.perf_counter()-start
        arm=out/f'native-{ratio}x-{interval}s'
        result=run(arm,'/state/fly-data',state/'solana-live'/plan['parent'],seconds=180,commit=commit,
                   fly_factory=lambda *a,**kw:fly,feed_factory=SyntheticFeed,fx_fetch=lambda:100.,quote_fx_fetch=lambda:1.,
                   account_mode='fresh_training_episode',all_observed=True,interval_seconds=interval,
                   head_factory=lambda:ReplayReadout(ratio=ratio))
        rows=[json.loads(s) for s in (arm/'decisions.jsonl').read_text().splitlines()]
        verified=audit(read(arm/'opening.json'),rows)
        native=[r['neural'] for r in rows if r.get('neural')]
        metrics=[n['head_training'] for n in native if n.get('head_training')]+result['terminal_reward_settlements']
        if result['new_optimizer_steps']!=ratio*len(metrics):raise ValueError('Replay optimizer count differs')
        if result['status']!='completed' or not verified['reward_reconciliation_verified']:raise ValueError('Native benchmark audit failed')
        record=dict(ratio=ratio,interval_seconds=interval,native_observations=len(native),
                    distinct_inferred_tokens=verified['distinct_tokens_with_native_inference'],
                    transitions=len(metrics),optimizer_steps=result['new_optimizer_steps'],fills=result['fills'],
                    elapsed=result['ended']-result['started'],initialization_seconds=init_seconds,
                    median_native_seconds=statistics.median(n['compute_seconds'] for n in native) if native else None,
                    reward_reconciliation_verified=True,paper_equity=result['equity_stress_usd'],
                    synthetic=True,profitable_learning_proven=False)
        atomic_json(arm/'completed.json',dict(result,account_audit=verified,benchmark=record));results.append(record);commit()
        del fly
    return results
