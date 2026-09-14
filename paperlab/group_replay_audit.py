"""Read-only Decimal ledger verification for the isolated group replay artifacts."""
from decimal import Decimal
import json
import math
from pathlib import Path

import numpy as np

from .core import digest


def near(actual, expected, label):
    if not math.isfinite(float(actual)) or not math.isclose(float(actual),float(expected),rel_tol=1e-9,abs_tol=1e-8):
        raise ValueError(f'Invalid replay {label}: {actual} != {expected}')


def verify_trajectory(trajectory, episode, costs):
    rows, frames = trajectory['rows'], episode['frames']
    if len(rows)!=len(frames):
        raise ValueError('Trajectory horizon mismatch')
    cash, qty, fees = Decimal(str(costs['capital'])), Decimal(0), Decimal(0)
    fee_rate = Decimal(str(costs['fee_bps']/10000))
    exit_factor = (1-costs['fee_bps']/10000)*(1-costs['slippage_bps']/10000)
    peak, drawdown, exposures, fills = float(cash), 0., [], 0
    decisions = {r['now']:r for r in rows if not r['terminal']}
    used_decisions = set()
    for index,(row,frame) in enumerate(zip(rows,frames)):
        near(row['now'],frame['now'],'frame time')
        t=frame['tick'];fill=row['fill'];mid=(t['ask']+t['bid'])/2
        if fill['status']=='filled':
            issued=fill['decision_ts']
            if (not t['available'] or not issued<t['received_at']<=row['now']
                    or row['now']-issued>costs['max_delay'] or issued in used_decisions):
                raise ValueError('Noncausal, duplicate or unavailable fill')
            if issued not in decisions or decisions[issued]['action']==0:
                raise ValueError('Fill without a model order')
            used_decisions.add(issued)
            near(fill['fill_ts'],t['ts'],'fill receipt')
            buy=fill['side']=='BUY'
            if fill['side'] not in ('BUY','SELL'):
                raise ValueError('Invalid side')
            price=t['ask']*(1+costs['slippage_bps']/10000) if buy else t['bid']*(1-costs['slippage_bps']/10000)
            near(fill['price'],price,'execution price')
            amount=Decimal(fill['quantity']);notional=amount*Decimal(fill['price'])
            if amount<=0 or not 1-1e-8<=float(notional)<=costs['max_order']+1e-8:
                raise ValueError('Invalid order size')
            fee=notional*fee_rate
            near(fill['fee'],fee,'fee')
            cash += -notional-fee if buy else notional-fee
            qty += amount if buy else -amount
            fees += fee;fills+=1
        if cash<0 or qty<0:
            raise ValueError('Negative balance')
        for name,expected in (('cash',cash),('qty',qty),('fees',fees)):
            near(row['state'][name],expected,name)
        equity=float(cash+qty*Decimal(str(t['bid']*exit_factor))) if t['available'] else float(cash)
        near(row['equity'],equity,'liquidation equity')
        peak=max(peak,equity);drawdown=max(drawdown,peak-equity)
        exposures.append(float(qty)*mid)
        if index<len(rows)-1:
            action=row['action'];probs=row['old_probs']
            if not row['mask'][action] or any(p<0 for p in probs):
                raise ValueError('Illegal sampled action')
            if not math.isclose(sum(probs),1.,abs_tol=1e-6):
                raise ValueError('Invalid probability normalization')
            near(row['old_log_prob'],math.log(probs[action]),'behavior log probability')
            for feature,expected in ((8,float(qty)*mid/2.5),(9,float(cash)/1000),(10,(equity-1000)/2.5)):
                if not math.isclose(row['x'][feature],float(np.clip(expected,-5,5)),abs_tol=1e-6):
                    raise ValueError('Invalid branch account feature')
    pnl=rows[-1]['equity']-costs['capital']
    penalty=.001*float(np.mean(exposures))
    for name,expected in (('pnl_usd',pnl),('drawdown_usd',drawdown),('exposure_penalty_usd',penalty),
                          ('reward',pnl-.25*drawdown-penalty),('fees_usd',fees),('fills',fills)):
        near(trajectory[name],expected,name)
    return dict(rows=len(rows),fills=fills)


def audit(output):
    output=Path(output)
    manifest=json.loads((output/'manifest.json').read_text())
    completed=json.loads((output/'completed.json').read_text())
    features=json.loads((output/'features.json').read_text())
    if digest(output/'manifest.json')!=completed['manifest_sha256']:
        raise ValueError('Manifest hash mismatch')
    train={e['mint']:e for e in features['train']};test={e['mint']:e for e in features['test']}
    if train.keys() & test.keys():
        raise ValueError('Mint split overlap')
    if max(e['end'] for e in train.values())>=min(e['start'] for e in test.values()):
        raise ValueError('Time split overlap')
    for split,episodes in (('train',train),('test',test)):
        if manifest['cohorts'][split]!=[dict(mint=e['mint'],start=e['start'],end=e['end']) for e in episodes.values()]:
            raise ValueError('Manifest cohort mismatch')
        for episode in episodes.values():
            for frame in episode['frames']:
                event=frame.get('neural')
                if event and (event['stimulus']!='none' or event['learning_diagnostics']['changed_this_step']!=0):
                    raise ValueError('Counterfactual cache has reward or plasticity')
    counts=dict(trajectories=0,rows=0,fills=0)
    evaluation=[];groups={}
    for filename in ('training-trajectories.jsonl','evaluation-trajectories.jsonl'):
        with (output/filename).open() as stream:
            for line in stream:
                trajectory=json.loads(line)
                episodes=test if trajectory.get('split')=='test' else train
                result=verify_trajectory(trajectory,episodes[trajectory['mint']],manifest['costs'])
                counts['trajectories']+=1;counts['rows']+=result['rows'];counts['fills']+=result['fills']
                if 'group' in trajectory:
                    groups.setdefault(trajectory['group'],[]).append(trajectory)
                else:evaluation.append(trajectory)
    if len(groups)!=completed['groups'] or any(len(g)!=12 for g in groups.values()):
        raise ValueError('Incomplete training group')
    for summary in completed['evaluation']:
        rows=[r for r in evaluation if r['split']==summary['split'] and r['policy']==summary['policy']]
        for name,field in (('mean_pnl_usd','pnl_usd'),('mean_reward','reward'),('mean_fees_usd','fees_usd'),('mean_fills','fills')):
            near(summary[name],np.mean([r[field] for r in rows]),'evaluation '+name)
    updates=json.loads((output/'updates.json').read_text())
    actual=[u for u in updates if u['status']=='updated']
    near(len(actual),completed['gradient_updates'],'gradient update count')
    for u in actual:
        for key in ('loss','policy_loss','entropy','kl','gradient_norm','weight_delta_l2','weight_delta_max'):
            if not math.isfinite(u[key]):raise ValueError('Nonfinite update')
        if u['gradient_norm']<0 or u['weight_delta_l2']<0:raise ValueError('Negative norm')
    import torch
    initial=torch.load(output/'actor-initial.pt',weights_only=True,map_location='cpu')
    trained=torch.load(output/'actor-trained.pt',weights_only=True,map_location='cpu')['model']
    if initial.keys()!=trained.keys():raise ValueError('Actor parameter mismatch')
    change=torch.cat([(trained[k]-initial[k]).flatten() for k in initial])
    near(float(change.norm()),completed['actor_weight_delta_l2'],'checkpoint weight delta')
    near(sum(v.numel() for v in trained.values()),completed['actor_parameters'],'actor parameter count')
    return dict(verified=True,paper_only=True,**counts,training_groups=len(groups),
                gradient_updates=len(actual),held_out_mints=len(test),
                artifact_hashes={p.name:digest(p) for p in output.iterdir() if p.is_file() and p.name!='audit.json'},
                note='Verifies recorded accounting and diagnostics, not executable prices or profitable learning.')


if __name__=='__main__':
    import sys
    from .core import atomic_json
    result=audit(sys.argv[1])
    atomic_json(Path(sys.argv[1])/'audit.json',result)
    print(json.dumps(result,indent=2))
