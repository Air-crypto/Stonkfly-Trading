"""Read-only comparison of archived account PnL and the rewards actually learned.

This never loads a model, submits a cloud job, or changes any historical ledger.
"""
import argparse
import hashlib
import json
from pathlib import Path


def analyze(root):
    episodes=[]
    for path in sorted(Path(root).glob('solana-*/decisions.jsonl')):
        rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not rows or rows[-1].get('account_mode')!='fresh_training_episode':
            raise ValueError('Only independent fresh training episodes may be summed')
        updates=[];native=[]
        for row in rows:
            n=row.get('neural')
            if n:
                native.append(n['learning_diagnostics']['equity_reward_usd'])
                if n.get('head_training'):updates.append(n['head_training'])
            updates.extend(row.get('reward_settlements',[]))
        pnl=rows[-1]['equity_stress_usd']-1000.
        raw=sum(h['reward_usd'] for h in updates)
        episodes.append(dict(run_id=path.parent.name,
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            reward_protocol=rows[-1].get('reward_protocol','legacy_unclosed_credit'),
            marked_pnl_usd=pnl,raw_q_reward_usd=raw,uncredited_difference_usd=pnl-raw,
            q_updates=len(updates),clipped_updates=sum(abs(h['reward_usd'])>25 for h in updates),
            scaled_reward_sum=sum(h['reward_scaled'] for h in updates),
            native_reward_usd=sum(native),
            terminal_updates=sum(bool(h.get('terminal')) for h in updates)))
    if not episodes:raise ValueError('No episode decision ledgers found')
    keys=('marked_pnl_usd','raw_q_reward_usd','uncredited_difference_usd','q_updates',
          'clipped_updates','scaled_reward_sum','native_reward_usd','terminal_updates')
    totals={k:sum(e[k] for e in episodes) for k in keys}
    totals['reward_clipped_fraction']=totals['clipped_updates']/max(1,totals['q_updates'])
    return dict(episodes=episodes,episode_count=len(episodes),totals=totals,
        account_semantics='Independent reset accounts, not a compounded portfolio.',
        reward_semantics='Raw dollars are compared before clipping and discounting; source records remain unchanged.',
        native_semantics='Native feedback was restricted to uninterrupted same-mint bursts; it is not a complete reward ledger.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();result=analyze(args.root)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result['totals'],indent=2))
