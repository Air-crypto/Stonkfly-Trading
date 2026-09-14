"""Independently reconstruct recorded multi-token paper fills; never propagate a brain."""
from decimal import Decimal
import json
from pathlib import Path
from .solana_paper import COSTS


def audit(opening, rows):
    s=opening['portfolio'];cash=Decimal(s['cash']);fees=Decimal(s['fees'])
    episodic=opening.get('account_mode')=='fresh_training_episode'
    if episodic and (cash!=1000 or fees!=0 or s['positions'] or s['halted']
                    or s.get('risk_policy')!='fresh_training_full_cash_v1'):
        raise ValueError('Training episode must start with fresh paper capital and empty inventory')
    qty={m:Decimal(p['qty']) for m,p in s['positions'].items()}
    flow={m:Decimal(p['cash_flow']) for m,p in s['positions'].items()}
    basis={m:Decimal(p['basis']) for m,p in s['positions'].items()}
    issued={};filled=set();fills=0;native=0;head=0;marks=0;nonzero=0;mints=set();last=0.
    for r in rows:
        if episodic and (r.get('account_mode')!='fresh_training_episode'
                         or r.get('risk_policy')!='fresh_training_full_cash_v1'):
            raise ValueError('Training account mode changed within episode')
        if not r['paper_only'] or r['at']<=last:raise ValueError('Invalid paper chronology')
        last=r['at']
        for e in r['executions']:
            f=e['fill'];m=e['mint'];t=e['tick']
            if f['status']!='filled':continue
            key=(m,f['decision_ts'])
            if key not in issued or key in filled:raise ValueError('Unissued or reused decision')
            if not f['simulation'] or not t['available'] or not 0<f['fill_ts']-f['decision_ts']<=15:
                raise ValueError('Unavailable or noncausal fill')
            if t['ts']!=f['fill_ts']:raise ValueError('Fill timestamp differs from receipt')
            n=Decimal(f['quantity']);price=Decimal(f['price']);fee=Decimal(f['fee']);buy=f['side']=='BUY'
            expected=t['ask']*1.01 if buy else t['bid']*.99
            if abs(float(price)/expected-1)>1e-12:raise ValueError('Fill price differs')
            cap=Decimal('1000.000000001') if episodic else Decimal('25.000000001')
            if n<=0 or n*price>cap:raise ValueError('Order cap differs')
            if episodic:
                liquidity=Decimal(str(e['real_sol_reserves']))/Decimal('1e9')*Decimal(str(e['sol_usd']))*Decimal('.01')
                if liquidity<=0 or n*price>liquidity+Decimal('1e-7'):
                    raise ValueError('Observed liquidity allowance exceeded')
            if abs(fee-n*price*Decimal('.0125'))>Decimal('1e-9'):raise ValueError('Fee differs')
            delta=-n*price-fee if buy else n*price-fee
            if buy and episodic:
                allowance=max(Decimal(0),min(Decimal(str(opening['episode']['max_token_acquisition_usd']))-basis.get(m,0),cash))
                if -delta>allowance+Decimal('1e-8'):raise ValueError('Training cash or token allowance exceeded')
            if buy and r.get('risk_policy')=='quarantined_inventory_cash_floor_v1':
                quarantined=set(r['portfolio'].get('quarantined', []))
                active_basis=sum(b for k,b in basis.items() if k not in quarantined)
                allowance=max(Decimal(0),min(Decimal('2.50')-basis.get(m,0),Decimal(100)-active_basis,cash-902))
                if -delta>allowance+Decimal('1e-8'):raise ValueError('Entry risk allowance exceeded')
            basis[m]=basis.get(m,Decimal(0))-delta if buy else basis[m]*(qty[m]-n)/qty[m]
            cash+=delta;fees+=fee;qty[m]=qty.get(m,Decimal(0))+(n if buy else -n)
            flow[m]=flow.get(m,Decimal(0))+delta;filled.add(key);fills+=1
        p=r['portfolio']
        if abs(cash-Decimal(p['cash']))>Decimal('1e-7') or abs(fees-Decimal(p['fees']))>Decimal('1e-7'):
            raise ValueError('Shared cash or fees do not reconstruct')
        if any(q and m not in p['positions'] for m,q in qty.items()):
            raise ValueError('Held inventory disappeared')
        for m,position in p['positions'].items():
            if abs(Decimal(position['qty'])-qty.get(m,Decimal(0)))>Decimal('1e-7'):
                raise ValueError('Inventory does not reconstruct')
            if abs(Decimal(position['cash_flow'])-flow.get(m,Decimal(0)))>Decimal('1e-7'):
                raise ValueError('Per-token cash flow does not reconstruct')
            if abs(Decimal(position['basis'])-basis.get(m,Decimal(0)))>Decimal('1e-7'):
                raise ValueError('Acquisition basis does not reconstruct')
        if cash<0 or any(q<0 for q in qty.values()):raise ValueError('Negative inventory or cash')
        if 'marks' in r:
            equity=float(cash)
            for m,t in r['marks'].items():
                if not t['available']:raise ValueError('Unavailable mark')
                equity+=float(qty.get(m,0))*t['bid']*.99*.9875
            if abs(equity-r['equity_stress_usd'])>1e-7:raise ValueError('Equity mark differs')
            marks+=1
        n=r.get('neural')
        if n:
            native+=1;mints.add(n['mint']);head+=n.get('head_training') is not None
            if n.get('head_training'):
                import math
                h=n['head_training']
                if not all(math.isfinite(h[k]) for k in ('loss','gradient_l2_before_clip','weight_delta_l2','reward_usd')):
                    raise ValueError('Nonfinite learning diagnostics')
                nonzero+=abs(h['reward_usd'])>1e-12
        if r.get('decision'):
            d=r['decision'];issued[(d['mint'],d['issued'])]=d
    return dict(rows=len(rows),paper_fills=fills,native_observations=native,
        distinct_tokens_with_native_inference=len(mints),new_head_updates=head,
        cash_usd=float(cash),cumulative_fees_usd=float(fees),equity_marks_verified=marks,
        nonzero_reward_updates=nonzero,
        paper_only=True,profitable_learning_proven=False,
        note='Recorded-fill accounting only; this does not establish executable prices or policy superiority.')


def main():
    import argparse
    from .core import atomic_json
    parser=argparse.ArgumentParser();parser.add_argument('root');args=parser.parse_args();root=Path(args.root)
    result=audit(json.loads((root/'opening.json').read_text()),
                 [json.loads(s) for s in (root/'decisions.jsonl').read_text().splitlines()])
    atomic_json(root/'account-audit.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
