"""Read-only accounting and native-array checks for a recorded live pilot."""
from decimal import Decimal
import hashlib
import json
from pathlib import Path

import numpy as np

from .solana_paper import COSTS


def audit(rows, *, native_root=None, complete=False, opening=None):
    opening=opening or {'cash':'1000','qty':'0','fees':'0'}
    cash=Decimal(opening['cash']);qty=Decimal(opening['qty']);fees=Decimal(opening['fees']);fills=0;observations=0;updates=0;native_checks=0
    decisions=set();elapsed=[];update_norms=[];last_ts=0.
    for row in rows:
        if row['at']<=last_ts: raise ValueError('Nonmonotonic ledger')
        last_ts=row['at'];f=row['fill'];t=row['tick']
        if f['status']=='filled':
            if not t or not t['available'] or not row['quote_available']: raise ValueError('Fill lacks available quote')
            if not 0<f['fill_ts']-f['decision_ts']<=COSTS.max_delay: raise ValueError('Noncausal or expired fill')
            if f['decision_ts'] in decisions: raise ValueError('Repeated filled decision')
            decisions.add(f['decision_ts']);buy=f['side']=='BUY'
            expected=t['ask']*(1+COSTS.slippage_bps/10000) if buy else t['bid']*(1-COSTS.slippage_bps/10000)
            if not np.isclose(float(f['price']),expected,rtol=1e-12,atol=0):raise ValueError('Execution price differs')
            q=Decimal(f['quantity']);p=Decimal(f['price']);fee=Decimal(f['fee']);notional=q*p
            if abs(float(fee-notional*Decimal(str(COSTS.fee_bps/10000))))>1e-9:raise ValueError('Fee differs')
            if q<=0 or notional>Decimal(str(COSTS.max_order))+Decimal('1e-8'):raise ValueError('Order cap differs')
            cash+=-notional-fee if buy else notional-fee;qty+=q if buy else -q;fees+=fee;fills+=1
        state=row['broker']
        for key,value in (('cash',cash),('qty',qty),('fees',fees)):
            if abs(float(Decimal(state[key])-value))>1e-7:raise ValueError('Account does not reconstruct: '+key)
        if cash<0 or qty<0:raise ValueError('Negative cash or inventory')
        equity=float(cash)
        if t and row['quote_available']:
            equity+=float(qty)*t['bid']*(1-COSTS.slippage_bps/10000)*(1-COSTS.fee_bps/10000)
        if abs(equity-row['equity_stress_usd'])>1e-7:raise ValueError('Stress mark does not reconstruct')
        n=row.get('neural')
        if n:
            observations+=1;elapsed.append(n['compute_seconds']);d=n['learning_diagnostics'];update_norms.append(d['weight_delta_l2'])
            if n.get('head_training'):
                updates+=1
                if not np.isfinite([n['head_training'][k] for k in ('loss','gradient_l2_before_clip','weight_delta_l2')]).all():
                    raise ValueError('Invalid head training diagnostics')
            if native_root is not None:
                p=Path(native_root)/f'{observations-1:05d}.npz'
                with np.load(p,allow_pickle=False) as a:
                    change=a['after']-a['before']
                    if not np.array_equal(change,a['delta']):raise ValueError('Native deltas differ')
                    if hashlib.sha256(a['counts'].tobytes()).hexdigest()!=n['spike_sha256']:raise ValueError('Spike digest differs')
                    if int(a['counts'].sum())!=n['total_spikes']:raise ValueError('Spike totals differ')
                    if int(np.count_nonzero(change))!=d['changed_this_step']:raise ValueError('Changed-edge count differs')
                    if not np.isclose(np.linalg.norm(change),d['weight_delta_l2'],rtol=1e-6):raise ValueError('Update norm differs')
                    native_checks+=1
    intervals=np.diff([r['at'] for r in rows])
    return {'status':'completed_pilot_audited' if complete else 'published_prefix_audited',
        'rows':len(rows),'fills':fills,'neural_observations':observations,'readout_updates':updates,
        'native_arrays_verified':native_checks,'nonzero_native_updates':sum(n>0 for n in update_norms),
        'end_equity_stress_usd':rows[-1]['equity_stress_usd'] if rows else 1000.,
        'fees_usd':float(fees),'median_published_step_interval_seconds':float(np.median(intervals)) if len(intervals) else None,
        'median_native_compute_seconds':float(np.median(elapsed)) if elapsed else None,
        'last_at':last_ts,'paper_only':True,'profitable_learning_proven':False,
        'note':'Independent fixed-fill account reconstruction. Native-array checks do not validate reward causality, model superiority or executable prices.'}


def main():
    import argparse
    from .core import atomic_json
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--native',action='store_true');p.add_argument('--out',required=True);p.add_argument('--plot');a=p.parse_args()
    root=Path(a.root);rows=[json.loads(s) for s in (root/'decisions.jsonl').read_text().splitlines() if s.strip()]
    opening=json.loads((root/'opening.json').read_text())['broker'] if (root/'opening.json').exists() else None
    report=audit(rows,native_root=root/'neural' if a.native else None,complete=(root/'completed.json').exists(),opening=opening)
    atomic_json(a.out,report);print(json.dumps(report,indent=2))
    if a.plot: render(rows,a.plot)


def render(rows,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    start=rows[0]['at'];minutes=[(r['at']-start)/60 for r in rows]
    native=[r for r in rows if r.get('neural')];x=[(r['at']-start)/60 for r in native]
    learned=[r for r in native if r['neural'].get('head_training')];lx=[(r['at']-start)/60 for r in learned]
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    ax=axes[0,0]
    for key,label,color in [('equity_stress_usd','Fly + learned readout','#155ca2'),('one_entry_hold_equity_usd','One entry, hold','#b45a14')]:
        ax.plot(minutes,[r[key] for r in rows],label=label,color=color)
    ax.axhline(1000,color='#777777',linestyle=':',label='Cash');ax.set_ylabel('Paper liquidation equity ($)');ax.legend(fontsize=8)
    axes[0,1].plot(x,[r['neural']['learning_diagnostics']['weight_delta_l2'] for r in native],color='#155ca2')
    axes[0,1].set_ylabel('Native plastic-weight change L2')
    axes[1,0].plot(lx,[r['neural']['head_training']['loss'] for r in learned],label='Huber loss',color='#155ca2')
    axes[1,0].plot(lx,[r['neural']['head_training']['gradient_l2_before_clip'] for r in learned],label='Gradient L2 before clipping',color='#b45a14')
    axes[1,0].legend(fontsize=8);axes[1,0].set_ylabel('Decision-layer training diagnostics')
    axes[1,1].plot(minutes,[r['step_seconds'] for r in rows],color='#155ca2',label='Recorded processing time')
    axes[1,1].axhline(5,color='#b45a14',linestyle='--',label='5-second target');axes[1,1].set_ylabel('Seconds');axes[1,1].legend(fontsize=8)
    for ax in axes.flat:ax.set_xlabel('Minutes since first collector step');ax.grid(alpha=.2)
    fig.suptitle('Solana live training prefix · one token · indicative paper execution\nOnline training diagnostics; profitable learning has not been established',fontsize=13)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=150);plt.close(fig)


if __name__=='__main__':main()
