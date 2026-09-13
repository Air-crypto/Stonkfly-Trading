"""Explain audited study 11 account losses using the unchanged recorded fills.

This is an accounting identity, not a zero-cost policy backtest or a new neural
trajectory. Changing costs would also change position sizing and learning.
"""
import argparse
from dataclasses import asdict
from decimal import Decimal
import json
from pathlib import Path

from .core import Costs, Tick, atomic_json, digest
from .fly_market_study import quote_at
from .fly_online_figure import evidence
from .fly_online_protocol import ARMS, chunk_name
from .fly_rate_metrics import replay_ledger

D = lambda value: Decimal(str(value))
COMPONENTS = ('gross_reference_pnl_usd', 'paid_fees_usd', 'execution_spread_usd',
              'execution_slippage_usd', 'exit_allowance_usd', 'missing_reference_deduction_usd', 'net_pnl_usd')
LABELS = ('Pristine frozen', 'Trained frozen', 'Online carry', 'Online reset histories')


def decompose(rows, ticks, costs):
    """Reconcile every cash/quantity/fee state and quote against a fixed-fill bridge."""
    cash = reference_cash = D(costs.capital); qty = fees = spread = slippage = D(0)
    result = []; fills = 0; turnover = D(0)
    fee_rate = D(costs.fee_bps/10000); slip = costs.slippage_bps/10000
    exit_factor = (1-costs.fee_bps/10000)*(1-slip)
    if not rows: raise ValueError('A complete account timeline is required')
    for row in rows:
        stamp = row['decision_ts']; index, quote = quote_at(ticks, stamp)
        if row['quote_ts'] != quote.ts or row['available'] != quote.available:
            raise ValueError('Accounting quote does not match the recorded timeline')
        usable = [t for t in ticks[:index+1] if t.available]
        if not usable: raise ValueError('Missing preceding usable price reference')
        reference = quote if quote.available else usable[-1]
        mid = D(quote.mid); fill = row['fill']
        if fill['status'] == 'filled':
            side = fill['side']; q, px, fee = (D(fill[k]) for k in ('quantity', 'price', 'fee'))
            if (side not in ('BUY','SELL') or any(not v.is_finite() for v in (q,px,fee))
                    or q <= 0 or px <= 0 or fee < 0 or not quote.available
                    or fill['fill_ts'] != quote.ts or fill['decision_ts'] >= fill['fill_ts']):
                raise ValueError('Invalid recorded fill')
            buy = side == 'BUY'; touch = D(quote.ask if buy else quote.bid)
            expected = D(quote.ask*(1+slip) if buy else quote.bid*(1-slip))
            if px != expected or abs(fee-q*px*fee_rate) > D('1e-12'):
                raise ValueError('Recorded execution costs differ from registered assumptions')
            if not buy and q > qty: raise ValueError('Recorded sale exceeds inventory')
            signed = q if buy else -q
            cash -= signed*px+fee; reference_cash -= signed*mid; qty += signed
            fees += fee; spread += q*(touch-mid if buy else mid-touch)
            slippage += q*(px-touch if buy else touch-px); fills += 1; turnover += q*px
        elif fill['status'] not in ('hold','rejected'):
            raise ValueError('Unknown fill status')
        state = row['broker']
        if any(D(state[k]) != v for k,v in (('cash',cash),('qty',qty),('fees',fees))):
            raise ValueError('Cash, quantity or paid fees do not reconstruct')
        reference_mid = D(reference.mid)
        gross = reference_cash+qty*reference_mid-D(costs.capital)
        exit_allowance = qty*(reference_mid-D(quote.bid*exit_factor)) if quote.available else D(0)
        missing = D(0) if quote.available else qty*reference_mid
        net = gross-spread-slippage-fees-exit_allowance-missing
        observed = D(row['equity'])-D(costs.capital)
        if not observed.is_finite() or abs(net-observed) > D('1e-8'):
            raise ValueError('P&L components do not reconcile with the account mark')
        result.append({'decision_ts':stamp,'quote_ts':quote.ts,'available':quote.available,
            'reference_ts':reference.ts,'reference_age_seconds':stamp-reference.ts,
            'reference_kind':'usable_indicative_midpoint' if quote.available else 'stale_reference_only_not_liquidatable',
            'gross_reference_pnl_usd':float(gross),'paid_fees_usd':float(fees),
            'execution_spread_usd':float(spread),'execution_slippage_usd':float(slippage),
            'exit_allowance_usd':float(exit_allowance),'missing_reference_deduction_usd':float(missing),
            'net_pnl_usd':float(net),'recorded_equity':row['equity'],'identity_error_usd':float(net-observed),
            'fills':fills,'turnover_usd':float(turnover),'inventory_quantity':str(qty)})
    return {'rows':result,'terminal':result[-1],
        'maximum_identity_error_usd':max(abs(r['identity_error_usd']) for r in result),
        'missing_marks':sum(not r['available'] for r in result),
        'maximum_missing_reference_deduction_usd':max(r['missing_reference_deduction_usd'] for r in result),
        'interpretation':'The midpoint component uses the same recorded quantities. When no usable quote exists, its stale reference value is explicitly deducted in full. It is not a realizable mark or a strategy evaluated with different costs.'}


def build(root, output):
    root, output = Path(root), Path(output)
    if output.exists(): raise ValueError('Preserve existing attribution evidence')
    report, _, panels, fixture = evidence(root)
    p = json.loads((root/'plan.json').read_text())['plan']; r = p['registration']; costs = Costs(**r['costs'])
    if asdict(costs) != r['costs']: raise ValueError('Registered execution costs differ')
    chunks = {}; totals = {}
    for stage in ('development','test'):
        totals[stage] = {}
        for arm, settings in ARMS.items():
            pair = []
            for pool, key in enumerate(r['cohort']):
                name = chunk_name(stage,pool,arm)
                summary = json.loads((root/'chunks'/name/'summary.json').read_text())
                replay_ledger(summary['outcome'],p['series'][key],r[stage+'_start'],settings)
                bridge = decompose(summary['outcome']['rows'],[Tick(**t) for t in p['series'][key]],costs)
                bridge.update(summary_sha256=digest(root/'chunks'/name/'summary.json'),
                              native_audit_sha256=digest(root/'audits'/(name+'.json')),pool=key)
                chunks[name] = bridge; pair.append(bridge)
            terminal = {field:sum(v['terminal'][field] for v in pair) for field in COMPONENTS}
            terminal.update(fills=sum(v['terminal']['fills'] for v in pair),
                turnover_usd=sum(v['terminal']['turnover_usd'] for v in pair),
                terminal_quotes_usable=all(v['terminal']['available'] for v in pair),
                missing_marks=sum(v['missing_marks'] for v in pair))
            terminal['total_friction_usd'] = sum(terminal[k] for k in (
                'paid_fees_usd','execution_spread_usd','execution_slippage_usd','exit_allowance_usd'))
            if abs(terminal['net_pnl_usd']-(panels[stage,arm]['equity'][-1]-1000)) > 1e-8:
                raise ValueError('Aggregate attribution differs from the audited report')
            totals[stage][arm] = terminal
    value = {'status':'audited_fixed_fill_loss_attribution','study':'11','validation_only':fixture,
        'source_report_sha256':digest(root/'report.json'),'plan_sha256':report['plan_sha256'],
        'costs':asdict(costs),'cohort':r['cohort'],'aggregate':totals,'chunks':chunks,
        'verification':{'original_sixteen_receipts_audits_and_selection_verified':True,
            'all_sixteen_ledgers_replayed':True,'decision_marks_reconciled':sum(len(v['rows']) for v in chunks.values()),
            'maximum_identity_error_usd':max(v['maximum_identity_error_usd'] for v in chunks.values()),
            'all_terminal_quotes_usable':all(v['terminal_quotes_usable'] for phase in totals.values() for v in phase.values())},
        'source_sha256':{name:digest(Path(__file__).with_name(name)) for name in (
            'fly_loss_attribution.py','core.py','fly_online_figure.py','fly_rate_metrics.py')},
        'new_neural_observations':0,'cloud_submissions':0,'policy_promoted':False,
        'interpretation':'Accounting decomposition of unchanged historical fills, not a different-cost backtest. The gross midpoint component is not the return a zero-cost strategy would have achieved. Hosting is excluded. Missing quotes can affect actions and learning even when terminal marks are usable. No model is selected from this diagnosis.'}
    output.mkdir(parents=True); atomic_json(output/'report.json',value); return value


def render(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    output = Path(output)
    background, foreground = '#101827','#edf3ff'
    names = list(ARMS); colors = ['#c496ff','#ffbf69','#ff827b','#78cbd5']
    components = ('paid_fees_usd','execution_spread_usd','execution_slippage_usd','exit_allowance_usd')
    captions = ('Paid fees','Spread at fills','Slippage at fills','Exit allowance')
    with plt.rc_context({'text.color':foreground,'axes.labelcolor':foreground,'xtick.color':foreground,
                         'ytick.color':foreground,'font.family':'DejaVu Sans','font.size':10,
                         'svg.fonttype':'none','text.parse_math':False}):
        fig,(left,right) = plt.subplots(1,2,figsize=(14,8),facecolor=background,gridspec_kw={'width_ratios':[1.4,1]})
        fig.subplots_adjust(left=.20,right=.975,top=.79,bottom=.25,wspace=.30)
        ys=[];labels=[]
        for block,stage in enumerate(('development','test')):
            for index,arm in enumerate(names):
                y=block*5+index;v=report['aggregate'][stage][arm];ys.append(y)
                labels.append(f'{"Dev" if block==0 else "Test"} · {LABELS[index]}')
                gross,net=v['gross_reference_pnl_usd'],v['net_pnl_usd']
                left.plot([net,gross],[y,y],color='#718199',linewidth=3)
                left.scatter(gross,y,color='#78cbd5',marker='D',s=48,zorder=3,label='Midpoint component' if y==0 else None)
                left.scatter(net,y,color='#ff827b',s=48,zorder=3,label='Recorded net P&L' if y==0 else None)
                left.annotate(f'{net:+.2f}',(net,y),xytext=(-5,9),textcoords='offset points',ha='right',fontsize=9)
                left.annotate(f'{gross:+.2f}',(gross,y),xytext=(5,-14),textcoords='offset points',ha='left',fontsize=9)
                start=0
                for component,color,label in zip(components,colors,captions):
                    amount=v[component];right.barh(y,amount,left=start,height=.58,color=color,label=label if y==0 else None)
                    start+=amount
                right.text(start+.25,y,f'${start:.2f} · {v["fills"]} fills',va='center',fontsize=9)
        for ax in (left,right):
            ax.set_facecolor('#182235');ax.set_ylim(9,-1);ax.grid(axis='x',alpha=.15)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value,_:f'${value:g}'))
            ax.spines[['top','right']].set_visible(False);ax.axhline(4,color='#53627b',linewidth=.8)
        left.set_yticks(ys,labels);right.set_yticks([])
        left.axvline(0,color='#98a6ba',linewidth=.8);left.set_xlim(-67,14);right.set_xlim(0,23)
        left.set_title('Price exposure and recorded result',pad=18);right.set_title('Friction deducted from that component',pad=18)
        left.set_xlabel('P&L relative to $1,000 starting capital');right.set_xlabel('Execution costs + terminal exit allowance')
        fig.legend(*left.get_legend_handles_labels(),loc='upper left',bbox_to_anchor=(.20,.17),frameon=False,ncol=2,fontsize=9)
        fig.legend(*right.get_legend_handles_labels(),loc='upper left',bbox_to_anchor=(.69,.17),frameon=False,ncol=2,fontsize=9)
        fig.suptitle('Study 11 loss decomposition',x=.035,ha='left',y=.96,fontsize=20)
        fig.text(.035,.90,'Same recorded fills and quantities · no new model run · all terminal quotes usable',fontsize=11,color='#bbc8df')
        fig.text(.035,.075,'Midpoint accounting is not a zero-cost strategy backtest. Removing costs would also change sizing and learning.',fontsize=10,color='#bbc8df')
        fig.text(.035,.04,'Exit allowance estimates liquidation spread, slippage and fees on remaining inventory. Hosting is excluded.',fontsize=10,color='#bbc8df')
        if report['validation_only']:fig.text(.035,.855,'SYNTHETIC VALIDATION FIXTURE',color='#ffbf69')
        for suffix in ('png','svg'):fig.savefig(output/f'loss-attribution.{suffix}',dpi=140,facecolor=background)
        plt.close(fig)
    atomic_json(output/'figure.json',{'report_sha256':digest(output/'report.json'),
        'figure_sha256':{suffix:digest(output/f'loss-attribution.{suffix}') for suffix in ('png','svg')},
        'source_sha256':digest(__file__),'cloud_submissions':0,'new_neural_observations':0})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--plot',action='store_true');a=p.parse_args();report=build(a.root,a.out)
    if a.plot:render(report,a.out)
    print(json.dumps({'status':report['status'],'aggregate':report['aggregate'],'verification':report['verification']},indent=2))


if __name__=='__main__':main()
