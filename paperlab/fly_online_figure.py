"""Plot completed study 11 audits, preserving decisions, later fills and gaps."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np

from .core import atomic_json, digest
from .fly_market_study import signature
from .fly_online_cloud import validate_selection_receipt
from .fly_online_protocol import ARMS, HOSTING_ALLOCATION, chunk_name, chunk_order
from .fly_online_schedule import aggregate
from .fly_online_study import select_development
from .fly_paper_inputs import validate

LABELS = dict(zip(ARMS, ('Pristine / frozen', 'Trained / frozen', 'Trained / online', 'Trained / online + trace reset')))
COLORS = dict(zip(ARMS, ('#8fbcff', '#67e8cf', '#ffbf69', '#e6a8e8')))
PHASES = ('development', 'test')
CHECKS = ('all_ledgers_and_feedback_reconstructed', 'all_full_bin_audits_passed',
    'all_raw_prices_reconstructed', 'all_news_clocks_verified', 'source_and_native_builds_match',
    'selection_precedes_every_test_chunk')


def evidence(root):
    """Reconcile plotted values with hash-pinned summaries and independent audits.

    This consumes existing full audits; it does not replace raw-array auditing
    or run a neural model. Synthetic prices force a visible fixture label.
    """
    root=Path(root)
    read=lambda path:json.loads((root/path).read_text())
    report=read('report.json');envelope=read('plan.json');p=validate(envelope);r=p['registration']
    names=[chunk_name(*c) for c in chunk_order()]
    if (report.get('status') not in ('paper_online_study_audited', 'paper_online_recovery_audited',
            'paper_online_completion_audited') or report.get('audited') is not True
            or r['study']!='11' or report['registration']!=r or report['plan_sha256']!=envelope['sha256']
            or any(report.get('verification',{}).get(k) is not True for k in CHECKS)
            or report['verification'].get('distinct_completed_worker_calls')!=16
            or set(report['audit_sha256'])!=set(names) or set(report['chunk_sha256'])!=set(names)):
        raise ValueError('A complete independently audited study 11 report is required')
    prices=read('price-audit.json')
    if (prices!=report['price_audit'] or prices.get('prices_reconstructed_from_snapshot') is not True
            or prices['plan_sha256']!=envelope['sha256'] or prices['snapshot_sha256']!=p['snapshot_sha256']):
        raise ValueError('Price audit differs from the reported inputs')
    summaries={};receipts={};lanes={}
    for stage,pool,arm in chunk_order():
        name=chunk_name(stage,pool,arm);summary=read(f'chunks/{name}/summary.json')
        audit_path=Path('audits')/(name+'.json');a=read(audit_path)
        receipt=read(f'receipts/{name}-completed.json');receipts[name]=receipt
        if (digest(root/audit_path)!=report['audit_sha256'][name]
                or signature(summary)!=report['chunk_sha256'][name]
                or receipt['summary_sha256']!=digest(root/f'chunks/{name}/summary.json')
                or summary['chunk']!=name or summary['plan_sha256']!=envelope['sha256']
                or summary['registration']!=r or summary['arm']!=arm or summary['stage']!=stage or summary['pool_index']!=pool
                or a['status']!='paper_online_chunk_audited' or a['chunk']!=name or a['plan_sha256']!=envelope['sha256']
                or a['artifact_sha256']!=summary['artifact_sha256'] or a['executed_source_sha256']!=summary['code_sha256']):
            raise ValueError('Figure evidence differs from its completed chunk: '+name)
        outcome=summary['outcome'];rows=outcome['rows'];observed=a['observations']
        if outcome!=report['phase_diagnostics'][r['cohort'][pool]][arm][stage]:
            raise ValueError('Plotted outcome differs from the audited report')
        checks=a['verification']
        if (any(checks.get(k) is not True for k in ('ledger_replayed','feedback_reconstructed','all_boundaries_verified','all_weights_exact'))
                or checks['decision_slots']!=25 or checks['observations']!=len(observed) or checks['native_bins']!=50*len(observed)):
            raise ValueError('Incomplete chunk audit')
        if [row['decision_ts'] for row in rows]!=[r[stage+'_start']+300*i for i in range(25)]:
            raise ValueError('Incomplete or shifted figure timeline')
        items=[];cursor=0;fees=0;fills=0
        for slot,row in enumerate(rows):
            event=row['event'];fill=row['fill'];terminal=slot==24
            if row['terminal']!=terminal or (terminal and event is not None):raise ValueError('Terminal mark became a decision')
            item={'slot':slot,'decision_ts':row['decision_ts'],'quote_ts':row['quote_ts'],
                'equity':row['equity'],'available':row['available'],'terminal':terminal,'fill':fill,
                'side':None,'gate_spikes':None,'weight_delta_l2':None,'earlier_component_share_pct':None,
                'observation':None,'equity_reward_usd':None}
            if event is not None:
                if cursor>=len(observed):raise ValueError('Missing observation audit')
                audited=observed[cursor];cursor+=1
                if (audited['decision_ts']!=row['decision_ts'] or audited['side']!=event['side']
                        or audited['gate_spikes']!=event['gate_spikes']
                        or audited['weight_norm_audit']['reported']!=event['weight_delta_l2']
                        or audited['equity_reward_usd']!=event['equity_reward_usd']
                        or event['plasticity_enabled']!=ARMS[arm]['learning']):
                    raise ValueError('Displayed neural diagnostic differs from its audit')
                drive=audited['rule']['integrated_absolute_drive'];earlier,current=drive['earlier'],drive['current']
                if not all(math.isfinite(x) and x>=0 for x in (earlier,current,event['weight_delta_l2'],event['gate_spikes'])):
                    raise ValueError('Nonfinite or negative neural diagnostic')
                item.update(side=event['side'],gate_spikes=event['gate_spikes'],weight_delta_l2=event['weight_delta_l2'],
                    earlier_component_share_pct=100*earlier/(earlier+current) if earlier+current else None,
                    observation=cursor-1,equity_reward_usd=event['equity_reward_usd'])
            if fill['status']=='filled':
                origin=next((old for old in items if old['decision_ts']==fill['decision_ts']),None)
                # BUY means target 50% exposure; rebalancing to that target can
                # sell inventory after a price rise. Decision and fill sides differ.
                if (origin is None or origin['side'] not in ('BUY','SELL') or fill['fill_ts']<=fill['decision_ts']
                        or fill['fill_ts']!=row['quote_ts']):
                    raise ValueError('Fill does not belong to an earlier decision')
                item['fill_origin_side']=origin['side']
                fees+=float(fill['fee']);fills+=1
            if not math.isfinite(row['equity']) or abs(float(row['broker']['fees'])-fees)>1e-8:
                raise ValueError('Figure ledger does not reconcile')
            item['fees']=fees;items.append(item)
        if (cursor!=len(observed) or abs(outcome['equity']-items[-1]['equity'])>1e-8
                or abs(outcome['fees']-fees)>1e-8 or outcome['fills']!=fills):
            raise ValueError('Figure totals differ from the audited ledger')
        summaries[name]=summary;lanes[name]={'rows':items,'fills':fills,'fees':fees,'observations':cursor}
    selection=select_development(envelope,{k:v for k,v in summaries.items() if v['stage']=='development'})
    expected=aggregate(envelope,summaries,selection)
    if any(report[k]!=expected[k] for k in ('total_equity','observations','selection','chunk_sha256')):
        raise ValueError('Figure aggregate or development selection differs')
    validate_selection_receipt(selection,report['selection_receipt'],receipts)
    if len({v['call_id'] for v in receipts.values()})!=16:raise ValueError('Chunk worker calls are not distinct')
    if report['status'] in ('paper_online_recovery_audited', 'paper_online_completion_audited'):
        from .fly_recovery_report import verify
        verify(root,report,envelope,summaries,receipts)
    elif 'recovery' in report:
        raise ValueError('Amended recovery cannot be relabeled as the original completed study')
    panels={}
    for stage in PHASES:
        for arm in ARMS:
            pair=[lanes[chunk_name(stage,pool,arm)] for pool in range(2)]
            panels[stage,arm]={'equity':[500+sum(lane['rows'][i]['equity'] for lane in pair) for i in range(25)],
                'unavailable':[any(not lane['rows'][i]['available'] for lane in pair) for i in range(25)],
                'fills':sum(lane['fills'] for lane in pair),'fees':sum(lane['fees'] for lane in pair),
                'observations':sum(lane['observations'] for lane in pair)}
    fixture=any('synthetic' in t['source'] for series in p['series'].values() for t in series)
    return report,lanes,panels,fixture


def render(root, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    root,output=Path(root),Path(output)
    report,lanes,panels,fixture=evidence(root)
    if output.exists():raise ValueError('Preserve earlier figure output; choose a new directory')
    output.mkdir(parents=True)
    background,foreground,muted='#0e1728','#e8edf6','#b9c6da'
    stamp=lambda t:datetime.fromtimestamp(t,timezone.utc).strftime('%H:%M')
    currency=FuncFormatter(lambda v,_:f'${v:+.2f}')
    def save(fig,name):
        for ext in ('png','svg'):fig.savefig(output/(name+'.'+ext),dpi=130,facecolor=background)
        plt.close(fig)
    def decorate(ax):
        ax.set_facecolor('#172237');ax.grid(alpha=.12);ax.yaxis.set_major_locator(MaxNLocator(5))
    with plt.rc_context({'font.size':10,'font.family':'DejaVu Sans','text.color':foreground,'text.parse_math':False,
            'axes.labelcolor':foreground,'axes.titlecolor':foreground,'xtick.color':muted,
            'ytick.color':muted,'axes.edgecolor':'#52617a','svg.fonttype':'none'}):
        fig,axes=plt.subplots(2,2,figsize=(14,9),gridspec_kw={'height_ratios':[2,1]},facecolor=background)
        fig.subplots_adjust(left=.085,right=.975,top=.79,bottom=.18,hspace=.32,wspace=.2)
        title='SYNTHETIC FIXTURE — comparison layout only' if fixture else 'Does resetting learning traces improve paper trading?'
        fig.text(.04,.955,title,fontsize=20,weight='bold')
        amended=report['status'] in ('paper_online_recovery_audited', 'paper_online_completion_audited')
        fig.text(.04,.918,('Amended recovery · ' if amended else '')+'Four conditions · two pools · separate $1,000 development and test accounts',fontsize=12)
        handles=[];bounds=[v-1000 for panel in panels.values() for v in panel['equity']]+[0]
        pad=max(1,max(bounds)-min(bounds))*.15
        for j,stage in enumerate(PHASES):
            ax=axes[0,j];decorate(ax);start=report['registration'][stage+'_start']
            ax.set_title(stage.title()+' · '+stamp(start)+'–'+stamp(start+7200)+' UTC',loc='left',fontsize=14)
            table=[]
            for k,arm in enumerate(ARMS):
                panel=panels[stage,arm];y=np.array(panel['equity'])-1000
                line,=ax.plot(np.arange(25)*5,y,color=COLORS[arm],lw=2,ls='--' if k>=2 else '-',label=LABELS[arm])
                if j==0:handles.append(line)
                gaps=np.array(panel['unavailable']);ax.scatter(np.arange(25)[gaps]*5,y[gaps],marker='x',color='#ff727d',s=40)
                table.append([LABELS[arm].replace('Trained / ','Trained '),f"${panel['equity'][-1]:,.2f}",
                    str(panel['fills']),f"${panel['fees']:.2f}",f"{panel['observations']}/48"])
            ax.axhline(0,color=muted,ls=':',lw=1);ax.axhline(HOSTING_ALLOCATION,color='#b2d975',ls=':',lw=1)
            ax.set_ylim(min(bounds)-pad,max(bounds)+pad);ax.yaxis.set_major_formatter(currency)
            ax.set_xlabel('Minutes since this phase began');ax.set_ylabel('Paper P&L, including execution costs')
            axes[1,j].axis('off')
            tab=axes[1,j].table(cellText=table,colLabels=['Condition','End equity','Fills','Fees','Observed'],
                colWidths=[.43,.21,.09,.13,.14],bbox=[0,.05,1,.94],cellLoc='left')
            tab.auto_set_font_size(False);tab.set_fontsize(8.5)
            for (row,col),cell in tab.get_celld().items():
                cell.set_facecolor('#172237' if row else '#25334b');cell.set_edgecolor('#35445c');cell.set_text_props(color=foreground)
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,.886),ncols=2,frameon=False,labelcolor=foreground)
        selected=report['selection']['selected']
        fig.text(.04,.12,'Development gate: '+('candidate failed.' if selected is None else 'trace-reset candidate passed; no automatic promotion.'),fontsize=12)
        fig.text(.04,.085,'Red ×: missing quote; affected inventory valued at $0, not an observed crash. Green: cash + $0.1111 hosting.',color=muted)
        fig.text(.04,.055,'Test cannot change development selection. Indicative DEX fills; two pools and four hours do not establish monthly profitability.',color=muted)
        fig.text(.04,.025,('Synthetic diagnostics and prices; no market result. ' if fixture else '')+'Plan: '+report['plan_sha256'][:24]+'…',fontsize=9,color=muted)
        save(fig,'comparison')
        for stage in PHASES:
            for pool in range(2):
                fig,axes=plt.subplots(5,1,figsize=(14,13),sharex=True,
                    gridspec_kw={'height_ratios':[1.6,1,1,1,1]},facecolor=background)
                fig.subplots_adjust(left=.115,right=.975,top=.835,bottom=.135,hspace=.23)
                title=('SYNTHETIC FIXTURE · ' if fixture else 'Amended recovery · ' if amended else '')+f'Pool {pool} · {stage}: decisions, fills and learning'
                fig.text(.04,.967,title,fontsize=18,weight='bold')
                fig.text(.04,.935,'▲ / ▼: BUY / SELL fills processed at this slot. B / H / S: new BUY / HOLD / SELL decisions. ×: no neural observation.',color=muted)
                fig.text(.04,.913,'BUY targets 50% exposure and can rebalance by selling. Red ×: missing quote; inventory valued at $0, not an observed crash.',fontsize=9.5,color=muted)
                heat=[];handles=[]
                for k,arm in enumerate(ARMS):
                    rows=lanes[chunk_name(stage,pool,arm)]['rows'];xs=np.arange(25)*5
                    y=np.array([row['equity']-250 for row in rows])
                    line,=axes[0].plot(xs,y,color=COLORS[arm],lw=1.7,ls='--' if k>=2 else '-',label=LABELS[arm]);handles.append(line)
                    gaps=np.array([not row['available'] for row in rows])
                    axes[0].scatter(xs[gaps],y[gaps],marker='x',s=40,color='#ff727d',zorder=6)
                    for side,marker in (('BUY','^'),('SELL','v')):
                        mask=np.array([row['fill']['status']=='filled' and row['fill']['side']==side for row in rows])
                        axes[0].scatter(xs[mask],y[mask],marker=marker,s=45,color=COLORS[arm],edgecolor=background,zorder=5)
                    heat.append([{'SELL':0,'HOLD':1,'BUY':2,None:3}[row['side']] for row in rows[:-1]])
                    for ax,field in ((axes[2],'gate_spikes'),(axes[3],'weight_delta_l2'),(axes[4],'earlier_component_share_pct')):
                        if ax is axes[4] and not ARMS[arm]['learning']:continue
                        values=[np.nan if row[field] is None else row[field] for row in rows[:-1]]
                        ax.plot(xs[:-1],values,color=COLORS[arm],lw=1.5,marker='.',ls='--' if k>=2 else '-')
                axes[1].imshow(heat,cmap=ListedColormap(['#fd929d','#a4b1c3','#67e8cf','#28354a']),vmin=0,vmax=3,
                    aspect='auto',extent=(-2.5,117.5,3.5,-.5),interpolation='none')
                for k,row in enumerate(heat):
                    for i,value in enumerate(row):axes[1].text(i*5,k,('S','H','B','×')[value],ha='center',va='center',color='#101a2b' if value<3 else muted,fontsize=9)
                axes[1].set_yticks(range(4),['Pristine frozen','Trained frozen','Online carry','Online reset']);axes[1].grid(False)
                axes[1].set_facecolor('#172237')
                for i in (0,2,3,4):decorate(axes[i])
                axes[0].yaxis.set_major_formatter(currency);axes[0].axhline(0,color=muted,ls=':',lw=1)
                axes[0].set_ylabel('Paper P&L\nfrom $250')
                axes[2].set_ylabel('Gate spikes')
                axes[3].set_ylabel('Weight update\nL2 norm')
                axes[4].set_ylabel('Earlier-image\ncomponent share (%)');axes[4].set_ylim(-3,103)
                axes[-1].set_xticks(np.arange(0,121,10));axes[-1].set_xlabel('Minutes since phase start; 120 is the terminal execution/mark')
                fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.54,.897),ncols=2,frameon=False,labelcolor=foreground)
                fig.text(.04,.089,'Share = |earlier| / (|earlier| + |current|), integrated across every plastic edge and bin; algebraic components, not causal credit.',fontsize=9,color=muted)
                fig.text(.04,.065,'Gaps stay missing; no-drive shares are undefined. Weight movement is not an optimizer loss or backprop gradient.',fontsize=9,color=muted)
                fig.text(.04,.04,('Synthetic diagnostics and prices; no market result. ' if fixture else '')+report['registration']['cohort'][pool],fontsize=9,color=muted)
                save(fig,f'{stage}-pool{pool}')
    paired=[];links=['# Online trace comparison'+(' — synthetic fixture' if fixture else ''),'',
        'Synthetic diagnostics and prices; no market result.' if fixture else
            'Completed audited study 11 recordings'+(' from the disclosed recovery amendment.' if amended else '.'),
        '', 'Start the debugger with `--out <study-root>/views --port 8767` before opening these links.',
        'Each link matches the same observed market slot. Fills execute earlier decisions; a BUY target may cause a rebalance SELL.',
        '', '| Phase / pool | UTC | New decision: carry / reset | Gate spikes: carry / reset | Earlier fills: carry / reset | Paired trace |',
        '| --- | --- | --- | --- | --- | --- |']
    for stage in PHASES:
        for pool in range(2):
            carry=chunk_name(stage,pool,'trained_online_carry');reset=chunk_name(stage,pool,'trained_online_reset_rates')
            for left,right in zip(lanes[carry]['rows'][:-1],lanes[reset]['rows'][:-1]):
                if (left['decision_ts']!=right['decision_ts'] or (left['side'] is None)!=(right['side'] is None)):
                    raise ValueError('Pair coverage differs')
                if left['side'] is None:continue
                url=f"http://127.0.0.1:8767/?run={reset}&step={right['observation']}&bin=49&neuron=10527&compare={carry}"
                pair={'phase':stage,'pool':pool,'slot':left['slot'],'decision_ts':left['decision_ts'],
                    'carry_side':left['side'],'reset_side':right['side'],'decision_changed':left['side']!=right['side'],
                    'gate_delta':right['gate_spikes']-left['gate_spikes'],
                    'equity_delta':right['equity']-left['equity'],'viewer_url':url}
                paired.append(pair)
                fill=lambda row:row['fill'].get('side',row['fill']['status'])
                links.append(f"| {stage} / {pool} | {stamp(left['decision_ts'])} | {left['side']} / {right['side']} | "
                    f"{left['gate_spikes']} / {right['gate_spikes']} | {fill(left)} / {fill(right)} | [Inspect]({url}) |")
    (output/'diagnostics.md').write_text('\n'.join(links)+'\n')
    result={'status':'online_figures_rendered','validation_only':fixture,'report_sha256':digest(root/'report.json'),
        'source_sha256':digest(__file__),'lanes':lanes,
        'paired_steps':paired,'diagnostics_sha256':digest(output/'diagnostics.md'),
        'figure_sha256':{p.name:digest(p) for p in sorted(output.iterdir()) if p.suffix in ('.png','.svg')}}
    atomic_json(output/'figures.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();result=render(args.root,args.out)
    print(json.dumps({**{k:v for k,v in result.items() if k not in ('lanes','paired_steps')},
        'paired_observations':len(result['paired_steps'])},indent=2))


if __name__=='__main__':main()
