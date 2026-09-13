"""Plot the two already-audited development controls; no model execution."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as dates
import matplotlib.pyplot as plt


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chunks',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise ValueError('Keep previous figures and evidence')
    root=Path(__file__).resolve().parents[1]
    failure=json.loads((root/'reports/fly-online-failure-11.json').read_text())
    data=[]
    for i,pin in enumerate(failure['completed_controls']):
        folder=args.chunks/pin['chunk'];summary=folder/'summary.json'
        projection=root/'reports'/('fly-online-control-'+('first' if i==0 else 'second')+'-11.json')
        assert sha(summary)==pin['summary_sha256'] and sha(projection)==pin['audit_report_sha256']
        s=json.loads(summary.read_text());rows=s['outcome']['rows']
        assert len(rows)==25 and sum(r['event'] is not None for r in rows)==24
        assert s['outcome']['initial_memory_sha256']==s['outcome']['final_memory_sha256']
        data.append(s)
    assert [r['decision_ts'] for r in data[0]['outcome']['rows']]==[r['decision_ts'] for r in data[1]['outcome']['rows']]
    assert all(a['event']['input_sha256']==b['event']['input_sha256'] for a,b in
        zip(data[0]['outcome']['rows'][:-1],data[1]['outcome']['rows'][:-1]))
    plt.rcParams.update({'figure.facecolor':'#0d1523','axes.facecolor':'#152034','savefig.facecolor':'#0d1523',
        'text.color':'#edf1fc','axes.labelcolor':'#c5cfe0','xtick.color':'#c5cfe0','ytick.color':'#c5cfe0',
        'axes.edgecolor':'#435269','font.size':10})
    fig,ax=plt.subplots(3,1,figsize=(12,8),sharex=True,gridspec_kw={'height_ratios':[2,1,1]})
    series=[]
    for i,(s,color,label) in enumerate(zip(data,['#f0b769','#62dccb'],['Pristine, frozen','Previously trained, frozen'])):
        rows=s['outcome']['rows'];t=[datetime.fromtimestamp(r['decision_ts'],timezone.utc) for r in rows]
        equity=[r['equity'] for r in rows]
        exposure=[100*(r['equity']-float(r['broker']['cash']))/r['equity'] for r in rows]
        action=[r['event']['side'] for r in rows[:-1]]
        ax[0].plot(t,equity,color=color,lw=2,label=f'{label}: ${equity[-1]:.2f}')
        ax[1].plot(t,exposure,color=color,lw=1.8)
        for side,marker in [('BUY','^'),('HOLD','o'),('SELL','v')]:
            selected=[t[j] for j,a in enumerate(action) if a==side]
            ax[2].scatter(selected,[1-i]*len(selected),color=color,marker=marker,s=45 if side!='HOLD' else 14,
                label=side if i==0 else None)
        series.append({'chunk':s['chunk'],'times_utc':[x.isoformat() for x in t], 'equity':equity,
            'inventory_percent_of_equity':exposure,'decisions':action})
    for a in ax:
        a.grid(alpha=.13);a.spines[['top','right']].set_visible(False)
    ax[0].axhline(250,color='#bac4d2',ls='--',lw=1,label='Starting cash: $250')
    ax[0].set_ylabel('Paper equity ($)');ax[0].legend(loc='lower left',frameon=False,fontsize=9)
    ax[1].set_ylabel('Inventory / equity (%)');ax[1].set_ylim(0,60)
    ax[2].set_yticks([0,1],['Trained','Pristine']);ax[2].set_ylim(-.6,1.6)
    ax[2].set_ylabel('Fixed decoder');ax[2].legend(loc='upper right',frameon=False,ncol=3,fontsize=9)
    ax[2].xaxis.set_major_formatter(dates.DateFormatter('%H:%M',tz=timezone.utc))
    ax[2].set_xlabel('September 12, 2026 · UTC decision clock (terminal valuation included)')
    fig.suptitle('ALL development controls: both lost money',x=.085,y=.98,ha='left',fontsize=19,weight='bold')
    fig.text(.085,.93,'Identical chart/news inputs · 24 decisions each · 5 actions differ · weights frozen during replay',color='#bac4d2')
    fig.text(.085,.025,'Partial development evidence only. Simulated trading costs included; hosting excluded.\nActions are decisions, not fills. Inventory can exceed its entry limit after prices change.',fontsize=9,color='#bac4d2')
    fig.subplots_adjust(left=.085,right=.97,top=.88,bottom=.14,hspace=.18)
    args.out.mkdir(parents=True)
    for extension in ('png','svg'):fig.savefig(args.out/('controls.'+extension),dpi=160)
    plt.close(fig)
    report={'status':'audited_control_figure','plan_sha256':failure['plan_sha256'],
        'partial_development_only':True,'data':series,'source_sha256':sha(__file__),
        'png_sha256':sha(args.out/'controls.png'),'svg_sha256':sha(args.out/'controls.svg'),
        'model_submissions':0,'new_neural_observations':0}
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
