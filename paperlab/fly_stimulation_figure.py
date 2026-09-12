"""Plot audited recipient activity and all decoded actions, without trading returns."""
import argparse
import json
from pathlib import Path

import numpy as np

from .core import atomic_json, digest


def render(audit_file, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    a=json.loads(Path(audit_file).read_text());out=Path(output)
    if out.exists():raise ValueError('Refuse to overwrite a rendered stimulation result')
    if a.get('status')!='verified' or a.get('full_observations_verified')!=36 or a.get('native_bins_verified')!=1800:
        raise ValueError('Requires all twelve independently audited conditions')
    arms=a['protocol']['arms'];names=list(arms)
    if list(a['rows'])!=names or any(len(a['rows'][n])!=3 for n in names):raise ValueError('Incomplete result rows')
    out.mkdir(parents=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(13,10),facecolor='#f7f8fa');grid=fig.add_gridspec(2,2,height_ratios=[1,1.55],hspace=.52,wspace=.32)
    colors={'pristine':'#426cc1','trained':'#bd4a24'}
    for pool in (0,1):
        ax=fig.add_subplot(grid[0,pool]);ax.set_facecolor('#f7f8fa')
        for memory in ('pristine','trained'):
            values=[sum(sum(r['recipient_counts'].values()) for r in a['rows'][f'pool{pool}-{memory}-current{current}']) for current in (0,5,10)]
            ax.plot([0,5,10],values,'o-',color=colors[memory],label=memory.capitalize(),lw=2)
            for x,y in zip([0,5,10],values):ax.annotate(str(y),(x,y),xytext=(0,7 if memory=='pristine' else -15),textcoords='offset points',ha='center',color=colors[memory])
        ax.set_title(('ALL','baton')[pool]+' · two targeted MBON11 cells',loc='left',weight='bold')
        ax.set_xlabel('Applied current per cell (model units)');ax.set_ylabel('Recipient spikes across three observations')
        ax.set_xticks([0,5,10]);ax.margins(y=.2);ax.legend(frameon=False);ax.grid(axis='y',alpha=.18)
    ax=fig.add_subplot(grid[1,0]);ax.set_facecolor('#f7f8fa')
    for pool in (0,1):
        rows=[next(c for c in a['comparisons'] if c['pristine'].startswith(f'pool{pool}-') and c['current']==cur) for cur in (0,5,10)]
        for obs in range(3):
            ax.plot([0,5,10],[r['changed_neurons'][obs] for r in rows],marker=('o','s','^')[obs],
                    linestyle='-' if pool==0 else '--',color=('#426cc1','#bd4a24')[pool],alpha=.8,
                    label=f'{("ALL","baton")[pool]} · image {obs+1}')
    ax.set_title('Trained vs pristine: changed neuron counts',loc='left',weight='bold');ax.set_xlabel('Applied current per cell (model units)')
    ax.set_ylabel('Neurons with different total spike counts');ax.set_xticks([0,5,10]);ax.grid(axis='y',alpha=.18)
    ax.legend(frameon=False,fontsize=8,ncol=2)
    ax=fig.add_subplot(grid[1,1]);order=sorted(names,key=lambda n:(int(n[4]),arms[n]['current'],arms[n]['memory']))
    actions={'SELL':0,'HOLD':1,'BUY':2};matrix=np.array([[actions[r['side']] for r in a['rows'][n]] for n in order])
    ax.imshow(matrix,cmap=ListedColormap(['#f0c4b8','#dce1e8','#bddcce']),vmin=0,vmax=2,aspect='auto')
    for i,n in enumerate(order):
        for j,r in enumerate(a['rows'][n]):ax.text(j,i,r['side'],ha='center',va='center',fontsize=9,color='#18283a')
    ax.set_yticks(range(12),[f'{("ALL","baton")[int(n[4])]} / {arms[n]["current"]} / {arms[n]["memory"]}' for n in order],fontsize=8)
    ax.set_xticks(range(3),['Image 1','Image 2','Image 3']);ax.set_title('Fixed decoder · every condition',loc='left',weight='bold')
    fig.suptitle('Recipient stimulation with frozen paper-trained memory',x=.08,y=.98,ha='left',weight='bold',fontsize=19)
    fig.text(.08,.942,'Same six historical market images · full graph · all four unstimulated controls reproduced',fontsize=11,color='#42536a')
    fig.text(.08,.045,'Targets: 10704 and 11402. Current applied for 500 ms per image; activity carried between images.\nPost-hoc mechanism diagnostic: no trading account, returns, learning, or policy promotion.',fontsize=10,color='#42536a')
    fig.subplots_adjust(left=.08,right=.97,top=.88,bottom=.13)
    for suffix in ('png','svg'):fig.savefig(out/('activity.'+suffix),dpi=170,facecolor=fig.get_facecolor())
    plt.close(fig)
    atomic_json(out/'provenance.json',{'audit_sha256':digest(audit_file),'source_sha256':digest(__file__),
                'artifacts':{s:digest(out/('activity.'+s)) for s in ('png','svg')}})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();render(a.audit,a.out)


if __name__=='__main__':main()
