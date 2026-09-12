"""Render the complete audited trace-reset matrix, without running a model."""
import argparse
import json
from pathlib import Path

from .core import atomic_json, digest


def figure(audit, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a=json.loads(Path(audit).read_text()); p=a['protocol']; output=Path(output)
    if (a['status']!='credit_reset_audited' or list(a['arms'])!=list(p['arms'])
            or a['verification']['observations']!=18 or a['verification']['bins']!=900
            or any(len(rows)!=3 for rows in a['arms'].values())):
        raise ValueError('A complete six-condition audit is required')
    required=('all_weights_exact','all_original_controls_reproduced','all_boundaries_verified',
              'frozen_reset_counts_unchanged','all_decoders_reconstructed')
    if any(a['verification'].get(k) is not True for k in required):raise ValueError('Missing audit checks')
    if output.with_suffix('.png').exists() or output.with_suffix('.svg').exists():
        raise ValueError('Refuse to overwrite an existing figure')
    fig=plt.figure(figsize=(12,5.4),facecolor='#101a2b');ax=fig.add_axes([0,0,1,1]);ax.set_axis_off()
    def text(x,y,s,size=11,color='#e8edf6',weight='normal'):
        ax.text(x,y,s,transform=ax.transAxes,fontsize=size,color=color,weight=weight,va='center')
    text(.03,.94,'Does clearing learning history change the extra BUY?',19,weight='bold')
    text(.03,.885,'Same trained memory and three historical images. Only KC/DAN rate traces reset between images.',10,color='#b2bfd1')
    xs=[.03,.17,.32,.54,.65,.76,.87]
    for x,label in zip(xs,['Updates','Pulses','Rate history','10:05','10:10','10:15','Final Δ L2']):
        text(x,.80,label,11,weight='bold')
    for i,(name,arm) in enumerate(p['arms'].items()):
        y=.72-i*.087; rows=a['arms'][name]
        if i==3:ax.plot([.03,.975],[y+.044,y+.044],transform=ax.transAxes,color='#62738e',lw=1)
        vals=['Online' if arm['learning'] else 'Frozen',arm['pulses'].title(),
              'Retained' if arm['boundary']=='carry' else 'Cleared at boundary',
              *[r['side'] for r in rows],f"{rows[-1]['weight_update_l2']:.4f}"]
        for x,value in zip(xs,vals):
            color={'BUY':'#67e8cf','SELL':'#ff9a9e','HOLD':'#b2bfd1'}.get(value,'#e8edf6')
            text(x,y,value,11,color=color)
    text(.03,.15,'All original controls reproduce. Frozen reset preserves all full-neuron spike-count bins.',10,color='#b2bfd1')
    text(.03,.095,'Δ L2 = weight movement during the final image, across all 7,835 plastic edges; not a loss gradient.',10,color='#b2bfd1')
    text(.03,.04,'18 observations / 900 bins audited. No account, fills, returns or profitable policy selection.',10,color='#b2bfd1')
    output.parent.mkdir(parents=True,exist_ok=True)
    for extension in ('.png','.svg'):fig.savefig(output.with_suffix(extension),dpi=160,facecolor=fig.get_facecolor())
    plt.close(fig)
    result={'audit_sha256':digest(audit),'source_sha256':digest(__file__),
            'figure_sha256':{e:digest(output.with_suffix(e)) for e in ('.png','.svg')}}
    atomic_json(output.with_suffix('.json'),result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(json.dumps(figure(a.audit,a.out),indent=2))


if __name__=='__main__':main()
