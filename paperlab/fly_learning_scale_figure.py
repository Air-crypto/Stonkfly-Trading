"""Render the complete audited learning-rate comparison without model compute."""
import argparse
import json
from pathlib import Path

from .core import atomic_json, digest


def render(audit, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a=json.loads(Path(audit).read_text());p=a['protocol'];v=a['verification']
    required=('all_weights_exact','all_three_original_controls_reproduced','all_boundaries_verified',
              'all_rate_histories_carried','all_decoders_reconstructed','all_displayed_topology_and_series_verified')
    if (a['status']!='learning_scale_audited' or list(a['arms'])!=p['execution_order']
        or len(a['arms'])!=5 or any(len(x)!=3 for x in a['arms'].values())
        or v['observations']!=15 or v['bins']!=750 or any(v.get(k) is not True for k in required)):
        raise ValueError('The complete five-condition recording audit is required')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    bg,fg,muted='#0e1728','#e8edf6','#b9c6da'
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':10,'text.color':fg,
        'axes.labelcolor':fg,'axes.titlecolor':fg,'xtick.color':muted,'ytick.color':muted,
        'axes.edgecolor':'#52617a','svg.fonttype':'none','text.parse_math':False}):
        fig=plt.figure(figsize=(13,8.5),facecolor=bg)
        fig.text(.035,.95,'Lower learning rate: recorded decisions and weight updates',fontsize=21,weight='bold')
        fig.text(.035,.90,'Both histories retained. Same trained starting memory, three historical inputs, and full native graph.',color=muted)
        ax=fig.add_axes([.035,.56,.93,.27]);ax.axis('off');rows=[]
        for name,arm in p['arms'].items():
            events=a['arms'][name]
            rows.append(['Online' if arm['learning'] else 'Frozen',arm['pulses'].title(),
                         f"{arm['eta']:.4f}" if arm['learning'] else 'Inactive',
                         *[e['side'] for e in events],'/'.join(str(e['gate_spikes']) for e in events),
                         f"{events[-1]['weight_update_l2']:.4f}"])
        tab=ax.table(cellText=rows,colLabels=['Learning','Pulses','Eta','Image 1','Image 2','Image 3','Gate spikes','Image 3 Δ L2'],
                     colWidths=[.11,.12,.11,.12,.12,.12,.15,.15],bbox=[0,0,1,1],cellLoc='left')
        tab.auto_set_font_size(False);tab.set_fontsize(10)
        for (r,c),cell in tab.get_celld().items():
            cell.set_facecolor('#25334b' if not r else '#172237');cell.set_edgecolor('#35445c')
            cell.set_text_props(color={'BUY':'#67e8cf','SELL':'#ff9a9e','HOLD':muted}.get(cell.get_text().get_text(),fg))
        axes=[fig.add_axes([.08,.245,.39,.235],facecolor='#172237'),fig.add_axes([.57,.245,.39,.235],facecolor='#172237')]
        for ax,pulses in zip(axes,('recorded','none')):
            for name,color,label in [('trained_online_'+pulses+'_carry','#ffbf69','Eta 0.001'),
                                      ('trained_low_eta_'+pulses+'_carry','#8fbcff','Eta 0.0001')]:
                ax.plot([1,2,3],[e['weight_update_l2'] for e in a['arms'][name]],marker='o',color=color,lw=2,label=label)
            ax.axhline(0,color='#67e8cf',lw=.8,ls=':');ax.set_xticks([1,2,3]);ax.set_xlim(.9,3.1)
            ax.set_ylim(bottom=-.05);ax.grid(alpha=.12);ax.set_xlabel('Historical image');ax.set_ylabel('Within-image weight-change L2')
            ax.set_title('Recorded pulses' if pulses=='recorded' else 'No injected pulses',loc='left',fontsize=12)
            ax.legend(frameon=False,labelcolor=fg,fontsize=9)
        fig.text(.035,.155,'All original controls reproduce. Each lower-rate trajectory is propagated independently; its first image can already differ.',color=muted)
        fig.text(.035,.11,'Weight movement covers all 7,835 plastic edges. It is not a loss gradient or a measure of trading quality.',color=muted)
        fig.text(.035,.065,'15 observations / 750 bins audited. No accounts, new feedback, trading returns, fresh holdout or selected policy.',color=muted)
        fig.text(.035,.025,'Audit SHA-256: '+digest(audit)[:32]+'…',color=muted,fontsize=9)
        hashes={}
        for e in ('.png','.svg'):
            path=output/('comparison'+e);fig.savefig(path,dpi=150,facecolor=bg)
            if e=='.svg':path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
            hashes[e]=digest(path)
        plt.close(fig)
    record={'status':'audited_learning_scale_figure_rendered','audit_sha256':digest(audit),
            'source_sha256':digest(__file__),'figure_sha256':hashes,'model_submissions':0}
    atomic_json(output/'figure.json',record);return record


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(json.dumps(render(a.audit,a.out),indent=2))


if __name__=='__main__':main()
