"""Locate recorded divergence after the audited rate-trace intervention.

This reads all retained neurons and plastic edges. It neither propagates a
network nor infers a causal path from the ordering of ten-millisecond samples.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .core import atomic_json, digest

FIELDS = ('u','w','weights','voltage','counts')
DECODER = ('10162','10059','10527','555871')
ROOT = Path(__file__).resolve().parents[1]
LABELS = ('Online + recorded pulses','Online + no pulses','Frozen + recorded pulses')
COLORS = ('#ffbf69','#8fbcff','#67e8cf')


def compare_arrays(carry, reset):
    for field in ('ms','neuron_ids','plastic_edges','plastic_pre','plastic_post'):
        if carry[field].dtype!=reset[field].dtype or not np.array_equal(carry[field],reset[field]):
            raise ValueError('Trace identity or time grid differs: '+field)
    times=carry['ms'];ids=carry['neuron_ids'];edges=carry['plastic_edges']
    if (times.shape!=(50,) or not np.isfinite(times).all() or np.any(np.diff(times)!=10)
            or len(set(ids.tolist()))!=len(ids) or len(set(edges.tolist()))!=len(edges)):
        raise ValueError('Expected fifty ordered native bins and unique identities')
    fields={}
    for field in FIELDS:
        left,right=carry[field],reset[field];memory=field in ('u','w','weights')
        if (left.dtype!=right.dtype or left.shape!=right.shape or left.shape!=(50,len(edges) if memory else len(ids))
                or not np.isfinite(left).all() or not np.isfinite(right).all()
                or (field=='counts' and (left.dtype.kind not in 'iu' or np.any(left<0) or np.any(right<0)))):
            raise ValueError('Invalid comparison array: '+field)
        different=left!=right;n=different.sum(axis=1);positions=np.flatnonzero(n)
        first=None
        if len(positions):
            row=int(positions[0]);columns=np.flatnonzero(different[row]);identities=edges if memory else ids
            first={'bin':row,'end_ms':float(times[row]),'observation_end_ms':float(times[row]-times[0]+10),
                'different_entities':len(columns),'first_ids':identities[columns[:12]].tolist()}
        delta=right.astype(np.float64)-left
        fields[field]={'first':first,'different_entities_per_bin':n.tolist(),
            'difference_l2_per_bin':np.linalg.norm(delta,axis=1).tolist()}
    first_edges=[]
    if fields['weights']['first'] is not None:
        row=fields['weights']['first']['bin'];indices=np.flatnonzero(carry['weights'][row]!=reset['weights'][row])
        for i in indices[:12]:
            first_edges.append({'edge':int(edges[i]),'source':str(ids[carry['plastic_pre'][i]]),
                'target':str(ids[carry['plastic_post'][i]]),'carry_weight':float(carry['weights'][row,i]),
                'reset_weight':float(reset['weights'][row,i]),
                'reset_minus_carry':float(reset['weights'][row,i])-float(carry['weights'][row,i])})
    decoder={}
    for label,a in (('carry',carry),('reset',reset)):
        counts={}
        for identity in DECODER:
            pos=np.flatnonzero(ids==int(identity))
            if len(pos)!=1:raise ValueError('Missing fixed decoder neuron')
            counts[identity]=a['counts'][:,int(pos[0])].astype(np.int64)
        direction=2*int(counts['10059'].sum()-counts['10162'].sum())
        gate=counts['10527']+counts['555871'];gate_total=int(gate.sum())
        side='HOLD' if not gate_total or abs(direction)<2 else 'BUY' if direction>0 else 'SELL'
        decoder[label]={'side':side,'difference_hz':direction,'gate_spikes':gate_total,
            'gate_spikes_per_bin':gate.tolist(),'cumulative_gate_spikes':np.cumsum(gate).tolist(),
            'cell_spikes':{k:int(v.sum()) for k,v in counts.items()}}
    return {'times_ms':times.tolist(),'fields':fields,'first_weight_edges':first_edges,'decoder':decoder,
        'initial_weights_identical':np.array_equal(carry['initial_weights'],reset['initial_weights'])}


def analyze(audit_path, artifacts):
    audit_path,artifacts=Path(audit_path),Path(artifacts)
    if digest(audit_path)!=digest(ROOT/'reports/fly-credit-reset-audit-01.json'):
        raise ValueError('Use the published complete trace-reset audit')
    a=json.loads(audit_path.read_text());checks=a['verification']
    if (a['status']!='credit_reset_audited' or checks['observations']!=18 or checks['bins']!=900
            or any(checks.get(k) is not True for k in ('all_weights_exact','all_original_controls_reproduced',
                'all_boundaries_verified','frozen_reset_counts_unchanged','all_decoders_reconstructed'))):
        raise ValueError('Incomplete reference audit')
    expected=[(name,name.removesuffix('_carry')+'_reset_rates') for name,arm in a['protocol']['arms'].items() if arm['boundary']=='carry']
    if [(p['control'],p['reset']) for p in a['comparisons']]!=expected:raise ValueError('Missing control pairs')
    used={};pairs=[]
    def verified(relative):
        path=artifacts/relative;sha=digest(path)
        if sha!=a['artifact_sha256'][relative]:raise ValueError('Audited artifact changed: '+relative)
        used[relative]=sha;return path
    for pair in a['comparisons']:
        control,reset=pair['control'],pair['reset'];observations=[]
        views={name:json.loads(verified(name+'/view.json').read_text()) for name in (control,reset)}
        types={node['id']:node['type'] for v in views.values() for node in v['nodes']}
        for i in range(1,4):
            paths=[verified(name+f'/step-{i:02}.npz') for name in (control,reset)]
            with np.load(paths[0],allow_pickle=False) as left,np.load(paths[1],allow_pickle=False) as right:
                result=compare_arrays(left,right)
            for label,name in (('carry',control),('reset',reset)):
                event=views[name]['report']['events'][i-1];audited=a['arms'][name][i-1];decoded=result['decoder'][label]
                if any(event[k]!=decoded[k] or audited[k]!=decoded[k] for k in ('side','difference_hz','gate_spikes')):
                    raise ValueError('Decoder differs from audited event')
                result[label+'_event']={k:event[k] for k in ('input_sha256','market_decision_ts','stimulus')}
            if result['carry_event']!=result['reset_event']:raise ValueError('Paired image, time or pulse differs')
            original=pair['observations'][i-1];count=result['fields']['counts'];weight=result['fields']['weights']
            if (original['first_different_bin']!=(count['first']['bin'] if count['first'] else None)
                    or original['different_count_bins']!=sum(n>0 for n in count['different_entities_per_bin'])
                    or original['different_weight_edges']!=weight['different_entities_per_bin'][-1]
                    or abs(original['end_weight_difference_l2']-weight['difference_l2_per_bin'][-1])>1e-10):
                raise ValueError('Detailed divergence does not reconcile to the published comparison')
            for edge in result['first_weight_edges']:
                edge['source_type']=types.get(edge['source']);edge['target_type']=types.get(edge['target'])
            result['observation']=i
            prefix='creditdivergence01-' if control=='trained_online_recorded_carry' else 'creditreset01-'
            connection='&edge=4110156' if prefix=='creditdivergence01-' else ''
            result['viewer_url']=f'http://127.0.0.1:8766/?run={prefix}{reset}&step={i-1}&bin=0&neuron=11402{connection}&compare={prefix}{control}'
            observations.append(result)
        pairs.append({'control':control,'reset':reset,'observations':observations})
    if any(any(o['fields'][f]['first'] is not None for f in FIELDS) for o in pairs[-1]['observations']):
        raise ValueError('Frozen control unexpectedly differs in memory or recorded activity')
    return {'status':'credit_reset_divergence_analyzed','audit_sha256':digest(audit_path),
        'protocol_sha256':digest(ROOT/'reports/fly-credit-reset-protocol-01.json'),'source_sha256':digest(__file__),
        'pairs':pairs,'artifact_sha256':used,'model_submissions':0,
        'verification':{'all_18_trace_hashes_match':True,'all_six_view_hashes_match':True,
            'all_fixed_decoders_reconstructed':True,'published_comparisons_reproduced':True,
            'frozen_memory_counts_and_sampled_voltages_identical':True},
        'interpretation':'First unequal recorded 10 ms bins, not exact event times or proven causal paths. The second observation begins at 500 ms of native time. This is the completed historical mechanism replay, with no new training, accounts or return claim.'}


def figure(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    background,foreground,muted='#0e1728','#e8edf6','#b9c6da'
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':10,'text.color':foreground,
            'axes.labelcolor':foreground,'axes.titlecolor':foreground,'xtick.color':muted,
            'ytick.color':muted,'axes.edgecolor':'#52617a','svg.fonttype':'none'}):
        fig=plt.figure(figsize=(13,12),facecolor=background)
        grid=fig.add_gridspec(4,1,left=.115,right=.97,top=.87,bottom=.15,height_ratios=[.9,1.2,1.2,1.1],hspace=.5)
        fig.text(.035,.959,'Where do carry and trace reset first diverge?',fontsize=21,weight='bold')
        fig.text(.035,.925,'Same historical images and training memory; clear only KC/DAN rate traces between observations.',fontsize=11,color=muted)
        table_ax=fig.add_subplot(grid[0]);table_ax.axis('off')
        table=[]
        for label,pair in zip(LABELS,report['pairs']):
            fields=pair['observations'][1]['fields']
            values=[str(int(fields[k]['first']['observation_end_ms']))+' ms' if fields[k]['first'] else 'Identical' for k in ('weights','voltage','counts')]
            table.append([label,*values])
        tab=table_ax.table(cellText=table,colLabels=['Condition pair','First weight difference','First sampled voltage difference','First spike-count difference'],
            colWidths=[.27,.22,.28,.23],bbox=[0,0,1,1],cellLoc='left')
        tab.auto_set_font_size(False);tab.set_fontsize(9)
        for (r,c),cell in tab.get_celld().items():
            cell.set_facecolor('#172237' if r else '#25334b');cell.set_edgecolor('#35445c');cell.set_text_props(color=foreground)
        table_ax.set_title('Second observation: elapsed time to the end of the first differing 10 ms bin',loc='left',fontsize=11,pad=10)
        axes=[fig.add_subplot(grid[i],facecolor='#172237') for i in (1,2,3)]
        for label,color,pair in zip(LABELS,COLORS,report['pairs']):
            xs=[t for obs in pair['observations'] for t in obs['times_ms']]
            axes[0].plot(xs,[n for obs in pair['observations'] for n in obs['fields']['weights']['difference_l2_per_bin']],color=color,lw=1.8,label=label)
            axes[1].plot(xs,[n for obs in pair['observations'] for n in obs['fields']['counts']['different_entities_per_bin']],color=color,lw=1.6)
        for ax in axes[:2]:
            for x in (500,1000):ax.axvline(x,color=muted,lw=1,ls=':')
            ax.set_xlim(0,1500);ax.set_xticks([0,250,500,750,1000,1250,1500]);ax.grid(alpha=.12)
        axes[0].set_title('Weight divergence spreads across the three observations',loc='left',fontsize=12)
        axes[0].set_ylabel('L2 of reset − carry\nacross 7,835 weights')
        axes[0].legend(loc='upper left',fontsize=9,frameon=False,labelcolor=foreground)
        axes[1].set_title('Full-neuron spike-count differences, including identical first observations',loc='left',fontsize=12)
        axes[1].set_yscale('symlog',linthresh=1);axes[1].set_ylabel('Differing neurons\nper 10 ms bin');axes[1].set_xlabel('Native elapsed time (ms); boundaries at 500 and 1,000 ms')
        maximum=max(max(o['fields']['counts']['different_entities_per_bin']) for p in report['pairs'] for o in p['observations'])
        axes[1].set_ylim(-.05,max(1,maximum)*1.3)
        last=report['pairs'][0]['observations'][2]
        for key,color in (('carry','#ffbf69'),('reset','#e6a8e8')):
            d=last['decoder'][key]
            axes[2].step(last['times_ms'],d['cumulative_gate_spikes'],where='post',color=color,lw=2,
                label=f"{key.title()}: {d['side']}; final direction {d['difference_hz']:+} Hz")
        axes[2].set_xlim(1000,1500);axes[2].set_ylim(-.07,1.35);axes[2].yaxis.set_major_locator(MaxNLocator(integer=True))
        axes[2].set_ylabel('Accumulated gate spikes');axes[2].set_xlabel('Native elapsed time during observation 3 (ms)')
        axes[2].set_title('Recorded-pulse pair: positive direction in both, but the reset loses the required gate spike',loc='left',fontsize=11)
        axes[2].legend(loc='upper left',frameon=False,labelcolor=foreground);axes[2].grid(alpha=.12)
        fig.text(.035,.101,'First five changed connections enter cells 10704 / 11402 (MBON11). Later effects spread through the recurrent network.',fontsize=10,color=muted)
        fig.text(.035,.074,'Sampled voltage and binned counts do not resolve within-bin event order or establish which connection mediates the action change.',fontsize=10,color=muted)
        fig.text(.035,.047,'Full frozen control remains identical. This historical intervention explains a decoder change; it does not establish profitable learning.',fontsize=10,color=muted)
        fig.text(.035,.02,'Audit SHA-256: '+report['audit_sha256'][:32]+'…',fontsize=9,color=muted)
        for ext in ('.png','.svg'):fig.savefig(Path(output).with_suffix(ext),dpi=140,facecolor=background)
        plt.close(fig)
        # Matplotlib emits trailing spaces inside multiline SVG path data.
        svg=Path(output).with_suffix('.svg')
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')


def run(audit, artifacts, output):
    output=Path(output)
    if output.exists():raise ValueError('Preserve the earlier diagnostic report')
    report=analyze(audit,artifacts);output.mkdir(parents=True)
    from .fly_credit_selection import create_views
    report['derived_views']=create_views(audit,artifacts,
        ROOT/'examples/fly-debugger/market10-pool0-trained_stimulated/view.json',output/'views')
    figure(report,output/'divergence')
    report['figure_sha256']={ext:digest((output/'divergence').with_suffix(ext)) for ext in ('.png','.svg')}
    atomic_json(output/'report.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('audit','artifacts','out'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();r=run(a.audit,a.artifacts,a.out)
    print(json.dumps({k:v for k,v in r.items() if k not in ('pairs','artifact_sha256')},indent=2))


if __name__=='__main__':main()
