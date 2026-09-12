"""Map audited full-neuron counts onto retained plastic edges without inference."""
import argparse
import json
from pathlib import Path

import numpy as np

from .core import atomic_json, digest
from .fly_market_study import memory_signature
from .fly_paper_figure import PHASES, evidence
from .fly_paper_memory import array_hash
from .fly_paper_study import trace_name


def activation(counts, pre, post, changed):
    counts, pre, post, changed = map(np.asarray, (counts, pre, post, changed))
    if (counts.ndim != 1 or counts.dtype.kind not in 'iu' or np.any(counts < 0)
            or pre.ndim != 1 or not len(pre) or pre.shape != post.shape or pre.shape != changed.shape
            or pre.dtype.kind not in 'iu' or post.dtype.kind not in 'iu' or changed.dtype != bool
            or np.any(pre < 0) or np.any(post < 0) or np.any(pre >= len(counts)) or np.any(post >= len(counts))):
        raise ValueError('Invalid full counts or plastic edge mapping')
    active = counts[pre] > 0
    return {'source_spikes': int(counts[np.unique(pre)].sum()),
            'active_sources': int(np.count_nonzero(counts[np.unique(pre)])),
            'active_edges': int(active.sum()), 'active_changed_edges': int((active & changed).sum()),
            'recipient_spikes': int(counts[np.unique(post)].sum()),
            'active_recipients': int(np.count_nonzero(counts[np.unique(post)]))}


def analyze(study, mapping):
    study, mapping = Path(study), Path(mapping)
    report = json.loads((study/'report.json').read_text());evidence(report)
    r = report['registration'];a = study/'artifacts'
    m = json.loads((mapping/'report.json').read_text())
    if (m['registration_sha256'] != digest(study/'preregistration.json')
            or m['training_audit_sha256'] != r['training_audit_sha256']
            or m['plastic_map_sha256'] != digest(mapping/'plastic-map.npz')
            or any(m['verification'].get(k) is not True for k in ('imported_memories_match_registration','recipients_reconcile_to_graph','all_weights_and_dynamics_unchanged'))
            or digest(a/'neuron-ids.npz') != report['neuron_ids_sha256']):
        raise ValueError('Native mapping provenance differs from the audited study')
    with np.load(mapping/'plastic-map.npz', allow_pickle=False) as f:
        preids, postids, edges, baseline = (f[k] for k in ('pre_ids','post_ids','edge_ids','baseline'))
    with np.load(a/'neuron-ids.npz', allow_pickle=False) as f:ids = f['neuron_ids']
    index = {str(n): i for i,n in enumerate(ids)}
    if len(index) != len(ids) or len(np.unique(edges)) != len(edges):raise ValueError('Duplicate graph identity')
    pre = np.array([index[str(n)] for n in preids]);post = np.array([index[str(n)] for n in postids])
    rows = []; traces = [];active_union = set();view_hashes = {}
    for i,key in enumerate(r['cohort']):
        path = a/f'imported/pool{i}-memory.npz'
        if digest(path) != r['source_memories'][key]['memory_file_sha256']:raise ValueError('Imported memory file differs')
        with np.load(path, allow_pickle=False) as f:memory = {k: f[k] for k in ('weights','u','w')}
        if memory_signature(memory) != r['source_memories'][key]['memory_sha256']:raise ValueError('Imported memory arrays differ')
        changed = memory['weights'] != baseline
        for phase in PHASES:
            for arm in r['arms']:
                events = [d['neural'] for d in report['phase_diagnostics'][key][arm][phase]['decisions'] if d['neural']]
                folder = a/'boundaries'/f'pool{i}-{arm}-{phase}'
                view = json.loads((study/trace_name(i,arm,phase)/'view.json').read_text()) if events else None
                if view:view_hashes[trace_name(i,arm,phase)] = digest(study/trace_name(i,arm,phase)/'view.json')
                if view and view['report']['events'] != events:raise ValueError('View differs from audited neural events')
                node = {n['id']: j for j,n in enumerate(view['nodes'])} if view else {}
                for j,event in enumerate(events):
                    path = folder/(f'boundary-{j+2:02}.npz' if j+1 < len(events) else 'final-counts.npz')
                    with np.load(path, allow_pickle=False) as f:counts = f['before__counts' if j+1 < len(events) else 'counts']
                    if counts.shape != ids.shape or array_hash(counts) != event['spike_sha256']:raise ValueError('Full neuron counts differ')
                    row = {'pool_index':i,'phase':phase,'arm':arm,'observation':j,
                           'decision_ts':event['market_decision_ts'], 'KC_spikes':event['KC_spikes'],
                           **activation(counts,pre,post,changed),
                           'recipient_counts':{str(ids[n]):int(counts[n]) for n in np.unique(post)}}
                    rows.append(row)
                    if phase == 'test':active_union.update(int(n) for n in np.flatnonzero(counts[pre] > 0))
                    f = view['frames'][j]; n = node['11402']; bins = np.asarray(f['counts'])
                    if int(bins[:,n].sum()) != counts[index['11402']]:raise ValueError('Displayed recipient counts differ')
                    if i == 0 and phase == 'test' and arm in ('pristine_frozen','trained_frozen'):
                        traces.append({'arm':arm,'observation':j,'times_ms':f['times_ms'],
                                       'voltage':[v[n] for v in f['voltage']], 'spikes':int(counts[index['11402']])})
    matched = all(
        [d['neural']['spike_sha256'] if d['neural'] else None for d in report['phase_diagnostics'][key][left]['test']['decisions']]
        == [d['neural']['spike_sha256'] if d['neural'] else None for d in report['phase_diagnostics'][key][right]['test']['decisions']]
        for key in r['cohort'] for left,right in (('pristine_frozen','trained_frozen'),('pristine_input_reset','trained_input_reset')))
    return {'kind':'audited_plastic_path_activity','plan_sha256':report['plan_sha256'],
            'report_sha256':digest(study/'report.json'),'mapping_sha256':digest(mapping/'plastic-map.npz'),
            'source_sha256':digest(__file__),'view_sha256':view_hashes,'plastic_edges':len(edges),'rows':rows,'sampled_voltage':traces,
            'matched_test_full_counts_identical':matched,
            'test_active_edge_union':[{'edge_id':int(edges[n]),'pre_id':str(preids[n]),'post_id':str(postids[n])} for n in sorted(active_union)],
            'verification':{'full_counts_match_audited_events':True,'mapping_matches_audited_memory':True,'neural_observations':0},
            'interpretation':'Active edge means its source spiked within that observation, not measured transmission. Source and recipient totals count each neuron once. Changed edges are relative to the pinned trained memory, including pristine controls. Voltages are recorded at 10 ms boundaries and do not measure within-bin peaks. This describes pathway engagement, not profitability or a validated intervention.'}


def render(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import ScalarFormatter
    test_rows = [r for r in report['rows'] if r['phase']=='test']
    if (len(test_rows)!=24 or any(r['recipient_spikes'] for r in test_rows)
            or not report['matched_test_full_counts_identical']
            or any(r['active_edges']!=r['active_changed_edges'] for r in test_rows)
            or any(r['spikes'] for r in report['sampled_voltage'])):
        raise ValueError('This figure requires complete test coverage, silent recipients and matched full counts')
    arms = ('pristine_frozen','trained_frozen','pristine_input_reset','trained_input_reset')
    labels = ('Pristine / carry','Trained / carry','Pristine / reset','Trained / reset')
    bg,fg,muted = '#0e1728','#e8edf6','#b9c6da'
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':10,'text.color':fg,'axes.labelcolor':fg,'xtick.color':muted,'ytick.color':fg,'svg.fonttype':'none'}):
        fig=plt.figure(figsize=(14,10.5),facecolor=bg)
        grid=fig.add_gridspec(2,2,left=.20,right=.93,top=.82,bottom=.22,height_ratios=[2,1],hspace=.5,wspace=.12)
        fig.text(.04,.95,'Learned weights can differ while recipient neurons stay silent',fontsize=20,weight='bold')
        fig.text(.04,.914,'Study 09 · full-count reconstruction · no new neural simulation',fontsize=12,color=muted)
        fig.text(.04,.877,'Cell labels: active plastic edges / total recipient spikes. Color: active edges, logarithmic scale.',fontsize=11)
        for k,phase in enumerate(PHASES):
            values=np.full((8,3),np.nan);counts=np.zeros((8,3),dtype=int)
            for row in report['rows']:
                if row['phase']==phase:
                    y=row['pool_index']*4+arms.index(row['arm']);x=row['observation']
                    values[y,x]=row['active_edges'];counts[y,x]=row['recipient_spikes']
            ax=fig.add_subplot(grid[0,k],facecolor='#172237')
            im=ax.imshow(values,norm=LogNorm(vmin=1,vmax=report['plastic_edges']),cmap='viridis',aspect='auto')
            for y in range(8):
                for x in range(3):
                    if np.isfinite(values[y,x]):ax.text(x,y,f'{int(values[y,x]):,} / {counts[y,x]:,}',ha='center',va='center',color=bg if values[y,x]>1000 else 'white',fontsize=10)
            ax.set_xticks(range(3),['1','2','3']);ax.set_xlabel('Observed input in phase')
            ax.set_yticks(range(8),[f'Pool {i} · {label}' for i in range(2) for label in labels] if k==0 else ['']*8)
            ax.axhline(3.5,color=fg,lw=1);ax.set_title(phase.title(),loc='left',color=fg,fontsize=15)
        cax=fig.add_axes([.945,.44,.012,.38]);bar=fig.colorbar(im,cax=cax);bar.ax.tick_params(colors=muted);bar.set_ticks([1,10,100,1000,7835]);bar.ax.yaxis.set_major_formatter(ScalarFormatter())
        ax=fig.add_subplot(grid[1,:],facecolor='#172237')
        for arm,color,label in zip(arms[:2],('#8fbcff','#67e8cf'),labels[:2]):
            ts=[];vs=[]
            for row in report['sampled_voltage']:
                if row['arm']==arm:ts+=row['times_ms'];vs+=row['voltage']
            ax.plot(ts,vs,color=color,lw=1.4,label=label)
        ax.axhline(-45,color='#ff8e8e',ls='--',lw=1,label='Native spike threshold (−45 mV)')
        ax.set_title('Pool 0 · test · MBON11 neuron 11402 · zero spikes in both conditions',loc='left',color=fg,fontsize=12)
        ax.set_xlabel('Recorded neural time (ms)');ax.set_ylabel('Sampled voltage (mV)');ax.grid(alpha=.12)
        ax.legend(loc='lower left',ncols=3,frameon=False,labelcolor=fg,fontsize=9)
        fig.text(.04,.145,'All six plastic recipients stayed silent throughout test; trained/pristine full-neuron spike counts matched.',fontsize=11)
        fig.text(.04,.111,'The few active test edges still had changed weights and altered sampled voltages. Memory was not absent.',fontsize=11)
        fig.text(.04,.077,'Source spikes do not establish delivered transmission. Sampling at 10 ms does not recover within-bin voltage peaks.',color=muted,fontsize=10)
        fig.text(.04,.043,'This identifies a quiet memory pathway on these inputs; it does not prove that increasing its activity improves trading.',color=muted,fontsize=10)
        try:fig.savefig(output,dpi=140,facecolor=bg)
        finally:plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('study','mapping','out'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--figures',action='store_true');args=parser.parse_args()
    if args.out.exists():raise ValueError('Refuse to overwrite activity evidence')
    report=analyze(args.study,args.mapping);args.out.mkdir(parents=True)
    atomic_json(args.out/'report.json',report)
    if args.figures:
        for ext in ('png','svg'):render(report,args.out/('activity.'+ext))
    print(args.out/'report.json')


if __name__=='__main__':main()
