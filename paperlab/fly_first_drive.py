"""Resolve the first audited weight difference into recorded KC/DAN terms.

The substitutions below hold one bin's recorded spikes and starting u/w fixed.
They are memory-rule calculations, not alternative neural trajectories.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from .core import atomic_json, digest
from .fly_credit_audit import close, rule_module
from .fly_credit_divergence import analyze

ROOT=Path(__file__).resolve().parents[1]


def terms(k_rate,d_rates,k_trace,d_traces,gain,parameters,eta=.001):
    D,Y,G=map(lambda a:np.asarray(a,dtype=np.float64),(d_rates,d_traces,gain))
    if (D.ndim!=1 or D.shape!=Y.shape or D.shape!=G.shape
            or not all(np.isfinite(a).all() for a in (D,Y,G))
            or not all(math.isfinite(x) for x in (k_rate,k_trace,eta))):
        raise ValueError('Require finite aligned recorded rates, traces and gains')
    ak=math.exp(-.005/parameters['trace_kc_seconds']);ad=math.exp(-.005/parameters['trace_dan_seconds'])
    kmid=k_trace*ak+k_rate*(1-ak);dmid=Y*ad+D*(1-ad)
    positive=eta*k_rate*G*dmid;negative=-eta*D*G*kmid
    return {'kc_mid_trace_hz':kmid,'dan_mid_trace_hz':dmid,
            'current_kc_times_dan_trace':positive,'current_dan_times_kc_trace':negative,
            'total_by_dan':positive+negative,'total':float(np.sum(positive+negative))}


def substitutions(carry,reset,K,D,circuit,rule):
    if any(not np.array_equal(carry[k],reset[k]) for k in ('u','w','weights')):
        raise ValueError('The first update must start from identical connection memory')
    results={}
    for name,k,d in [('carry',0,0),('reset_kc_history_only',1,0),('reset_dan_history_only',0,1),('reset_both_histories',1,1)]:
        options=(carry,reset);state={key:value.copy() for key,value in carry.items()}
        state['kc']=options[k]['kc'].copy();state['dan']=options[d]['dan'].copy()
        rule.advance(state['kc'],state['dan'],state['u'],state['w'],K,D,circuit['gain'],.01,.001)
        state['weights']=(circuit['baseline']*(1+state['w'])).astype(carry['weights'].dtype)
        results[name]=state
    return results


def inspect(audit_path,artifacts):
    audit_path,artifacts=Path(audit_path),Path(artifacts)
    detailed=analyze(audit_path,artifacts);audit=json.loads(audit_path.read_text());used={}
    def verified(relative):
        path=artifacts/relative;sha=digest(path)
        if sha!=audit['artifact_sha256'][relative]:raise ValueError('Audited artifact differs: '+relative)
        used[relative]=sha;return path
    def arrays(relative):
        with np.load(verified(relative),allow_pickle=False) as a:return {k:a[k] for k in a.files}
    rule_path=ROOT/'vendor/stonkfly/stonkfly/neural/rule.py'
    if digest(rule_path)!=audit['code_sha256'][str(rule_path.relative_to(ROOT))]:raise ValueError('Recorded learning rule differs')
    rule=rule_module();c=arrays('circuit.npz');pairs=[]
    for pair in detailed['pairs'][:2]:
        names=[pair['control'],pair['reset']];first=pair['observations'][1]['fields']['weights']['first'];row=first['bin']
        first_edges=pair['observations'][1]['first_weight_edges']
        if first['different_entities']!=len(first_edges):raise ValueError('First-edge preview is incomplete')
        records=[];states=[];boundaries=[];views=[]
        for name in names:
            a=arrays(name+'/step-02.npz');boundary=arrays(name+'/boundary-02.npz')
            before=arrays(name+'/step-01.npz') if row==0 else a;previous=-1 if row==0 else row-1
            states.append({**{k:before[k][previous].copy() for k in ('u','w','weights')},
                'kc':boundary['after__rate_kc'].copy() if row==0 else a['kc'][row-1].copy(),
                'dan':boundary['after__rate_dan'].copy() if row==0 else a['dan'][row-1].copy()})
            records.append(a);boundaries.append(boundary)
            views.append(json.loads(verified(name+'/view.json').read_text()))
        if not np.array_equal(records[0]['counts'][:row+1],records[1]['counts'][:row+1]):
            raise ValueError('Do not substitute histories after recorded firing has diverged')
        K=records[0]['counts'][row,c['pre']]/.01;D=records[0]['counts'][row,c['dan']]/.01
        models=substitutions(*states,K,D,c,rule)
        errors={}
        for model,index in [('carry',0),('reset_both_histories',1)]:
            if not np.array_equal(models[model]['weights'],records[index]['weights'][row]):
                raise ValueError('One-bin weight reconstruction differs from its recorded control')
            errors[model]={k:close(models[model][k],records[index][k][row],k) for k in ('kc','dan','u','w')}
        cases=[]
        for name,state in models.items():
            cases.append({'name':name,'weights_equal_recorded_carry':np.array_equal(state['weights'],records[0]['weights'][row]),
                'weights_equal_recorded_reset':np.array_equal(state['weights'],records[1]['weights'][row]),
                'weight_count':len(c['edges']),'different_weights_from_carry':int(np.count_nonzero(state['weights']!=records[0]['weights'][row]))})
        entries=[];types={n['id']:n['type'] for v in views for n in v['nodes']}
        for edge in first_edges:
            pi=int(np.flatnonzero(c['edges']==edge['edge'])[0]);details={}
            for index,label in enumerate(('carry','reset')):
                state=states[index];value=terms(K[pi],D,state['kc'][pi],state['dan'],c['gain'][:,pi],rule.PARAMETERS)
                # Independent per-driver sum must agree with the matrix form used by advance.
                ak=math.exp(-.005/rule.PARAMETERS['trace_kc_seconds']);ad=math.exp(-.005/rule.PARAMETERS['trace_dan_seconds'])
                kmid=state['kc']*ak+K*(1-ak)
                dmid=state['dan']*ad+D*(1-ad)
                matrix=.001*(K*(c['gain'].T@dmid)-(c['gain'].T@D)*kmid)
                if not np.isclose(value['total'],matrix[pi],rtol=1e-12,atol=1e-14):raise ValueError('Per-driver terms do not sum to recorded rule drive')
                drivers=[]
                for j in np.flatnonzero(c['gain'][:,pi]):
                    identity=str(records[index]['neuron_ids'][c['dan'][j]])
                    drivers.append({'id':identity,'type':types.get(identity),'gain':float(c['gain'][j,pi]),
                        'spikes':int(records[index]['counts'][row,c['dan'][j]]),
                        **{k:float(value[k][j]) for k in ('current_kc_times_dan_trace','current_dan_times_kc_trace','total_by_dan')}})
                details[label]={'source_spikes':int(records[index]['counts'][row,c['pre'][pi]]),
                    'kc_mid_trace_hz':float(value['kc_mid_trace_hz']),
                    'earlier_image_kc_mid_trace_hz':float(boundaries[index]['after__rate_kc'][pi]*math.exp(-(.01*row+.005)/rule.PARAMETERS['trace_kc_seconds'])),
                    'drive_u_per_second':value['total'],'drivers':drivers,
                    'weight_before':float(state['weights'][pi]),'weight_after':float(records[index]['weights'][row,pi])}
            entries.append({'edge':edge['edge'],'source':edge['source'],'target':edge['target'],**details})
        pairs.append({'control':names[0],'reset':names[1],'observation':2,'bin':row,
            'end_ms_in_observation':first['observation_end_ms'],'current_counts_identical_through_this_bin':True,
            'starting_connection_memory_identical':True,'recorded_control_errors':errors,'entries':entries,'substitutions':cases})
    return {'status':'first_learning_drive_reconstructed','audit_sha256':digest(audit_path),'source_sha256':digest(__file__),
        'rule_sha256':digest(rule_path),'rule_parameters':rule.PARAMETERS,'pairs':pairs,'artifact_sha256':used,
        'full_divergence_verification':detailed['verification'],'neural_observations':0,
        'interpretation':'Algebraic substitutions of recorded rate histories at the first differing memory-update bin. All recorded spikes and the common starting connection memory are held fixed. These are not KC-only or DAN-only full-network replays and make no claim about their later spikes, actions, or returns. Prior activity traces may be part of delayed reinforcement; removing a drive is not proof of correcting an error.'}


def figure(report,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for pair in report['pairs']:
        cases={c['name']:c for c in pair['substitutions']}
        if not (cases['reset_kc_history_only']['weights_equal_recorded_reset']
                and cases['reset_dan_history_only']['weights_equal_recorded_carry']
                and all(e['carry']['source_spikes']==0 and e['reset']['drive_u_per_second']==0 for e in pair['entries'])):
            raise ValueError('Figure interpretation does not match reconstructed findings')
    bg,fg,muted='#0e1728','#e8edf6','#b9c6da'
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':10,'text.color':fg,'axes.labelcolor':fg,
        'xtick.color':muted,'ytick.color':muted,'axes.edgecolor':'#52617a','svg.fonttype':'none'}):
        fig,axes=plt.subplots(1,2,figsize=(13,8),facecolor=bg)
        fig.subplots_adjust(left=.16,right=.965,bottom=.33,top=.76,wspace=.43)
        for ax,pair,title in zip(axes,report['pairs'],('Recorded aversive pulse','No injected pulse')):
            ax.set_facecolor('#172237');entries=pair['entries'];ys=np.arange(len(entries));left=np.zeros(len(entries))
            for driver,color in [('11327','#8fbcff'),('11900','#e6a8e8')]:
                values=np.array([next(d['total_by_dan'] for d in e['carry']['drivers'] if d['id']==driver) for e in entries])
                ax.barh(ys,values,left=left,color=color,label='DAN '+driver);left+=values
            ax.scatter(np.zeros(len(entries)),ys,color='#67e8cf',marker='|',s=140,label='Trace reset: zero drive',zorder=3)
            ax.set_yticks(ys,[f"{e['edge']}\n{e['source']} → {e['target']}" for e in entries],fontsize=9)
            ax.invert_yaxis();ax.axvline(0,color=muted,lw=.7);ax.set_xlim(-.60,.035)
            ax.set_xlabel('Learning drive (u / second)');ax.grid(axis='x',alpha=.12)
            ax.set_title(title+'\nfirst weight difference at '+str(int(pair['end_ms_in_observation']))+' ms',color=fg,fontsize=12)
        fig.suptitle('Old KC activity combines with current DAN spikes',x=.035,y=.97,ha='left',fontsize=20,weight='bold',color=fg)
        fig.text(.035,.91,'The source neurons of these five connections have zero spikes in this bin. Their carried traces still drive updates.',color=muted)
        handles,labels=axes[1].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper left',bbox_to_anchor=(.16,.87),ncol=3,frameon=False,labelcolor=fg,fontsize=9)
        fig.text(.035,.225,'One-bin substitutions with identical recorded firing and starting u/w:',fontsize=12,weight='bold',color=fg)
        fig.text(.035,.175,'Use reset KC history only → all 7,835 resulting weights match the recorded both-traces reset.',color=muted)
        fig.text(.035,.13,'Use reset DAN history only → all 7,835 resulting weights match the recorded carry control.',color=muted)
        fig.text(.035,.075,'This locates the immediate rule term. It does not simulate later dynamics or establish better decisions or returns.',color=muted)
        fig.text(.035,.035,'Recorded second image, two audited conditions. The cloud paper comparison is unchanged.',color=muted)
        for ext in ('.png','.svg'):fig.savefig(Path(output).with_suffix(ext),dpi=140,facecolor=bg)
        plt.close(fig)
        svg=Path(output).with_suffix('.svg');svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')


def run(audit,artifacts,output):
    output=Path(output)
    if output.exists():raise ValueError('Preserve earlier diagnostic outputs')
    report=inspect(audit,artifacts);output.mkdir(parents=True);figure(report,output/'first-drive')
    report['figure_sha256']={ext:digest((output/'first-drive').with_suffix(ext)) for ext in ('.png','.svg')}
    atomic_json(output/'report.json',report);return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('audit','artifacts','out'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();r=run(a.audit,a.artifacts,a.out)
    print(json.dumps({'status':r['status'],'pairs':[{k:p[k] for k in ('control','bin','substitutions')} for p in r['pairs']]},indent=2))


if __name__=='__main__':main()
