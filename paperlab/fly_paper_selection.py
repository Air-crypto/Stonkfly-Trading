"""Extend an audited frozen paper view with explicitly selected recorded edges."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

from .core import atomic_json, digest
from .fly_market_study import memory_signature
from .fly_paper_figure import evidence, PHASES
from .fly_paper_study import trace_name
from .fly_trace_memory import enrich_frame


def expand(view, reference, recordings, edge_ids, baseline, memory, boundaries=None):
    """Copy this run's native arrays; the reference supplies endpoint labels only."""
    edge_ids=list(map(str,edge_ids))
    if (not edge_ids or len(edge_ids)>32 or len(set(edge_ids))!=len(edge_ids)
            or any(not str(e).isdecimal() for e in edge_ids)):
        raise ValueError('Select 1–32 unique plastic edge IDs')
    for field in ('graph','native_build','upstream_commit','plan_sha256'):
        if not view['report'].get(field) or view['report'][field]!=reference['report'].get(field):
            raise ValueError('Reference graph, build or study differs')
    if (not view['frames'] or len(view['frames'])!=len(view['report']['events'])
            or any(f['event']!=e for f,e in zip(view['frames'],view['report']['events']))):
        raise ValueError('Original frame events differ')
    result=copy.deepcopy(view);nodes=result['nodes'];edges=result['edges']
    original_columns=np.array([n['index'] for n in nodes],dtype=np.int64)
    node_index={n['id']:i for i,n in enumerate(nodes)}
    edge_index={str(e['id']):e for e in edges}
    candidates={str(e['id']):e for e in reference['edges']}
    if (len(node_index)!=len(nodes) or len(set(original_columns))!=len(nodes)
            or len(edge_index)!=len(edges) or len(candidates)!=len(reference['edges'])):
        raise ValueError('Ambiguous displayed identities')
    added_nodes=[];added_edges=[];selected=list(result['plastic_selection'])
    for identity in map(str,edge_ids):
        if identity not in candidates:raise ValueError('Requested edge missing from reference')
        source=candidates[identity];pi=source['plastic_index']
        if type(pi) is not int or not 0<=pi<len(baseline) or source['weight']!=float(baseline[pi]):
            raise ValueError('Reference plastic edge or pristine weight differs')
        endpoints=[]
        for field in ('source','target'):
            index=source[field]
            if type(index) is not int or not 0<=index<len(reference['nodes']):raise ValueError('Invalid reference endpoint')
            n=reference['nodes'][index]
            if n['id'] in node_index:
                if nodes[node_index[n['id']]]['index']!=n['index']:raise ValueError('Reference neuron index differs')
            else:
                if n['index'] in [x['index'] for x in nodes]:raise ValueError('Reference reuses a neuron index')
                node_index[n['id']]=len(nodes);nodes.append(copy.deepcopy(n));added_nodes.append(n['id'])
            endpoints.append(node_index[n['id']])
        desired={**source,'source':endpoints[0],'target':endpoints[1]}
        if identity in edge_index:
            if edge_index[identity]!=desired:raise ValueError('Existing edge differs from reference')
        else:
            edges.append(desired);edge_index[identity]=desired;added_edges.append(identity)
        if pi not in selected:selected.append(pi)
    result['plastic_selection']=selected
    columns=np.array([n['index'] for n in nodes],dtype=np.int64)
    totals=np.zeros(len(nodes),dtype=np.int64);hashes={};boundary_hashes={}
    for i,(old,f) in enumerate(zip(view['frames'],result['frames']),1):
        path=Path(recordings)/f'step-{i:02}.npz'
        original_memory=enrich_frame(view,old,path)
        for k,v in original_memory.items():
            if k in old and old[k]!=v:raise ValueError('Displayed connection memory differs')
        with np.load(path,allow_pickle=False) as a:
            ids=a['neuron_ids'];native_edges=a['plastic_edges']
            if (np.any(columns<0) or np.any(columns>=len(ids))
                    or any(str(ids[k])!=n['id'] for k,n in zip(columns,nodes))):
                raise ValueError('Extended neuron identities differ from native trace')
            voltage=a['voltage']
            if (voltage.shape!=(50,len(ids)) or not np.isfinite(voltage).all()
                    or not np.array_equal(voltage[:,original_columns].round(3),old['voltage'])):
                raise ValueError('Recorded voltages differ')
            for k in ('weights','u','w'):
                if not np.array_equal(a[k],np.broadcast_to(memory[k],a[k].shape)):
                    raise ValueError('Trace differs from registered frozen memory')
            if not np.array_equal(a['initial_weights'],memory['weights']):raise ValueError('Initial memory differs')
            for e in edges:
                pi=e['plastic_index']
                if pi is None:continue
                if (str(native_edges[pi])!=str(e['id']) or e['weight']!=float(baseline[pi])
                        or int(a['plastic_pre'][pi])!=nodes[e['source']]['index']
                        or int(a['plastic_post'][pi])!=nodes[e['target']]['index']):
                    raise ValueError('Extended edge differs from native endpoint map')
            f['counts']=a['counts'][:,columns].tolist();totals+=a['counts'][:,columns].sum(axis=0,dtype=np.int64)
            f['voltage']=voltage[:,columns].round(3).tolist()
            f['plastic_initial']=a['initial_weights'][selected].tolist()
            for raw,key in (('weights','plastic_weights'),('u','plastic_u'),('w','plastic_w')):
                f[key]=a[raw][:,selected].tolist()
        hashes[path.name]=digest(path)
        if 'activity_state' in old:
            if boundaries is None:raise ValueError('Full boundary arrays required for added neurons')
            path=Path(boundaries)/f'boundary-{i:02}.npz'
            if digest(path)!=old['event']['activity_boundary']['artifact_sha256']:raise ValueError('Boundary hash differs')
            with np.load(path,allow_pickle=False) as a:
                state={}
                for prefix in ('before','after'):
                    for field in ('v','g'):
                        values=a[prefix+'__'+field];key=prefix+'_'+field
                        if (values.shape!=(view['report']['graph']['neurons'],) or not np.isfinite(values).all()
                                or not np.array_equal(values[original_columns],old['activity_state'][key])):
                            raise ValueError('Original boundary state differs')
                        state[key]=values[columns].tolist()
                f['activity_state']=state
            boundary_hashes[path.name]=digest(path)
    for n,total in zip(nodes,totals):n['total_spikes']=int(total)
    result['selection']+=' Explicit recorded connections and endpoints added; original display retained.'
    return result,{'requested_edges':list(map(str,edge_ids)),'added_edges':added_edges,'added_neurons':added_nodes,
                   'trace_sha256':hashes,'boundary_sha256':boundary_hashes,'observations':len(view['frames'])}


def extend(study,trace,reference,recordings,edge_ids,output):
    study,output=Path(study),Path(output)
    if output.exists():raise ValueError('Write a new view; never overwrite a recording')
    report_path=study/'report.json';report=json.loads(report_path.read_text());evidence(report)
    r=report['registration'];catalog={trace_name(i,arm,phase):(i,key,arm,phase)
        for i,key in enumerate(r['cohort']) for arm in r['arms'] for phase in PHASES}
    if trace not in catalog or reference not in catalog:raise ValueError('Trace outside audited study')
    pristine=study/'artifacts/pristine-memory.npz'
    if digest(pristine)!=report['pristine_memory_file_sha256']:raise ValueError('Pristine memory differs')
    with np.load(pristine,allow_pickle=False) as a:baseline=a['weights'].copy()
    graph={**report['graph'],'plastic_edges':len(baseline)}
    views=[]
    for name in (trace,reference):
        _,key,arm,phase=catalog[name];v=json.loads((study/name/'view.json').read_text())
        events=[d['neural'] for d in report['phase_diagnostics'][key][arm][phase]['decisions'] if d['neural']]
        if (v['report']['events']!=events or v['report']['plan_sha256']!=report['plan_sha256']
                or v['report']['graph']!=graph or v['report']['native_build']!=report['native_build']):
            raise ValueError('View differs from audited study')
        views.append(v)
    i,key,arm,phase=catalog[trace]
    trained=r['arms'][arm]['memory']=='paper_trained'
    memory_path=study/f'artifacts/imported/pool{i}-memory.npz' if trained else pristine
    expected=r['source_memories'][key] if trained else {'memory_file_sha256':report['pristine_memory_file_sha256'],'memory_sha256':report['pristine_memory_sha256']}
    if digest(memory_path)!=expected['memory_file_sha256']:raise ValueError('Imported memory file differs')
    with np.load(memory_path,allow_pickle=False) as a:memory={k:a[k].copy() for k in ('weights','u','w')}
    if memory_signature(memory)!=expected['memory_sha256']:raise ValueError('Imported memory arrays differ')
    result,metadata=expand(*views,recordings,edge_ids,baseline,memory,study/f'artifacts/boundaries/pool{i}-{arm}-{phase}')
    result['selection_extension']={'schema':1,'study_report_sha256':digest(report_path),
        'source_view_sha256':digest(study/trace/'view.json'),'reference_view_sha256':digest(study/reference/'view.json'),
        'source_sha256':digest(__file__),**metadata,
        'interpretation':"Selected counts, voltages and weights/u/w come from this run's full recordings. Reference view supplies endpoint labels only. Original events and simulation are unchanged; no inference or new trading result."}
    atomic_json(output,result);return result['selection_extension']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study','recordings','out'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('trace','reference'):p.add_argument('--'+name,required=True)
    p.add_argument('--edge',action='append',required=True)
    a=p.parse_args();print(json.dumps(extend(a.study,a.trace,a.reference,a.recordings,a.edge,a.out),indent=2))


if __name__=='__main__':main()
