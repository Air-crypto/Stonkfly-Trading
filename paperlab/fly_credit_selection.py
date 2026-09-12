"""Expose a selected connection in the audited online carry/reset recordings."""
import copy
import json
from pathlib import Path

import numpy as np

from .core import atomic_json, digest
from .fly_credit_audit import reconstruct, rule_module
from .fly_trace_memory import enrich_frame

ROOT=Path(__file__).resolve().parents[1]


def expand(view, reference, recordings, baseline, circuit, memory, edge_id):
    """Reference supplies labels only; all activity and memory use this run."""
    for field in ('graph','native_build','upstream_commit'):
        if view['report'].get(field)!=reference['report'].get(field):raise ValueError('Reference graph or native build differs')
    edge=next((e for e in reference['edges'] if str(e['id'])==str(edge_id)),None)
    if edge is None:raise ValueError('Reference does not display this connection')
    pi=edge['plastic_index']
    if (type(pi) is not int or not 0<=pi<len(baseline) or int(circuit['edges'][pi])!=int(edge_id)
            or edge['weight']!=float(baseline[pi])):raise ValueError('Reference connection identity or baseline differs')
    if len(view['frames'])!=3 or [f['event'] for f in view['frames']]!=view['report']['events']:
        raise ValueError('Expected three original observations')
    result=copy.deepcopy(view);nodes=result['nodes'];old_columns=np.array([n['index'] for n in nodes],dtype=np.int64)
    lookup={n['id']:i for i,n in enumerate(nodes)};endpoints=[];added=[]
    for side,field in (('source','pre'),('target','post')):
        node=reference['nodes'][edge[side]]
        if node['index']!=int(circuit[field][pi]):raise ValueError('Reference endpoint index differs')
        if node['id'] not in lookup:
            lookup[node['id']]=len(nodes);nodes.append(copy.deepcopy(node));added.append(node['id'])
        elif nodes[lookup[node['id']]]['index']!=node['index']:raise ValueError('Original endpoint differs')
        endpoints.append(lookup[node['id']])
    desired={**edge,'source':endpoints[0],'target':endpoints[1]}
    present=next((e for e in result['edges'] if str(e['id'])==str(edge_id)),None)
    if present is not None and present!=desired:raise ValueError('Existing connection differs')
    if present is None:result['edges'].append(desired)
    if pi not in result['plastic_selection']:result['plastic_selection'].append(pi)
    selected=np.array(result['plastic_selection'],dtype=np.int64);columns=np.array([n['index'] for n in nodes],dtype=np.int64)
    state={k:v.copy() for k,v in memory.items()};rule=rule_module();totals=np.zeros(len(nodes),dtype=np.int64)
    checks=[];hashes={};recordings=Path(recordings)
    for i,(old,frame) in enumerate(zip(view['frames'],result['frames']),1):
        path=recordings/f'step-{i:02}.npz';boundary=recordings/f'boundary-{i:02}.npz'
        original=enrich_frame(view,old,path)
        if any(old.get(k)!=v for k,v in original.items()):raise ValueError('Original displayed memory differs')
        with np.load(path,allow_pickle=False) as archive,np.load(boundary,allow_pickle=False) as b:
            a={k:archive[k] for k in archive.files};ids=a['neuron_ids']
            if (len(set(columns.tolist()))!=len(columns) or np.any(columns<0) or np.any(columns>=len(ids))
                    or any(str(ids[n['index']])!=n['id'] for n in nodes)
                    or not np.array_equal(a['voltage'][:,old_columns].round(3),old['voltage'])):
                raise ValueError('Selected neuron identity or recorded voltage differs')
            if digest(boundary)!=old['event']['activity_boundary']['artifact_sha256']:
                raise ValueError('Boundary record differs')
            state['kc']=b['after__rate_kc'].copy();state['dan']=b['after__rate_dan'].copy()
            credit,metrics=reconstruct(a,state,circuit,baseline,rule,view['report']['config']['learning'],selected)
            for k in memory:state[k]=a[k][-1].copy()
            frame['counts']=a['counts'][:,columns].tolist();totals+=a['counts'][:,columns].sum(axis=0,dtype=np.int64)
            frame['voltage']=a['voltage'][:,columns].round(3).tolist()
            frame['plastic_initial']=a['initial_weights'][selected].tolist()
            for source,dest in (('weights','plastic_weights'),('u','plastic_u'),('w','plastic_w')):
                frame[dest]=a[source][:,selected].tolist()
            frame['plastic_credit']=credit
            for prefix in ('before','after'):
                for field in ('v','g'):
                    value=b[prefix+'__'+field];key=prefix+'_'+field
                    if not np.array_equal(value[old_columns],old['activity_state'][key]):raise ValueError('Original activity boundary differs')
                    frame['activity_state'][key]=value[columns].tolist()
            checks.append(metrics)
        hashes[path.name]=digest(path);hashes[boundary.name]=digest(boundary)
        enrich_frame(result,frame,path)
    for node,total in zip(nodes,totals):node['total_spikes']=int(total)
    result['selection']+=' Explicit audited online connection added from this run; reference provides labels only.'
    result['credit_audit']={'schema':1,'verification':{'observations':3,'bins':150,'all_weights_exact':True},
        'interpretation':'Every native learning bin reconstructed for this derived selection. Reference activity and memory are never imported.'}
    return result,{'edge':str(edge_id),'added_neurons':added,'trace_and_boundary_sha256':hashes,
        'all_bins_reconstructed':True,'observations':3,'rule':checks}


def create_views(audit_path, artifacts, reference_path, output, edge_id='4110156'):
    audit_path,artifacts,reference_path,output=map(Path,(audit_path,artifacts,reference_path,output))
    if output.exists():raise ValueError('Preserve the original and prior derived views')
    if digest(audit_path)!=digest(ROOT/'reports/fly-credit-reset-audit-01.json'):raise ValueError('Published audit differs')
    audit=json.loads(audit_path.read_text());reference=json.loads(reference_path.read_text())
    rule_path=ROOT/'vendor/stonkfly/stonkfly/neural/rule.py'
    if digest(rule_path)!=audit['code_sha256'][str(rule_path.relative_to(ROOT))]:raise ValueError('Recorded learning rule differs')
    def read_arrays(name):
        if digest(artifacts/name)!=audit['artifact_sha256'][name]:raise ValueError('Audited native artifact differs: '+name)
        with np.load(artifacts/name,allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}
    circuit=read_arrays('circuit.npz');baseline=circuit.pop('baseline');memory=read_arrays('trained-memory.npz')
    metadata={}
    for name in ('trained_online_recorded_carry','trained_online_recorded_reset_rates'):
        for relative in (name+'/view.json',*[name+f'/{kind}-{i:02}.npz' for i in range(1,4) for kind in ('step','boundary')]):
            if digest(artifacts/relative)!=audit['artifact_sha256'][relative]:raise ValueError('Audited recording differs: '+relative)
        original=json.loads((artifacts/name/'view.json').read_text())
        view,checks=expand(original,reference,artifacts/name,baseline,circuit,memory,edge_id)
        view['selection_extension']={'schema':1,'source_view_sha256':digest(artifacts/name/'view.json'),
            'audit_sha256':digest(audit_path),'reference_view_sha256':digest(reference_path),
            'source_sha256':digest(__file__),**checks}
        folder=output/('creditdivergence01-'+name)
        atomic_json(folder/'view.json',view);atomic_json(folder/'report.json',view['report'])
        for i in range(1,4):(folder/f'step-{i:02}.npz').hardlink_to(artifacts/name/f'step-{i:02}.npz')
        metadata[name]={'view_sha256':digest(folder/'view.json'),'added_neurons':checks['added_neurons'],
            'edge':checks['edge'],'observations':3,'bins':150}
    atomic_json(output/'selection.json',metadata)
    return metadata
