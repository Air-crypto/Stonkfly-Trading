"""Check debugger topology and every plotted series against retained arrays.

Load verified graph data only. Never construct or propagate a native brain.
"""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from .core import atomic_json, digest
from .fly import configure
from .fly_market_activity_audit import read_arrays

ROOT=Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:raise ValueError(message)


def load_graph(data):
    configure(data)
    from stonkfly.data import verify
    from stonkfly.neural.common import annotations
    from stonkfly.neural.circuit import identify
    from stonkfly.neural.visual import projection
    verify()
    with np.load(Path(data)/'graph.npz',allow_pickle=False) as a:
        graph=SimpleNamespace(**{k:a[k] for k in ('ids','ptr','post','weight','retina')})
    graph.n=len(graph.ids);ann=annotations(graph.ids);graph.types=ann.type.fillna('').to_numpy()
    graph.circuit=identify(graph);graph.circuit['post']=graph.post[graph.circuit['edges']]
    # Pure annotation/contact mapping; this does not initialize a simulator.
    r8,_,_=projection(graph,ann)
    # The existing visual adapter changes these existing contacts' sign before
    # capture. Audit its declared baseline, rather than the earlier raw sign.
    corrected=[]
    for i in np.flatnonzero(ann.type.fillna('').str.startswith('R8')):
        edges=np.arange(graph.ptr[i],graph.ptr[i+1]);chosen=edges[ann.type.iloc[graph.post[edges]].eq('aMe12').to_numpy()]
        graph.weight[chosen]=np.abs(graph.weight[chosen]);corrected.extend(chosen.tolist())
    graph.groups={'visual':np.unique(np.r_[graph.retina,r8]),'KC':graph.circuit['kc'],
        'DAN':graph.circuit['dan'],'MBON':graph.circuit['mb'],
        'output':np.flatnonzero(ann.type.isin(['DNp20','DNpe017']))}
    graph.source_sha256={name:digest(Path(data)/name) for name in ('graph.npz','annotations.feather','normalized/neurons.feather')}
    graph.corrected_existing_edges=len(corrected)
    return graph


def topology(view, graph):
    require(view['report']['graph']=={'neurons':len(graph.ids),'edges':len(graph.post),
        'plastic_edges':len(graph.circuit['edges'])}, 'Displayed graph dimensions differ')
    nodes=view['nodes'];require(bool(nodes), 'No displayed neurons')
    columns=[n['index'] for n in nodes]
    require(all(type(i) is int and 0<=i<len(graph.ids) for i in columns)
        and len(set(columns))==len(columns), 'Invalid displayed neuron indices')
    group_lookup={int(i):name for name,indices in graph.groups.items() for i in indices}
    for n,i in zip(nodes,columns):
        require(n['id']==str(graph.ids[i]) and n['type']==str(graph.types[i])
            and n['group']==group_lookup.get(i,'other'), 'Displayed neuron identity, type or group differs')
    plastic={int(e):i for i,e in enumerate(graph.circuit['edges'])};seen=set();selection=set()
    for e in view['edges']:
        identity,source,target=e['id'],e['source'],e['target']
        require(type(identity) is int and 0<=identity<len(graph.post) and identity not in seen,
            'Invalid or duplicate displayed connection')
        require(all(type(i) is int and 0<=i<len(nodes) for i in (source,target)), 'Invalid displayed endpoint index')
        pre,post=columns[source],columns[target]
        require(graph.ptr[pre]<=identity<graph.ptr[pre+1] and graph.post[identity]==post, 'Displayed connection endpoints differ')
        pi=plastic.get(identity)
        require(e['plastic_index'] is None if pi is None else type(e['plastic_index']) is int and e['plastic_index']==pi,
            'Displayed plastic connection identity differs')
        require(type(e['weight']) in (int,float) and np.isfinite(e['weight'])
            and float(graph.weight[identity])==e['weight'], 'Displayed baseline connection weight differs')
        seen.add(identity)
        if pi is not None:selection.add(pi)
    selected=set(columns);expected={int(edge) for source in columns
        for edge in range(graph.ptr[source],graph.ptr[source+1]) if int(graph.post[edge]) in selected}
    require(seen==expected, 'Displayed graph omits or adds connections between its selected neurons')
    require(all(type(i) is int for i in view['plastic_selection'])
        and view['plastic_selection']==sorted(selection), 'Displayed plastic selection differs')
    return np.asarray(columns,dtype=np.int64)


def finite_equal(actual, expected, message):
    a=np.asarray(actual);b=np.asarray(expected)
    require(a.shape==b.shape and a.dtype.kind in 'iuf' and np.isfinite(a).all()
        and np.array_equal(a,b),message)


def norm_series(actual, delta, name):
    a=np.asarray(actual,dtype=np.float64);expected=np.linalg.norm(delta.astype(np.float64),axis=1)
    # The exporter reduces float32 weight deltas. Allow the same four-ULP
    # rounding tolerance as the native event audit, never a relative 1% drift.
    tolerance=4*np.spacing(expected.astype(np.float32)).astype(np.float64) if delta.dtype==np.float32 else 1e-12+1e-12*expected
    require(a.shape==expected.shape and np.isfinite(a).all() and np.all(a>=0)
        and np.all(np.abs(a-expected)<=tolerance), 'Displayed norm differs: '+name)
    return float(np.max(np.abs(a-expected),initial=0))


def audit_image(frame, path, rgb):
    require(rgb.dtype==np.uint8 and rgb.ndim==3 and rgb.shape[2]==3, 'Expected sealed RGB image')
    require(hashlib.sha256(rgb.tobytes()).hexdigest()==frame['event']['input_sha256'], 'Input image hash differs')
    for source in (path,io.BytesIO(base64.b64decode(frame['input_png'],validate=True))):
        with Image.open(source) as image:
            require(np.array_equal(np.asarray(image.convert('RGB')),rgb), 'Saved or displayed input differs')


def frame(view, f, a, graph, columns, observation):
    counts=a['counts'];n=len(graph.ids);edges=graph.circuit['edges'];baseline=graph.weight[edges]
    require(np.array_equal(a['neuron_ids'],graph.ids) and np.array_equal(a['plastic_edges'],edges), 'Raw trace identity differs')
    require(counts.shape==(50,n) and counts.dtype.kind in 'iu' and not np.any(counts<0), 'Invalid raw counts')
    require(a['voltage'].shape==(50,n) and np.isfinite(a['voltage']).all(), 'Invalid sampled voltage')
    for key in ('weights','u','w','kc'):
        require(a[key].shape==(50,len(edges)) and np.isfinite(a[key]).all(), 'Invalid raw memory: '+key)
    require(a['dan'].shape==(50,len(graph.circuit['dan'])) and np.isfinite(a['dan']).all(), 'Invalid raw DAN rates')
    require(a['initial_weights'].shape==baseline.shape and np.isfinite(a['initial_weights']).all(), 'Invalid initial weights')
    require(f['step']==observation and np.array_equal(a['ms'],np.arange(10,501,10)+500*(observation-1)), 'Frame ordinal or native times differ')
    finite_equal(f['times_ms'],a['ms'],'Displayed time grid differs')
    finite_equal(f['counts'],counts[:,columns],'Displayed spike counts differ')
    finite_equal(f['voltage'],a['voltage'][:,columns].round(3),'Displayed sampled voltage differs')
    finite_equal(f['total_spikes'],counts.sum(axis=1),'Displayed full-neuron spike curve differs')
    require(set(f['groups'])==set(graph.groups),'Displayed population groups differ')
    for name,indices in graph.groups.items():
        finite_equal(f['groups'][name],counts[:,indices].sum(axis=1),'Displayed population spike curve differs: '+name)
    total=counts.sum(axis=0,dtype=np.int64)
    require(not np.any(total>np.iinfo(np.int32).max)
        and array_digest(total.astype(np.int32))==f['event']['spike_sha256']
        and int(total.sum())==f['event']['total_spikes'], 'Frame event spike totals differ')
    require(f['event']['brain_ms']==a['ms'][-1], 'Frame event time differs')
    errors={};delta=a['weights']-a['initial_weights'];pristine=a['weights']-baseline
    for key,values in (('weight_delta_l2',delta),('weight_from_pristine_l2',pristine),
                       ('memory_u_l2',a['u']),('memory_w_l2',a['w'])):
        errors[key]=norm_series(f[key],values,key)
    for key,values in (('changed_edges',delta),('changed_from_pristine',pristine)):
        finite_equal(f[key],np.count_nonzero(values,axis=1),'Displayed changed-edge curve differs: '+key)
    for key,values in (('kc_mean_hz',a['kc']),('dan_mean_hz',a['dan'])):
        actual=np.asarray(f[key],dtype=np.float64);expected=values.mean(axis=1)
        require(actual.shape==expected.shape and np.isfinite(actual).all()
            and np.allclose(actual,expected,rtol=1e-12,atol=1e-12),'Displayed mean rate trace differs: '+key)
    selection=np.asarray(view['plastic_selection'],dtype=np.int64)
    for key,values in (('plastic_weights',a['weights']),('plastic_u',a['u']),('plastic_w',a['w'])):
        finite_equal(f[key],values[:,selection],'Displayed connection memory differs: '+key)
    finite_equal(f['plastic_initial'],a['initial_weights'][selection],'Displayed initial connection weight differs')
    return total,errors


def array_digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def audit_view(view, root, graph):
    columns=topology(view,graph);root=Path(root);total=np.zeros(len(graph.ids),dtype=np.int64);errors=[]
    require(len(view['frames'])==len(view['report']['events'])==3, 'Expected the three-image mechanism view')
    for i,f in enumerate(view['frames'],1):
        require(f['event']==view['report']['events'][i-1], 'Frame event differs from report')
        a=read_arrays(root/f'step-{i:02}.npz');count,checked=frame(view,f,a,graph,columns,i)
        total+=count;errors.append(checked)
    finite_equal([n['total_spikes'] for n in view['nodes']],total[columns], 'Displayed neuron totals differ')
    return {'observations':3,'bins':150,'selected_neurons':len(columns),'displayed_connections':len(view['edges']),
        'plastic_connections':len(view['plastic_selection']),'all_topology_verified':True,
        'all_plotted_series_verified':True,'norm_reduction_errors':errors}


def run(audit, artifacts, data, output):
    root=Path(artifacts);output=Path(output);audit=Path(audit)
    if output.exists():raise ValueError('Preserve earlier projection audit evidence')
    require(digest(audit)==digest(ROOT/'reports/fly-credit-reset-audit-01.json'),'Use the published parent audit')
    parent=json.loads(audit.read_text());graph=load_graph(data);checks={};hashes={}
    for name in parent['protocol']['arms']:
        for file in ('view.json','step-01.npz','step-02.npz','step-03.npz'):
            relative=name+'/'+file;sha=digest(root/relative)
            require(sha==parent['artifact_sha256'][relative],'Parent artifact differs: '+relative);hashes[relative]=sha
        view=json.loads((root/name/'view.json').read_text());checks[name]=audit_view(view,root/name,graph)
    report={'status':'recorded_view_projections_audited','parent_audit_sha256':digest(audit),
        'source_sha256':digest(__file__),'graph_data_sha256':graph.source_sha256,
        'corrected_existing_visual_edges':graph.corrected_existing_edges,'views':checks,'artifact_sha256':hashes,
        'neural_observations':0,'cloud_submissions':0,
        'interpretation':'Every displayed connection and plotted series checked against retained full arrays and verified graph data. Source firing is a recorded event, not proof of transmission or a causal effect. No new trajectories, training or return claim.'}
    output.mkdir(parents=True);atomic_json(output/'report.json',report);return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('audit','artifacts','fly-data','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();r=run(a.audit,a.artifacts,a.fly_data,a.out)
    print(json.dumps({'status':r['status'],'views':len(r['views']),
        'observations':sum(v['observations'] for v in r['views'].values()),'neural_observations':0},indent=2))


if __name__=='__main__':main()
