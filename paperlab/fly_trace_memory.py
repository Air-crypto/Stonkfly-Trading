"""Add recorded per-connection memory to a new copy of an existing compact view."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from .core import atomic_json,digest


def enrich_frame(view, frame, path):
    """Verify a full trace against its compact frame before exposing memory values."""
    with np.load(path,allow_pickle=False) as a:
        ids=a['neuron_ids'];edges=a['plastic_edges'];selection=np.asarray(view['plastic_selection'],dtype=np.int64)
        if ids.ndim!=1 or edges.ndim!=1 or len(set(ids))!=len(ids) or len(set(edges))!=len(edges):raise ValueError('Invalid retained identity map')
        if len(ids)!=view['report']['graph']['neurons'] or len(edges)!=view['report']['graph']['plastic_edges']:raise ValueError('Retained graph dimensions differ')
        if len(set(selection))!=len(selection) or np.any(selection<0) or np.any(selection>=len(edges)):raise ValueError('Invalid plastic selection')
        nodes=view['nodes'];columns=np.asarray([n['index'] for n in nodes],dtype=np.int64)
        if np.any(columns<0) or np.any(columns>=len(ids)) or any(str(ids[i])!=str(n['id']) for i,n in zip(columns,nodes)):raise ValueError('Displayed neuron identities differ')
        if a['plastic_pre'].shape!=edges.shape or a['plastic_post'].shape!=edges.shape:raise ValueError('Invalid endpoint map')
        for e in view['edges']:
            pi=e['plastic_index']
            if pi is None:continue
            if pi not in selection or str(edges[pi])!=str(e['id']):raise ValueError('Displayed edge identity differs')
            pre,post=int(a['plastic_pre'][pi]),int(a['plastic_post'][pi])
            if not 0<=pre<len(ids) or not 0<=post<len(ids):raise ValueError('Endpoint outside retained graph')
            if str(ids[pre])!=str(nodes[e['source']]['id']) or str(ids[post])!=str(nodes[e['target']]['id']):raise ValueError('Displayed edge endpoints differ')
        times=a['ms'];counts=a['counts'];n=len(times)
        if not np.array_equal(times,frame['times_ms']) or n!=50 or times[-1]!=frame['event']['brain_ms'] or not np.all(np.diff(times)==10):raise ValueError('Recorded time bins differ')
        if counts.shape!=(n,len(ids)) or counts.dtype.kind not in 'iu' or np.any(counts<0):raise ValueError('Invalid recorded spike counts')
        totals=counts.sum(axis=0,dtype=np.int64)
        if np.any(totals>np.iinfo(np.int32).max) or hashlib.sha256(totals.astype(np.int32).tobytes()).hexdigest()!=frame['event']['spike_sha256']:raise ValueError('Whole-network spike counts differ')
        if not np.array_equal(counts[:,columns],frame['counts']):raise ValueError('Displayed spike counts differ')
        for name in ('weights','u','w'):
            if a[name].shape!=(n,len(edges)) or not np.isfinite(a[name]).all():raise ValueError('Invalid recorded memory dimensions or values')
        if a['initial_weights'].shape!=edges.shape:raise ValueError('Initial weight dimensions differ')
        if not np.array_equal(a['weights'][:,selection],frame['plastic_weights']) or not np.array_equal(a['initial_weights'][selection],frame['plastic_initial']):raise ValueError('Displayed weights differ')
        if not np.allclose(np.linalg.norm(a['u'],axis=1),frame['memory_u_l2'],rtol=1e-6,atol=1e-8) or not np.allclose(np.linalg.norm(a['w'],axis=1),frame['memory_w_l2'],rtol=1e-6,atol=1e-8):raise ValueError('Recorded memory norms differ')
        return {'plastic_u':a['u'][:,selection].tolist(),'plastic_w':a['w'][:,selection].tolist()}


def enrich(view_path,recordings,output,allow_partial=False):
    view_path=Path(view_path);root=Path(recordings);output=Path(output)
    if output.resolve()==view_path.resolve() or output.exists():raise ValueError('Write a new view; never overwrite the original recording')
    view=json.loads(view_path.read_text());result=copy.deepcopy(view);hashes={};included=[];missing=[]
    if len(view['frames'])!=len(view['report']['events']) or any(f['event']!=e for f,e in zip(view['frames'],view['report']['events'])):raise ValueError('Frame events differ from the original report')
    if any('plastic_u' in f or 'plastic_w' in f for f in view['frames']):raise ValueError('View already contains per-connection memory')
    for i,f in enumerate(view['frames']):
        name=f'step-{i+1:02}.npz';path=root/name
        if not path.exists():
            if not allow_partial:raise ValueError('Missing full trace: '+name)
            missing.append(i+1);continue
        result['frames'][i].update(enrich_frame(view,f,path));hashes[name]=digest(path);included.append(i+1)
    if not included:raise ValueError('No full traces available to enrich')
    result['memory_enrichment']={'schema':1,'source_view_sha256':digest(view_path),'trace_sha256':hashes,
        'included_observations':included,'missing_observations':missing,
        'interpretation':'Per-connection u/w copied from verified recorded arrays. Original report, spikes, inputs and weights preserved. No new simulation.'}
    atomic_json(output,result);return result['memory_enrichment']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('view','recordings','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--allow-partial',action='store_true',help='Explicitly leave unavailable observations without memory columns')
    a=p.parse_args();print(json.dumps(enrich(a.view,a.recordings,a.out,a.allow_partial),indent=2))


if __name__=='__main__':main()
