"""Independently reconcile exported weights against the locked graph and checkpoint.

Array operations only: no native brain, propagation, cloud call or training.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .core import atomic_json, digest
from .fly_market_activity import array_hash
from .fly_rate_checkpoint_cloud import verify_download


def verify_weights(full_weight, expected_weight, edges, baseline, memory):
    if full_weight.shape!=expected_weight.shape or full_weight.dtype!=expected_weight.dtype:
        raise ValueError('Checkpoint full weight layout differs')
    for key in ('weights','u','w'):
        if memory[key].shape!=baseline.shape or not np.isfinite(memory[key]).all():
            raise ValueError('Invalid exported memory shape or values')
    if not np.array_equal(full_weight[edges],memory['weights']):
        raise ValueError('Exported weights differ from the full checkpoint')
    mask=np.ones(len(full_weight),dtype=bool);mask[edges]=False
    if not np.array_equal(full_weight[mask],expected_weight[mask]):
        raise ValueError('Checkpoint changed nonplastic graph weights')
    if not np.array_equal(memory['weights'],(baseline*(1+memory['w'])).astype(full_weight.dtype)):
        raise ValueError('Stored efficacy does not reconstruct exported weights')
    if any(np.any(memory[key]<-.9) or np.any(memory[key]>1) for key in ('u','w')):
        raise ValueError('Exported memory violates the registered efficacy bounds')
    return {'plastic_edges':len(edges),'nonplastic_edges':int(mask.sum()),
            'changed_weights':int(np.count_nonzero(memory['weights']!=baseline)),
            'delta_l2':{'weights':float(np.linalg.norm(memory['weights'].astype(float)-baseline)),
                        'u':float(np.linalg.norm(memory['u'])),'w':float(np.linalg.norm(memory['w']))}}


def audit(root,data,out):
    from .fly_credit_audit import graph_arrays
    root,data,out=map(Path,(root,data,out))
    if out.exists():raise ValueError('Preserve an earlier checkpoint audit')
    result=json.loads((root/'cloud-result.json').read_text()); verified=verify_download(root,result)
    ids,circuit,baseline=graph_arrays(data)
    from stonkfly.neural.common import annotations
    types=annotations(ids).type.fillna('')
    with np.load(data/'graph.npz',allow_pickle=False) as graph:
        expected=graph['weight'].copy();ptr=graph['ptr'];post=graph['post']
    # Match the declared original visual adapter, independently of checkpoint data.
    for i in np.flatnonzero(types.str.startswith('R8')):
        edges=np.arange(ptr[i],ptr[i+1]);edges=edges[types.iloc[post[edges]].eq('aMe12').to_numpy()]
        expected[edges]=np.abs(expected[edges])
    checks={}
    for i,key in enumerate(result['cohort']):
        recorded=result['training_audit']['pools'][key];metadata=recorded['source_metadata']
        identities={'graph_ids_sha256':array_hash(ids),'graph_ptr_sha256':array_hash(ptr),
                    'graph_post_sha256':array_hash(post),'plastic_edges_sha256':array_hash(circuit['edges'])}
        if any(metadata[k]!=v for k,v in identities.items()):raise ValueError('Checkpoint graph identity differs')
        with np.load(root/f'artifacts/export/pool{i}-memory.npz',allow_pickle=False) as z:
            memory={k:z[k] for k in z.files}
        with np.load(root/f'artifacts/capture/pool{i}-paper-checkpoint.npz',allow_pickle=False) as z:
            checked=verify_weights(z['weight'],expected,circuit['edges'],baseline,memory)
        if checked['changed_weights']!=recorded['changed_weights'] or any(
                not np.isclose(v,recorded['delta_l2'][k],rtol=1e-12,atol=1e-12)
                for k,v in checked['delta_l2'].items()):
            raise ValueError('Reported memory-change metrics differ')
        checks[key]={**checked,**identities}
    report={'status':'rate_checkpoint_arrays_audited','call_id':result['call_id'],
            'cloud_result_sha256':digest(root/'cloud-result.json'),
            'training_audit_sha256':digest(root/'artifacts/export/audit.json'),
            'cohort_selection_verified':verified['cohort'], 'graph_sha256':digest(data/'graph.npz'),
            'pools':checks,'source_sha256':{'paperlab/'+p.name:digest(p) for p in (
                Path(__file__),Path(__file__).with_name('fly_rate_checkpoint_cloud.py'),
                Path(__file__).with_name('fly_credit_audit.py'))},
            'native_constructions':0,'neural_observations':0,'cloud_submissions':0,
            'interpretation':'All plastic exports and nonplastic full-checkpoint weights reconcile '
                'against the locked graph, with original visual-adapter corrections. This proves '
                'checkpoint integrity, not profitable learning.'}
    atomic_json(out,report);return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','fly-data','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(audit(a.root,a.fly_data,a.out),indent=2))


if __name__=='__main__':main()
