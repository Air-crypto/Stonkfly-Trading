"""Verify training-to-inference memory using saved market-study checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .fly_market_study import memory_signature, signature, validate


def read_memory(path):
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files)!={'weights','u','w'}:
            raise ValueError('Unexpected memory checkpoint arrays')
        state={k:saved[k].copy() for k in saved.files}
    if any(v.ndim!=1 or v.dtype.kind!='f' or not np.isfinite(v).all() for v in state.values()):
        raise ValueError('Invalid memory checkpoint values')
    if not state['weights'].size or len({v.shape for v in state.values()})!=1:
        raise ValueError('Memory checkpoint dimensions differ')
    return state


def audit(envelope, report, artifacts):
    plan=validate(envelope['plan']);root=Path(artifacts)
    if signature(plan)!=envelope['sha256'] or report['plan_sha256']!=envelope['sha256']:
        raise ValueError('Memory audit plan mismatch')
    if set(report['phase_diagnostics'])!=set(plan['cohort']):
        raise ValueError('Memory audit cohort mismatch')
    results={};hashes={}
    def checkpoint(name):
        path=root/name;hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        return read_memory(path)
    mapping=None
    if plan['schema']==5:
        path=root/'plastic-map.npz';hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path,allow_pickle=False) as saved:
            if set(saved.files)!={'edge_ids','post_ids'}:raise ValueError('Invalid plastic mapping')
            mapping={k:saved[k].copy() for k in saved.files}
        if any(v.ndim!=1 or v.dtype.kind not in 'iu' for v in mapping.values()) or mapping['edge_ids'].shape!=mapping['post_ids'].shape:
            raise ValueError('Invalid plastic mapping dimensions')
        if len(np.unique(mapping['edge_ids']))!=len(mapping['edge_ids']):raise ValueError('Duplicate mapped plastic edge')
    for i,pool in enumerate(plan['cohort']):
        phases=report['phase_diagnostics'][pool]
        if set(phases)!=set(plan['arms']):raise ValueError('Missing memory audit arms')
        pristine=checkpoint(f'pool{i}-pristine_frozen-training-memory.npz')
        baseline=memory_signature(pristine);results[pool]={}
        for name,arm in plan['arms'].items():
            rows=phases[name];trained=checkpoint(f'pool{i}-{name}-training-memory.npz')
            initial=checkpoint(f'pool{i}-{name}/initial-memory.npz')
            trained_hash=memory_signature(trained)
            source=rows['training'].get('training_compute_source',name)
            reused=rows['training'].get('training_compute_reused',False)
            if mapping is not None:
                if source not in plan['arms'] or reused!=(source!=name):raise ValueError('Invalid shared training source')
                other=plan['arms'][source]
                if any(arm.get(k,False)!=other.get(k,False) for k in ('eta','train','view','reinforcement_only')):
                    raise ValueError('Shared training crossed a recipe boundary')
                if trained_hash!=phases[source]['training']['final_memory_sha256'] or phases[source]['training'].get('training_compute_reused',False):
                    raise ValueError('Shared training checkpoint differs from its source')
            if rows['training']['initial_memory_sha256']!=baseline:
                raise ValueError('Training did not start from common pristine memory')
            if trained_hash!=rows['training']['final_memory_sha256']:
                raise ValueError('Training checkpoint hash differs from phase result')
            targets=arm.get('restore_post_ids',[]);mask=np.zeros(trained['weights'].shape,dtype=bool)
            expected=trained
            if mapping is not None:
                posts=mapping['post_ids'].astype(str)
                if posts.shape!=trained['weights'].shape or any(t not in set(posts) for t in targets):raise ValueError('Restoration target absent or mapping dimensions differ')
                mask=np.isin(posts,targets)
                if any(trained[k].shape!=pristine[k].shape for k in pristine):raise ValueError('Memory dimensions differ')
                expected={k:np.where(mask,pristine[k],trained[k]) for k in trained}
                prepared=checkpoint(f'pool{i}-{name}-inference-memory.npz')
                if any(not np.array_equal(prepared[k],expected[k]) for k in expected):raise ValueError('Prepared inference memory changed undeclared connections or failed restoration')
            expected_hash=memory_signature(expected)
            if any(not np.array_equal(initial[k],expected[k]) for k in expected):
                raise ValueError('Test checkpoint does not match declared inference memory')
            for phase in ('development','test'):
                if rows[phase]['initial_memory_sha256']!=expected_hash:
                    raise ValueError('Inference did not start from declared memory')
                if mapping is not None:
                    restoration=rows[phase].get('restoration') or {}
                    required={'post_ids':targets,'edge_ids':[str(e) for e in mapping['edge_ids'][mask]],'edge_count':int(mask.sum()),
                              'training_memory_sha256':trained_hash,'inference_memory_sha256':expected_hash}
                    if restoration!=required:raise ValueError('Declared restoration differs from checkpoint mapping')
            for phase in ('training','development','test'):
                frozen=not arm['train' if phase=='training' else 'online']
                if frozen and rows[phase]['initial_memory_sha256']!=rows[phase]['final_memory_sha256']:
                    raise ValueError('Frozen phase changed synaptic memory')
            if any(trained[k].shape!=pristine[k].shape for k in pristine):
                raise ValueError('Training and pristine checkpoint dimensions differ')
            results[pool][name]={
                'training_memory_sha256':trained_hash,
                'training_compute_source':source,'training_compute_reused':reused,
                'inference_memory_sha256':expected_hash,'restored_post_ids':targets,'restored_edge_count':int(mask.sum()),
                'inference_delta_from_training_l2':{k:float(np.linalg.norm(expected[k].astype(np.float64)-trained[k])) for k in trained},
                'training_changed_memory':trained_hash!=baseline,
                'training_delta_l2':{k:float(np.linalg.norm(trained[k].astype(np.float64)-pristine[k])) for k in pristine},
                'training_changed_weights':int(np.count_nonzero(trained['weights']!=pristine['weights'])),
                'development_initial_hash_matches_training':expected_hash==trained_hash,'test_checkpoint_matches_training':expected_hash==trained_hash,
                'development_initial_hash_matches_inference':True,'test_checkpoint_matches_inference':True,
                'frozen_inference_reported_memory_unchanged':True if not arm['online'] else None}
    return {'plan_sha256':envelope['sha256'],'checkpoint_sha256':hashes,'pools':results,
            'interpretation':'Checkpoint equality verifies saved weights and efficacy memory; it does not prove useful learning or profitable trading.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();result=audit(json.loads(a.plan.read_text()),json.loads(a.report.read_text()),a.artifacts)
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['pools'],indent=2))


if __name__=='__main__':main()
