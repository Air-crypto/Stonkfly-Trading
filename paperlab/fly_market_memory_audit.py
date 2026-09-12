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
    for i,pool in enumerate(plan['cohort']):
        phases=report['phase_diagnostics'][pool]
        if set(phases)!=set(plan['arms']):raise ValueError('Missing memory audit arms')
        pristine=checkpoint(f'pool{i}-pristine_frozen-training-memory.npz')
        baseline=memory_signature(pristine);results[pool]={}
        for name,arm in plan['arms'].items():
            rows=phases[name];trained=checkpoint(f'pool{i}-{name}-training-memory.npz')
            initial=checkpoint(f'pool{i}-{name}/initial-memory.npz')
            trained_hash=memory_signature(trained)
            if rows['training']['initial_memory_sha256']!=baseline:
                raise ValueError('Training did not start from common pristine memory')
            if trained_hash!=rows['training']['final_memory_sha256']:
                raise ValueError('Training checkpoint hash differs from phase result')
            if memory_signature(initial)!=trained_hash:
                raise ValueError('Test checkpoint does not retain training memory')
            for phase in ('development','test'):
                if rows[phase]['initial_memory_sha256']!=trained_hash:
                    raise ValueError('Inference did not start from training memory')
            for phase in ('training','development','test'):
                frozen=not arm['train' if phase=='training' else 'online']
                if frozen and rows[phase]['initial_memory_sha256']!=rows[phase]['final_memory_sha256']:
                    raise ValueError('Frozen phase changed synaptic memory')
            if any(trained[k].shape!=pristine[k].shape for k in pristine):
                raise ValueError('Training and pristine checkpoint dimensions differ')
            results[pool][name]={
                'training_memory_sha256':trained_hash,
                'training_changed_memory':trained_hash!=baseline,
                'training_delta_l2':{k:float(np.linalg.norm(trained[k].astype(np.float64)-pristine[k])) for k in pristine},
                'training_changed_weights':int(np.count_nonzero(trained['weights']!=pristine['weights'])),
                'development_initial_hash_matches_training':True,'test_checkpoint_matches_training':True,
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
