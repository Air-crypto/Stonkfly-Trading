"""Audit and export learned memory from captured paper-trader checkpoints.

No propagation, training, account mutation or checkpoint restore occurs here.
Only plastic weights and their stored u/w are exported for fresh-state inference.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sqlite3

import numpy as np

from .core import atomic_json, digest
from .fly_market_study import memory_signature


def array_hash(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


def exposure(db, pool, sleeve, assigned):
    events=[]
    for slot,raw in db.execute("SELECT slot,payload FROM ledger WHERE name='fly' ORDER BY slot"):
        if slot*300+300<assigned:
            continue
        matching=[e for e in json.loads(raw)['sleeves'] if e['sleeve']==sleeve and e.get('pool')==pool]
        if len(matching)>1:raise ValueError('Duplicate ledger sleeve')
        if not matching:continue
        detail=matching[0]['detail'];learning=detail.get('learning_diagnostics')
        if learning is None:continue
        reward=learning['equity_reward_usd']
        if not np.isfinite(reward) or learning['learning_enabled'] is not True:
            raise ValueError('Invalid recorded learning exposure')
        events.append({'slot':slot,'reward':reward,'brain_ms':detail['brain_ms'],
                       'memory_sha256':detail['memory']['sha256']})
    if not events or any(e['brain_ms']!=(i+1)*500 for i,e in enumerate(events)):
        raise ValueError('Ledger does not cover the complete assigned neural history')
    if not any(e['reward']>.01 for e in events) or not any(e['reward']<-.01 for e in events):
        raise ValueError('Checkpoint lacks both positive and negative trading-reward exposure')
    return events


def verify_checkpoint(brain, path, expected_sha256, last_event):
    """Check source/configuration and all arrays, without loading dynamics into brain.

    Platform-specific native binary hashes may differ during this array-only
    export. Kernel source, flags, model, graph and complete configuration must
    match; both binary identities are retained as evidence.
    """
    from stonkfly.neural.brain import MODEL, PARAMETERS
    if digest(path)!=expected_sha256:raise ValueError('Captured checkpoint file hash differs')
    edges=brain.circuit['edges']
    with np.load(path,allow_pickle=False) as saved:
        if set(saved.files)!={'metadata','weight',*brain.fields}:
            raise ValueError('Incomplete checkpoint arrays')
        metadata=json.loads(str(saved['metadata']))
        expected={'model':MODEL,'eta':brain.eta,'parameters':PARAMETERS,
                  'graph_ids_sha256':array_hash(brain.ids),'graph_ptr_sha256':array_hash(brain.ptr),
                  'graph_post_sha256':array_hash(brain.post),'plastic_edges_sha256':array_hash(edges),
                  'configuration_sha256':brain.configuration_signature()}
        if any(metadata.get(k)!=v for k,v in expected.items()):
            raise ValueError('Checkpoint graph or configuration differs')
        recorded=metadata.get('build',{})
        if any(recorded.get(k)!=brain.build.get(k) for k in ('source_sha256','model','flags')):
            raise ValueError('Checkpoint native kernel source or build flags differ')
        if not re.fullmatch('[0-9a-f]{64}',recorded.get('binary_sha256','')):
            raise ValueError('Missing checkpoint native binary identity')
        if metadata.get('weights_frozen') is not False or metadata.get('cursor')!=last_event['brain_ms']*10:
            raise ValueError('Checkpoint clock or learning mode differs from its ledger')
        for key in ('weight',*brain.fields):
            value=saved[key];reference=getattr(brain,key)
            if value.shape!=reference.shape or value.dtype!=reference.dtype:
                raise ValueError('Checkpoint array shape or dtype differs: '+key)
            if value.dtype.kind=='f' and not np.isfinite(value).all():
                raise ValueError('Nonfinite checkpoint array: '+key)
        weights=saved['weight'];mask=np.ones(weights.shape,dtype=bool);mask[edges]=False
        if not np.array_equal(weights[mask],brain.weight[mask]):
            raise ValueError('Checkpoint changed nonplastic graph weights')
        state={'weights':weights[edges].copy(),'u':saved['memory_u'].copy(),'w':saved['memory_w'].copy()}
        effective=(brain.baseline_plastic*(1+state['w'])).astype(brain.weight.dtype)
        if not np.array_equal(state['weights'],effective):
            raise ValueError('Stored efficacy does not reproduce plastic weights')
        if array_hash(state['weights'])!=last_event['memory_sha256']:
            raise ValueError('Checkpoint plastic weights differ from final ledger event')
    return state,metadata


def export(capture_root,output,brain):
    capture_root=Path(capture_root);output=Path(output)
    if output.exists():raise ValueError('Refuse to overwrite an exported training snapshot')
    capture=json.loads((capture_root/'capture.json').read_text())
    if digest(capture_root/'paper.db')!=capture['ledger_sha256']:
        raise ValueError('Captured ledger file differs')
    if len(capture['cohort'])!=2 or len(set(capture['cohort']))!=2 or set(capture['pools'])!=set(capture['cohort']):
        raise ValueError('Preserve both captured cohort pools')
    if not np.isfinite(capture['captured_at']) or capture['captured_at']<=0:
        raise ValueError('Invalid capture time')
    db=sqlite3.connect(f'file:{(capture_root/"paper.db").resolve()}?mode=ro',uri=True)
    result={'schema':1,'capture_sha256':digest(capture_root/'capture.json'),
            'export_source_sha256':digest(__file__),
            'ledger_sha256':capture['ledger_sha256'],'captured_at':capture['captured_at'],
            'graph':{'neurons':len(brain.ids),'edges':len(brain.post),'plastic_edges':len(brain.circuit['edges'])},
            'export_build':brain.build,'pools':{},
            'interpretation':'Verified paper-trading reward exposure, not profitable learning. Only plastic weights/u/w are exported. Inference must start with fresh dynamics. Historical prices, news, inventory and other activity state are not imported.'}
    states={}
    try:
        if db.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise ValueError('Invalid ledger database')
        lanes=json.loads(db.execute("SELECT payload FROM state WHERE name='fly'").fetchone()[0])
        for i,pool in enumerate(capture['cohort']):
            pinned=capture['pools'][pool];lane=lanes[pinned['sleeve']]
            if lane['pool']!=pool or lane['checkpoint']!=pinned['remote_checkpoint'] or lane['assigned']!=pinned['assigned']:
                raise ValueError('Captured assignment differs from ledger state')
            events=exposure(db,pool,pinned['sleeve'],lane['assigned'])
            if events!=pinned['events'] or len(events)!=pinned['observations']:
                raise ValueError('Captured exposure differs from independent ledger reconstruction')
            if lane['checkpoint'].split('/')[-1]!=f"fly-{pinned['sleeve']}-{events[-1]['slot']}.npz":
                raise ValueError('Checkpoint version does not match last observed slot')
            if events[-1]['slot']*300+300>=capture['captured_at']:
                raise ValueError('Capture must follow the complete last observed slot')
            state,metadata=verify_checkpoint(brain,capture_root/f'pool{i}-paper-checkpoint.npz',pinned['checkpoint_sha256'],events[-1])
            states[i]=state
            result['pools'][pool]={'checkpoint_sha256':pinned['checkpoint_sha256'],
                'source_metadata':metadata,'memory_sha256':memory_signature(state),
                'observations':len(events),'positive_rewards':sum(e['reward']>.01 for e in events),
                'negative_rewards':sum(e['reward']<-.01 for e in events),
                'neutral_rewards':sum(abs(e['reward'])<=.01 for e in events),
                'first_slot':events[0]['slot'],'last_slot':events[-1]['slot'],
                'last_brain_ms':events[-1]['brain_ms'],'exposure_sha256':hashlib.sha256(json.dumps(events,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
                'changed_weights':int(np.count_nonzero(state['weights']!=brain.baseline_plastic)),
                'delta_l2':{'weights':float(np.linalg.norm(state['weights'].astype(float)-brain.baseline_plastic)),
                            'u':float(np.linalg.norm(state['u'])),'w':float(np.linalg.norm(state['w']))}}
    finally:db.close()
    output.mkdir(parents=True)
    for i,state in states.items():
        path=output/f'pool{i}-memory.npz';np.savez_compressed(path,**state)
        result['pools'][capture['cohort'][i]]['memory_file_sha256']=digest(path)
    atomic_json(output/'audit.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('capture','out','fly-data'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    from .fly_trace import TraceLab
    result=export(a.capture,a.out,TraceLab(a.fly_data).brain)
    print(json.dumps({key:{k:p[k] for k in ('observations','positive_rewards','negative_rewards','changed_weights','memory_sha256')} for key,p in result['pools'].items()},indent=2))


if __name__=='__main__':main()
