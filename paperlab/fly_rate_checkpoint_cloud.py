"""Prepare, submit once, and observe the same bounded checkpoint export call."""
import argparse
import asyncio
import json
from pathlib import Path
import time

from .core import atomic_json, digest
from .fly_market_study import signature
from .fly_rate_checkpoint import APP, DIRECTORY, POLICY, select_cohort, source_hashes
from .fly_recording_download import transfer_files

ARTIFACTS = {'request.json', 'selection.json', 'universe.db', 'capture/paper.db',
             'capture/capture.json', 'capture/pool0-paper-checkpoint.npz',
             'capture/pool1-paper-checkpoint.npz', 'export/audit.json',
             'export/pool0-memory.npz', 'export/pool1-memory.npz'}


def prepare(root):
    root=Path(root)
    if root.exists(): raise ValueError('Preserve the existing prepared call')
    root.mkdir(parents=True)
    atomic_json(root/'request.json', {'policy': POLICY, 'source_sha256': source_hashes()})
    return {'status': 'checkpoint_request_prepared', 'request_sha256': digest(root/'request.json')}


def submit(root):
    import modal
    root=Path(root); request=json.loads((root/'request.json').read_text())
    if request != {'policy': POLICY, 'source_sha256': source_hashes()}:
        raise ValueError('Prepared checkpoint sources changed')
    if (root/'submission.json').exists(): raise ValueError('A submission was already attempted; observe it, never resubmit')
    v=modal.Volume.from_name('fly-paper-lab-state', environment_name='main')
    try:
        next(iter(v.read_file('/'+DIRECTORY+'/claim.json')))
    except (FileNotFoundError, modal.exception.NotFoundError): pass
    else: raise ValueError('Remote checkpoint capture already claimed')
    owner=modal.Dict.from_name('fly-paper-lab-writers', environment_name='main').get('worker')
    fn=modal.Function.from_name(APP, 'worker', environment_name='main')
    main=modal.Function.from_name('fly-paper-lab', 'worker', environment_name='main')
    stats=[f.get_current_stats() for f in (fn, main)]
    if owner is not None or any(s.num_total_runners or s.num_running_inputs or s.backlog for s in stats):
        raise RuntimeError('Shared worker is active; no submission made')
    atomic_json(root/'submission.json', {'status': 'submitting', 'at': time.time(),
                                       'request_sha256': signature(request)})
    call=fn.spawn(request)
    receipt={'status': 'submitted', 'call_id': call.object_id, 'submitted_at': time.time(),
             'request_sha256': signature(request)}
    atomic_json(root/'cloud-call.json', receipt)
    return receipt


def terminal_result(result, request, receipt):
    hashes=result.get('artifact_sha256', {})
    if any(not isinstance(h,str) or len(h)!=64 or any(c not in '0123456789abcdef' for c in h) for h in hashes.values()):
        raise ValueError('Invalid artifact digest')
    if (result.get('status') != 'rate_checkpoint_exported' or result.get('call_id') != receipt['call_id']
            or not result.get('input_id') or result.get('request_sha256') != signature(request)
            or result.get('remote_path') != '/state/'+DIRECTORY
            or result.get('neural_observations') != 0 or result.get('paper_orders') != 0
            or result.get('native_reference_constructions') != 1
            or set(result.get('artifact_sha256', {})) != ARTIFACTS):
        raise ValueError('Checkpoint export identity or manifest differs')
    b=result.get('budget', {})
    if (b.get('monthly_limit_usd') != 25 or b.get('nonpreemptible') is not True
            or b.get('price_multiplier') != 3 or not 0 < b.get('reserved_usd', 0) <= .1608936+1e-12):
        raise ValueError('Checkpoint export budget differs')
    return result


def verify_download(root, result):
    """Recheck cohort, ledger exposure and exported arrays without a native brain."""
    import numpy as np
    from .fly_market_study import memory_signature
    root=Path(root); a=root/'artifacts'
    for name, sha in result['artifact_sha256'].items():
        if digest(a/name) != sha: raise ValueError('Downloaded artifact differs: '+name)
    selection=json.loads((a/'selection.json').read_text())
    if (selection != result['selection'] or selection != select_cohort(
            a/'capture/paper.db', a/'universe.db', selection['selected_at'])):
        raise ValueError('Cohort selection does not independently reproduce')
    audit=json.loads((a/'export/audit.json').read_text())
    capture=json.loads((a/'capture/capture.json').read_text())
    if (audit != result['training_audit'] or audit['capture_sha256'] != digest(a/'capture/capture.json')
            or audit['ledger_sha256'] != digest(a/'capture/paper.db') or capture['cohort'] != selection['cohort']):
        raise ValueError('Capture and export provenance differs')
    for i,key in enumerate(capture['cohort']):
        imported=a/f'export/pool{i}-memory.npz'; checkpoint=a/f'capture/pool{i}-paper-checkpoint.npz'
        meta=audit['pools'][key]
        chosen=selection['selected'][i];events=chosen['events']
        if (capture['pools'][key] != {**{k:v for k,v in chosen.items() if k!='key'},
                                    'checkpoint_sha256':meta['checkpoint_sha256']}
                or meta['exposure_sha256'] != signature(events)
                or meta['observations'] != len(events)
                or meta['first_slot'] != events[0]['slot'] or meta['last_slot'] != events[-1]['slot']
                or meta['positive_rewards'] != sum(e['reward']>.01 for e in events)
                or meta['negative_rewards'] != sum(e['reward']<-.01 for e in events)
                or capture['captured_at'] <= (events[-1]['slot']+1)*300):
            raise ValueError('Exported exposure differs from the independently selected ledger')
        if (digest(imported) != meta['memory_file_sha256'] or digest(checkpoint) != meta['checkpoint_sha256']
                or meta['checkpoint_sha256'] != capture['pools'][key]['checkpoint_sha256']):
            raise ValueError('Checkpoint or exported memory bytes differ')
        with np.load(imported, allow_pickle=False) as z:
            state={k:z[k] for k in z.files}
        if set(state) != {'weights','u','w'} or memory_signature(state) != meta['memory_sha256']:
            raise ValueError('Exported memory identity differs')
        with np.load(checkpoint, allow_pickle=False) as z:
            if (json.loads(str(z['metadata'])) != meta['source_metadata']
                    or not np.array_equal(z['memory_u'],state['u'])
                    or not np.array_equal(z['memory_w'],state['w'])):
                raise ValueError('Exported memory does not match its saved checkpoint')
    executed=json.loads((a/'request.json').read_text())['source_sha256']
    report={'status':'checkpoint_download_and_selection_verified', 'call_id':result['call_id'],
            'cohort':capture['cohort'], 'executed_source_sha256':executed,
            'verification_source_sha256':digest(__file__),
            'artifact_sha256':result['artifact_sha256'], 'native_constructions_local':0,
            'neural_observations':0, 'policy_promoted':False,
            'scope':'Local receipt, cohort, exposure and u/w verification; full graph/configuration '
                    'and weight export were verified against the native reference in the cloud.'}
    atomic_json(root/'verification.json',report);return report


def observe(root):
    import modal
    root=Path(root);request=json.loads((root/'request.json').read_text())
    receipt=json.loads((root/'cloud-call.json').read_text())
    if (root/'cloud-result.json').exists(): result=json.loads((root/'cloud-result.json').read_text())
    else:
        try: result=modal.FunctionCall.from_id(receipt['call_id']).get(timeout=50)
        except (TimeoutError, modal.exception.TimeoutError):
            return {'status':'pending','call_id':receipt['call_id'],'action':'Observe this same call again; do not resubmit.'}
        terminal_result(result, request, receipt);atomic_json(root/'cloud-result.json',result)
    terminal_result(result, request, receipt)
    v=modal.Volume.from_name('fly-paper-lab-state', environment_name='main')
    downloaded=asyncio.run(transfer_files(v, '/state/'+DIRECTORY, root/'artifacts',
                                         result['artifact_sha256'], label='rate-checkpoint-12'))
    atomic_json(root/'download.json',downloaded)
    return verify_download(root,result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare','submit','observe'))
    parser.add_argument('--out', type=Path, required=True);args=parser.parse_args()
    print(json.dumps(globals()[args.action](args.out),indent=2))


if __name__=='__main__':main()
