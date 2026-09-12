"""Submit the registered experiment once; resume its exact call and audit outputs."""
import argparse
import json
from pathlib import Path
import uuid

from .core import atomic_json
from .fly_credit_reset import source_hashes, validate, verify_reference
from .fly_market_study import signature


def cloud_run(payload, output, reference, data):
    import modal
    p, _, _, _ = validate(payload); root=Path(output); root.mkdir(parents=True,exist_ok=True)
    receipt_path=root/'cloud-call.json'
    if not receipt_path.exists():
        verify_reference(p,reference)
        function=modal.Function.from_name('fly-paper-lab','worker',environment_name='main')
        owner=modal.Dict.from_name('fly-paper-lab-writers',environment_name='main').get('worker')
        stats=function.get_current_stats()
        if owner is not None or stats.num_total_runners or stats.num_running_inputs or stats.backlog:
            raise RuntimeError('Worker has active work; no trace-reset call submitted')
        receipt={'run_id':'assay-credit-'+uuid.uuid4().hex,'status':'submitting','payload_sha256':signature(payload)}
        atomic_json(receipt_path,receipt); atomic_json(root/'payload.json',payload)
        atomic_json(root/'source-hashes.json',source_hashes())
        call=function.spawn(debug={'run_id':receipt['run_id'],'credit_plan':payload})
        receipt.update(status='pending',call_id=call.object_id); atomic_json(receipt_path,receipt)
    receipt=json.loads(receipt_path.read_text())
    if receipt['payload_sha256']!=signature(payload): raise ValueError('Output belongs to another payload')
    if 'call_id' not in receipt: raise RuntimeError('Uncertain submission; inspect the original call, never resubmit')
    if receipt['status']=='completed':
        print('Already audited: '+receipt['call_id'],flush=True); return receipt
    print('Observing saved call '+receipt['call_id'],flush=True)
    try: result=modal.FunctionCall.from_id(receipt['call_id']).get(timeout=50)
    except TimeoutError:
        print('Same call remains pending; repeat this observer to reattach.',flush=True); return receipt
    atomic_json(root/'cloud-result.json',result)
    base='/state/fly-debugger/'+receipt['run_id']; report=result.get('report',{})
    if (result.get('status')!='credit_reset_completed' or result.get('run_id')!=receipt['run_id']
            or result.get('remote_path')!=base or report.get('protocol')!=p
            or report.get('code_sha256')!=json.loads((root/'source-hashes.json').read_text())):
        raise RuntimeError('Terminal result or executed source differs; no automatic resubmission')
    volume=modal.Volume.from_name('fly-paper-lab-state',environment_name='main')
    def download(name):
        path=root/'artifacts'/name;path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():return
        partial=path.with_suffix(path.suffix+'.partial')
        with partial.open('wb') as f:
            for block in volume.read_file(base.removeprefix('/state')+'/'+name):f.write(block)
        partial.replace(path)
    for name in report['artifact_sha256']:download(name)
    for name in p['arms']:
        print('Downloading native evidence: '+name,flush=True)
        for file in ('view.json','report.json','initial-memory.npz','end-01.npz','end-02.npz'):
            download(name+'/'+file)
        for i in range(1,4):
            for file in (f'boundary-{i:02}.npz',f'step-{i:02}.npz',f'input-{i:02}.png'):download(name+'/'+file)
    from .fly_credit_reset_audit import audit
    verified,views=audit(report,payload,root/'artifacts',reference,data)
    atomic_json(root/'audit.json',verified)
    for name,view in views.items():
        folder=root/name;atomic_json(folder/'view.json',view);atomic_json(folder/'report.json',view['report'])
        atomic_json(folder/'remote.json',{'remote_path':base+'/'+name,'call_id':receipt['call_id']})
    atomic_json(root/'summary.json',report)
    receipt.update(status='completed',budget=result['budget']);atomic_json(receipt_path,receipt)
    print(json.dumps({'comparisons':verified['comparisons'],'budget':result['budget']},indent=2),flush=True)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    pack=sub.add_parser('pack')
    for key in ('protocol','pulse-payload','reference','audit','out'):pack.add_argument('--'+key,type=Path,required=True)
    cloud=sub.add_parser('cloud')
    for key in ('payload','reference-recordings','fly-data','out'):cloud.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.command=='pack':
        if a.out.exists():raise ValueError('Refuse to overwrite a payload')
        payload={'protocol_json':a.protocol.read_text(),'pulse_payload_json':a.pulse_payload.read_text(),
                 'reference_json':a.reference.read_text(),'audit_json':a.audit.read_text()}
        validate(payload);atomic_json(a.out,payload)
    else:cloud_run(json.loads(a.payload.read_text()),a.out,a.reference_recordings,a.fly_data)


if __name__=='__main__':main()
