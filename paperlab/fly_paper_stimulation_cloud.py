"""Submit once, resume the same bounded Modal call, and audit before publishing views."""
import argparse
import json
from pathlib import Path
import uuid

from .core import atomic_json, digest
from .fly_market_study import signature
from .fly_paper_stimulation import FILES, protocol, run, validate


def cloud_run(payload, output, *, isolation=False, reference_artifacts=None):
    import modal
    if isolation:
        from .fly_recipient_isolation import FILES as files, validate as validate_isolation
        if reference_artifacts is None:raise ValueError('Isolation requires original audited control recordings')
        p, _, parent_audit, _ = validate_isolation(payload)
        plan_field, expected_status = 'isolation_plan', 'paper_recipient_isolation_completed'
    else:
        p, _, _ = validate(payload);files=FILES
        plan_field, expected_status = 'stimulation_plan', 'paper_stimulation_completed'
    root=Path(output); root.mkdir(parents=True,exist_ok=True)
    receipt_path=root/'cloud-call.json'
    if not receipt_path.exists():
        if isolation:
            from .fly_recipient_isolation_cloud import prior_comparison_complete
            prior_comparison_complete()
            from .fly_recipient_isolation import verify_reference_artifacts
            verify_reference_artifacts(parent_audit, reference_artifacts)
        receipt={'run_id':'assay-recipient-'+uuid.uuid4().hex,'status':'submitting','payload_sha256':signature(payload)}
        atomic_json(receipt_path,receipt)
        atomic_json(root/'source-hashes.json',{n:digest(Path(__file__).with_name(n)) for n in files})
        atomic_json(root/'payload.json',payload)
        call=modal.Function.from_name('fly-paper-lab','worker',environment_name='main').spawn(
            debug={'run_id':receipt['run_id'],plan_field:payload})
        receipt.update(status='pending',call_id=call.object_id); atomic_json(receipt_path,receipt)
    receipt=json.loads(receipt_path.read_text())
    if receipt['payload_sha256']!=signature(payload):raise ValueError('Output belongs to a different payload')
    if 'call_id' not in receipt:raise RuntimeError('Uncertain submission: inspect Modal; do not resubmit')
    if receipt['status']=='completed':
        print('Previously completed and audited: '+receipt['call_id'],flush=True);return receipt
    print('Observing saved call '+receipt['call_id'],flush=True)
    try:result=modal.FunctionCall.from_id(receipt['call_id']).get(timeout=50)
    except TimeoutError:
        print('Same call pending; rerun this observer without resubmission.',flush=True);return receipt
    atomic_json(root/'cloud-result.json',result)
    if result.get('status')!=expected_status or result.get('run_id')!=receipt['run_id']:
        raise RuntimeError('Unexpected terminal result; no automatic resubmission: '+str(result.get('status')))
    report=result['report']; base='/state/fly-debugger/'+receipt['run_id']
    if report['protocol']!=p or report['code_sha256']!=json.loads((root/'source-hashes.json').read_text()) or result['remote_path']!=base:
        raise ValueError('Executed protocol, source code or remote path differs')
    volume=modal.Volume.from_name('fly-paper-lab-state',environment_name='main')
    def download(name):
        path=root/'artifacts'/name;path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            partial=path.with_suffix(path.suffix+'.partial')
            with partial.open('wb') as f:
                for block in volume.read_file(base.removeprefix('/state')+'/'+name):f.write(block)
            partial.replace(path)
        return path
    for name in ('initial-dynamics.npz','neuron-ids.npz','pristine-memory.npz','news.db',
                 'imported/audit.json','imported/pool0-memory.npz','imported/pool1-memory.npz'):
        download(name)
    for name in p['arms']:
        print('Downloading native evidence: '+name,flush=True)
        for file in ('view.json','report.json','initial-memory.npz'):download(name+'/'+file)
        for j in range(1,4):
            for file in (f'boundary-{j:02}.npz',f'current-{j:02}.npz',f'step-{j:02}.npz',f'input-{j:02}.png'):
                download(name+'/'+file)
    if isolation:
        from .fly_recipient_isolation_audit import audit
        verified=audit(report,payload,root/'artifacts',reference_artifacts)
    else:
        from .fly_paper_stimulation_audit import audit
        verified=audit(report,payload,root/'artifacts')
    atomic_json(root/'audit.json',verified)
    # Expose portable recordings only after every condition passes independent audit.
    for name in p['arms']:
        folder=root/name;folder.mkdir(exist_ok=True)
        view=json.loads((root/'artifacts'/name/'view.json').read_text())
        atomic_json(folder/'view.json',view);atomic_json(folder/'report.json',view['report'])
        atomic_json(folder/'remote.json',{'remote_path':base+'/'+name,'call_id':receipt['call_id']})
    atomic_json(root/'summary.json',report)
    receipt.update(status='completed',budget=result['budget']);atomic_json(receipt_path,receipt)
    print(json.dumps({'comparisons':verified['comparisons'],'budget':result['budget']},indent=2),flush=True)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    pack=sub.add_parser('pack')
    for field in ('reference','plan','out'):pack.add_argument('--'+field,type=Path,required=True)
    for name in ('run','cloud'):
        cmd=sub.add_parser(name)
        for field in ('payload','out'):cmd.add_argument('--'+field,type=Path,required=True)
        if name=='run':
            for field in ('memory','news','fly-data'):cmd.add_argument('--'+field,type=Path,required=True)
    a=parser.parse_args()
    if a.command=='pack':
        if a.out.exists():raise ValueError('Refuse to overwrite an experiment payload')
        raw=a.reference.read_text();env=json.loads(a.plan.read_text())
        payload={'protocol':protocol(raw,env),'reference_json':raw,'paper_plan':env}
        validate(payload);atomic_json(a.out,payload)
    else:
        payload=json.loads(a.payload.read_text())
        cloud_run(payload,a.out) if a.command=='cloud' else run(payload,a.memory,a.news,a.fly_data,a.out)


if __name__=='__main__':main()
