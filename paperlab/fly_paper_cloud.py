"""Observe and audit a registered paper study. This command cannot submit compute."""
import argparse
import json
from pathlib import Path

from .core import atomic_json,digest
from .fly_market_study import signature
from .fly_paper_inputs import validate
from .fly_paper_study import SOURCE_FILES,PHASES,trace_name
from .fly_paper_audit import audit
from .fly_paper_price_audit import audit_prices
from .fly_paper_schedule import DIRECTORY,directory


def observe(registration,output):
    import modal
    from modal.exception import NotFoundError
    volume=modal.Volume.from_name('fly-paper-lab-state',environment_name='main')
    remote_directory=directory(registration.get('study','09'))
    def read(name):return json.loads(b''.join(volume.read_file('/'+remote_directory+'/'+name)))
    try:
        receipt=read('cloud-call.json')
    except (FileNotFoundError,NotFoundError):
        print('No scheduled claim yet; repeat this read-only observer after the next worker cycle.',flush=True)
        return
    if receipt.get('dispatch')!='scheduled-worker-once' or not receipt.get('call_id') or receipt.get('status') not in ('pending','completed'):
        raise RuntimeError('Scheduled comparison unready, failed or uncertain; never submit a replacement')
    if registration.get('study')=='10':
        armed=read('armed.json')
        if armed['registration_signature']!=signature(registration) or not registration['recorded_at']<=armed['armed_at']<registration['development_start']:
            raise ValueError('Cloud did not witness this registration before development')
    envelope=read('plan.json');p=validate(envelope)
    if p['registration']!=registration or read('preregistration.json')!=registration:raise ValueError('Scheduled registration differs')
    if receipt['plan_sha256']!=envelope['sha256'] or receipt['request_sha256']!=signature({'plan':envelope,'registration':registration}):raise ValueError('Scheduled receipt differs from sealed request')
    sources=read('source-hashes.json')
    if sources!={name:digest(Path(__file__).with_name(name)) for name in SOURCE_FILES}:raise ValueError('Local audit source differs from execution source')
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    for name,value in (('plan.json',envelope),('preregistration.json',registration),('source-hashes.json',sources)):
        path=root/name
        if path.exists() and json.loads(path.read_text())!=value:raise ValueError('Output belongs to a different comparison')
        atomic_json(path,value)
    receipt_path=root/'cloud-call.json'
    if receipt_path.exists():
        old=json.loads(receipt_path.read_text())
        if any(old.get(k)!=receipt.get(k) for k in ('run_id','call_id','request_sha256')):raise ValueError('Output belongs to another call')
    else:atomic_json(receipt_path,receipt)
    if registration.get('study')=='10':atomic_json(root/'armed.json',armed)
    print('Observing saved call '+receipt['call_id'],flush=True)
    try:result=modal.FunctionCall.from_id(receipt['call_id']).get(timeout=50)
    except TimeoutError:
        print('Same call is pending; repeat this observer without submitting compute.',flush=True);return
    atomic_json(root/'cloud-result.json',result)
    base='/state/fly-debugger/'+receipt['run_id']
    if result.get('status')!='paper_checkpoint_study_completed' or result.get('run_id')!=receipt['run_id'] or result.get('remote_path')!=base or result['report']['code_sha256']!=sources:
        raise ValueError('Unexpected completed result or executed source')
    artifacts=root/'artifacts'
    def download(name,remote=None):
        path=artifacts/name;path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            partial=path.with_suffix(path.suffix+'.partial')
            with partial.open('wb') as f:
                for block in volume.read_file(remote or base.removeprefix('/state')+'/'+name):f.write(block)
            partial.replace(path)
        return path
    prices=audit_prices(envelope,download('universe.db','/'+remote_directory+'/universe.db'))
    atomic_json(root/'price-audit.json',prices)
    for name in ('results.json','selection.json','news.db','pristine-memory.npz','initial-dynamics.npz','neuron-ids.npz','imported/audit.json'):download(name)
    raw=json.loads((artifacts/'results.json').read_text())
    for i,key in enumerate(registration['cohort']):
        download(f'imported/pool{i}-memory.npz')
        for name in registration['arms']:
            for stage in PHASES:
                trace=trace_name(i,name,stage);download(trace+'/initial-memory.npz')
                events=[row['event'] for row in raw[key][name][stage]['rows'] if row['event'] is not None]
                folder=f'boundaries/pool{i}-{name}-{stage}'
                for j in range(1,len(events)+1):download(folder+f'/boundary-{j:02}.npz')
                download(folder+'/final-counts.npz')
                if events:
                    download(trace+'/view.json')
                    if 'recipient_current' in registration['arms'][name]:
                        for j in range(1,len(events)+1):download(trace+f'/current-{j:02}.npz')
    report,checked=audit(envelope,result['report'],artifacts)
    report['verification']['code_hashes_match_submission']=True
    report['verification']['prices_reconstructed_from_snapshot']=True
    if registration.get('study')=='10':
        report['verification']['prospective_cloud_registration_verified']=True
        report['verification']['native_recipient_current_verified']=True
        report['cloud_registration']=armed
    report['price_audit']=prices
    atomic_json(root/'report.json',report);atomic_json(root/'audit.json',checked)
    # Only expose recordings after all input, ledger, memory and boundary audits pass.
    for i,key in enumerate(registration['cohort']):
        for name in registration['arms']:
            for stage in PHASES:
                trace=trace_name(i,name,stage);path=artifacts/trace/'view.json'
                if not path.exists():continue
                view=json.loads(path.read_text());dest=root/trace
                atomic_json(dest/'view.json',view);atomic_json(dest/'report.json',view['report'])
                atomic_json(dest/'remote.json',{'remote_path':base+'/'+trace,'call_id':receipt['call_id']})
    receipt.update(status='completed',budget=result['budget']);atomic_json(receipt_path,receipt)
    print(json.dumps({'selection':report['selection'],'equity':report['total_equity'],'verification':report['verification'],'budget':result['budget']},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('registration','out'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();observe(json.loads(a.registration.read_text()),a.out)


if __name__=='__main__':main()
