"""Read-only observation and full-recording audit of scheduled study 11."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time

from .core import atomic_json, digest
from .fly_market_study import signature
from .fly_online_audit import audit_chunk
from .fly_online_protocol import chunk_name, chunk_order
from .fly_online_schedule import DIRECTORY, aggregate, source_hashes
from .fly_online_study import SOURCE_FILES, select_development
from .fly_paper_inputs import validate
from .fly_paper_price_audit import audit_prices


def pin_json(path,value):
    path=Path(path)
    if path.exists() and json.loads(path.read_text())!=value:
        raise ValueError('Saved observation belongs to different evidence: '+str(path))
    atomic_json(path,value)


def expected_sources():
    # Reuse the exact manifest logic with the local worker source as its deployed
    # snapshot. No import or execution of cloud.py occurs here.
    with tempfile.TemporaryDirectory(prefix='fly-online-source-') as temporary:
        shutil.copyfile(Path(__file__).resolve().parents[1]/'cloud.py',Path(temporary)/'cloud-source.py')
        return source_hashes(temporary)


def validate_selection_receipt(selection,receipt,receipts):
    if (not receipt.get('call_id') or not receipt.get('input_id')
            or receipt['development_chunk_sha256']!=selection['development_chunk_sha256']):
        raise ValueError('Selection receipt identity differs')
    development=[r for name,r in receipts.items() if name.startswith('development-')]
    tests=[r for name,r in receipts.items() if name.startswith('test-')]
    if len(development)!=8 or receipt['selected_at']<max(r['completed_at'] for r in development):
        raise ValueError('Selection preceded completed development')
    if any(r['claimed_at']<receipt['selected_at'] for r in tests):
        raise ValueError('Test began before development selection')


def observe(registration,output,*,audit=False,data=None):
    import modal
    from modal.exception import NotFoundError
    if audit and data is None:raise ValueError('Full audit requires the prepared graph data')
    root=Path(output);volume=modal.Volume.from_name('fly-paper-lab-state',environment_name='main')
    def raw(relative):return b''.join(volume.read_file(DIRECTORY+'/'+relative))
    def read(relative):return json.loads(raw(relative))
    def optional(relative):
        try:return read(relative)
        except (FileNotFoundError,NotFoundError):return None
    def finish(progress):
        progress={**progress,'observed_at':time.time()}
        atomic_json(root/'progress.json',progress)
        return progress
    def download(remote,path,expected):
        path=Path(path)
        if path.exists():
            if digest(path)!=expected:raise ValueError('Existing download differs: '+str(path))
            return path
        path.parent.mkdir(parents=True,exist_ok=True);partial=path.with_suffix(path.suffix+'.partial')
        with partial.open('wb') as f:
            for block in volume.read_file(DIRECTORY+'/'+remote):f.write(block)
        if digest(partial)!=expected:raise ValueError('Downloaded artifact hash differs: '+remote)
        partial.replace(path);return path
    armed=optional('armed.json')
    if armed is None:
        missed=optional('missed.json')
        return {'status':'registration_missed' if missed else 'waiting_for_cloud_arming','receipt':missed}
    sources=read('source-hashes.json')
    if (armed['registration_signature']!=signature(registration)
            or not registration['recorded_at']<=armed['armed_at']<registration['development_start']
            or armed['source_sha256']!=signature(sources) or sources!=expected_sources()):
        raise ValueError('Cloud registration or pinned local execution source differs')
    remote_registration=raw('preregistration.json')
    if json.loads(remote_registration)!=registration or hashlib.sha256(remote_registration).hexdigest()!=armed['registration_sha256']:
        raise ValueError('Cloud registration bytes differ')
    root.mkdir(parents=True,exist_ok=True)
    pin_json(root/'armed.json',armed);pin_json(root/'source-hashes.json',sources)
    download('preregistration.json',root/'preregistration.json',armed['registration_sha256'])
    try:arming_result=modal.FunctionCall.from_id(armed['call_id']).get(timeout=0)
    except TimeoutError:return finish({'status':'arming_call_pending','call_id':armed['call_id']})
    if (arming_result.get('study_11') or {}).get('status')!='paper_online_collecting':
        raise ValueError('Arming worker did not return the expected study registration')
    pin_json(root/'arming-call-result.json',arming_result)
    now=time.time()
    phase=('waiting_for_development' if now<registration['development_start'] else
           'development_collection' if now<registration['test_start'] else
           'test_collection' if now<registration['end'] else 'evaluation')
    progress={'status':phase,'armed':armed,'completed_chunks':0,'audited_chunks':0,'total_chunks':16,
        'development_start':registration['development_start'],'test_start':registration['test_start'],'end':registration['end']}
    sealed=optional('sealing.json')
    if sealed is None:return finish(progress)
    if (sealed.get('registration_sha256')!=armed['registration_sha256'] or sealed.get('claimed_at',0)<registration['end']
            or not sealed.get('call_id') or not sealed.get('input_id')
            or sealed.get('status') not in ('claimed','failed','completed')):
        raise ValueError('Sealing did not follow the registered endpoint')
    if sealed['status']!='completed':
        try:result=modal.FunctionCall.from_id(sealed['call_id']).get(timeout=0)
        except TimeoutError:
            return finish({**progress,'status':'sealing_call_pending','receipt':sealed})
        return finish({**progress,'status':'sealing_receipt_unresolved','receipt':sealed,'worker_status':result.get('status')})
    envelope=read('plan.json');p=validate(envelope)
    if sealed['plan_sha256']!=envelope['sha256'] or p['registration']!=registration:
        raise ValueError('Sealed request differs from registration')
    pin_json(root/'sealing.json',sealed);pin_json(root/'plan.json',envelope)
    receipts={};summaries={};audits={};selection=None;selection_receipt=None
    if audit:
        path=download('universe.db',root/'universe.db',p['snapshot_sha256'])
        prices=audit_prices(envelope,path);pin_json(root/'price-audit.json',prices)
    for stage,pool,arm in chunk_order():
        name=chunk_name(stage,pool,arm);remote='chunks/'+name;receipt=optional(remote+'/receipt.json')
        if receipt is None:break
        expected_path='/state/'+DIRECTORY+'/'+remote+'/artifacts'
        if (receipt.get('chunk')!=name or receipt.get('plan_sha256')!=envelope['sha256']
                or receipt.get('source_sha256')!=signature(sources) or receipt.get('remote_path')!=expected_path
                or receipt.get('dispatch')!='scheduled-worker-after-paper-cycle'
                or not receipt.get('call_id') or not receipt.get('input_id')
                or receipt.get('status') not in ('claimed','failed','completed')
                or receipt.get('claimed_at',0)<registration['end']
                or (receipt['status']=='completed' and receipt.get('completed_at',0)<receipt['claimed_at'])):
            raise ValueError('Chunk ownership or request differs')
        for earlier in (root/'receipts').glob(name+'-*.json'):
            old=json.loads(earlier.read_text())
            if any(old.get(k)!=receipt.get(k) for k in ('call_id','input_id','chunk','plan_sha256','claimed_at','remote_path')):
                raise ValueError('A saved chunk was replaced by another call')
        pin_json(root/'receipts'/f"{name}-{receipt['status']}.json",receipt)
        if receipt['status']!='completed':
            # Observing a receipt is not proof its owner is terminal. Reattach its
            # exact handle; never spawn a replacement, even after a timeout.
            try:result=modal.FunctionCall.from_id(receipt['call_id']).get(timeout=0)
            except TimeoutError:
                return finish({**progress,'status':'chunk_call_pending','chunk':name,'call_id':receipt['call_id']})
            return finish({**progress,'status':'chunk_receipt_unresolved','chunk':name,'receipt':receipt,'worker_status':result.get('status')})
        try:result=modal.FunctionCall.from_id(receipt['call_id']).get(timeout=0)
        except TimeoutError:return finish({**progress,'status':'chunk_worker_return_pending','chunk':name,'call_id':receipt['call_id']})
        child=result.get('study_11') or {}
        if child.get('status')!='paper_online_chunk_completed' or child.get('chunk')!=name or child.get('receipt')!=receipt:
            raise ValueError('Saved worker result differs from committed chunk')
        pin_json(root/'call-results'/f'{name}.json',result)
        destination=root/'chunks'/name
        summary_path=download(remote+'/artifacts/summary.json',destination/'summary.json',receipt['summary_sha256'])
        summary=json.loads(summary_path.read_text())
        if (summary.get('status')!='paper_online_chunk_completed' or summary.get('chunk')!=name
                or summary.get('registration')!=registration or summary.get('plan_sha256')!=envelope['sha256']
                or summary.get('stage')!=stage or summary.get('pool_index')!=pool or summary.get('arm')!=arm
                or summary.get('code_sha256')!={f:sources['paperlab/'+f] for f in SOURCE_FILES}):
            raise ValueError('Chunk execution summary differs')
        receipts[name]=receipt;summaries[name]=summary;progress['completed_chunks']=len(summaries)
        if len({r['call_id'] for r in receipts.values()})!=len(receipts):
            raise ValueError('More than one chunk claimed the same worker call')
        if stage=='test':
            selection=read('selection.json');selection_receipt=read('selection-receipt.json')
            development={k:v for k,v in summaries.items() if v['stage']=='development'}
            if select_development(envelope,development)!=selection or summary['selection']!=selection:
                raise ValueError('Test selection differs from completed development')
            if hashlib.sha256(raw('selection.json')).hexdigest()!=selection_receipt['selection_sha256']:
                raise ValueError('Selection file hash differs')
            validate_selection_receipt(selection,selection_receipt,receipts)
            pin_json(root/'selection.json',selection);pin_json(root/'selection-receipt.json',selection_receipt)
        if audit:
            for file,sha in summary['artifact_sha256'].items():
                relative=Path(file)
                if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe artifact path')
                download(remote+'/artifacts/'+file,destination/relative,sha)
            checked,view=audit_chunk(envelope,summary,destination,data)
            atomic_json(root/'audits'/f'{name}.json',checked);audits[name]=checked
            progress['audited_chunks']=len(audits)
            if view:
                folder=root/'views'/name
                atomic_json(folder/'view.json',view);atomic_json(folder/'report.json',view['report'])
                atomic_json(folder/'remote.json',{'remote_path':expected_path+'/trace','call_id':receipt['call_id']})
                # Arbitrary-neuron inspection reuses downloaded arrays rather than
                # making another cloud copy. All links remain under this output.
                for file in (destination/'trace').glob('step-*.npz'):
                    target=folder/file.name
                    if not target.exists():target.hardlink_to(file)
            print(json.dumps({'event':'paper_online_chunk_audited','chunk':name,'observations':checked['verification']['observations']}),flush=True)
    if len(summaries)==16:
        expected=aggregate(envelope,summaries,selection)
        remote_summary=optional('summary.json')
        if remote_summary is not None and remote_summary!=expected:raise ValueError('Cloud aggregate differs from its chunks')
        progress['status']='captured_pending_audit'
        if audit:
            identities={(s['native_build']['binary_sha256'],s['artifact_sha256']['initial-dynamics.npz'],
                s['artifact_sha256']['pristine-memory.npz']) for s in summaries.values()}
            if len(identities)!=1:raise ValueError('Conditions differ in native build or fresh starting state')
            expected['audited']=True;expected['status']='paper_online_study_audited'
            expected['cloud_registration']=armed;expected['selection_receipt']=selection_receipt
            expected['verification']={'all_ledgers_and_feedback_reconstructed':True,'all_full_bin_audits_passed':True,
                'all_raw_prices_reconstructed':True,'all_news_clocks_verified':True,
                'source_and_native_builds_match':True,'selection_precedes_every_test_chunk':True,
                'distinct_completed_worker_calls':16}
            expected['price_audit']=prices
            expected['phase_diagnostics']={key:{arm:{stage:summaries[chunk_name(stage,pool,arm)]['outcome']
                for stage in ('development','test')} for arm in registration['arms']}
                for pool,key in enumerate(registration['cohort'])}
            expected['audit_sha256']={name:digest(root/'audits'/f'{name}.json') for name in audits}
            expected['interpretation']='Registered paper comparison with independently reconstructed inputs, feedback, counts and plasticity. Two assets and four hours do not establish monthly profitability; no policy promoted.'
            atomic_json(root/'report.json',expected)
            progress.update(status=expected['status'],selection=selection['selected'],total_equity=expected['total_equity'])
    return finish(progress)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--audit',action='store_true');parser.add_argument('--fly-data',type=Path)
    args=parser.parse_args()
    print(json.dumps(observe(json.loads(args.registration.read_text()),args.out,audit=args.audit,data=args.fly_data),indent=2))


if __name__=='__main__':main()
