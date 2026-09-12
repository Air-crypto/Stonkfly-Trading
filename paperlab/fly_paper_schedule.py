"""Once-only scheduled execution of registered paper-checkpoint comparisons."""
import json
from pathlib import Path
import shutil
import sqlite3
import time

from .core import atomic_json,digest
from .fly_market_study import signature
from .fly_paper_inputs import seal
from .fly_paper_study import SOURCE_FILES

DIRECTORY='registered-paper-09'


def directory(study):
    if study not in ('09','10'):raise ValueError('Unknown scheduled paper study')
    return 'registered-paper-'+study


def execute_due(state,discovery,specifications,memory,*,call_id,input_id,commit,run,settle_budget,now=None,study="09"):
    state,discovery,specifications,memory=map(Path,(state,discovery,specifications,memory))
    root=state/directory(study);receipt_path=root/'cloud-call.json'
    if receipt_path.exists():return None
    registration_path=specifications/f'fly-market-study-{study}-preregistration.json'
    r=json.loads(registration_path.read_text());now=time.time() if now is None else now
    if r['study']!=study:raise ValueError('Scheduled registration study differs')
    parent='09' if study=='10' else '08'
    if study=='10':
        arm_path=root/'armed.json'
        if not arm_path.exists():
            if now>=r['development_start']:
                root.mkdir(parents=True,exist_ok=True)
                atomic_json(receipt_path,{'status':'registration_missed','observed_at':now,
                    'registration_sha256':digest(registration_path),'reason':'Cloud registration was not witnessed before development; no compute submitted.'})
                commit();return None
            from .fly_paper_protocol import validate_registration
            validate_registration(r,json.loads((memory/'audit.json').read_text()))
            if digest(memory/'audit.json')!=r['training_audit_sha256']:raise ValueError('Training audit differs before arming')
            for file,key in ((f'fly-market-study-{parent}.json','parent_report_sha256'),
                    ('fly-paper-stimulation-study-01.json','mechanism_report_sha256'),
                    ('fly-paper-stimulation-audit-01.json','mechanism_audit_sha256')):
                if digest(specifications/file)!=r[key]:raise ValueError('Pinned activation evidence differs before arming')
            root.mkdir(parents=True,exist_ok=True);shutil.copyfile(registration_path,root/'preregistration.json')
            atomic_json(arm_path,{'armed_at':now,'registration_sha256':digest(registration_path),
                'registration_signature':signature(r),'call_id':call_id,'input_id':input_id})
            commit();print('paper_activation_armed '+json.dumps({'study':study,'armed_at':now,'start':r['development_start']}),flush=True)
        armed=json.loads(arm_path.read_text())
        if armed['registration_sha256']!=digest(registration_path) or armed['registration_signature']!=signature(r) or not r['recorded_at']<=armed['armed_at']<r['development_start']:
            raise ValueError('Prospective cloud registration differs or arrived late')
    if now<r['end']:return None
    archive=discovery/'universe-snapshot.db';news=state/'meme-pools-v1/news.db'
    if not archive.exists() or not news.exists():return None
    db=sqlite3.connect(f'file:{archive.resolve()}?mode=ro',uri=True)
    try:last=db.execute("SELECT max(json_extract(payload,'$.observed')) FROM observations").fetchone()[0]
    finally:db.close()
    if last is None or last<r['end']:return None
    if not call_id or not input_id:raise ValueError('Scheduled comparison requires its owning call/input IDs')
    root.mkdir(parents=True,exist_ok=True)
    receipt={'status':'claimed','call_id':call_id,'input_id':input_id,'claimed_at':now,
             'run_id':'assay-paper-'+study+'-'+digest(registration_path)[:16],
             'registration_sha256':digest(registration_path),'dispatch':'scheduled-worker-once',
             'dispatch_source_sha256':digest(__file__)}
    atomic_json(receipt_path,receipt);commit()
    try:
        if digest(specifications/f'fly-market-study-{parent}.json')!=r['parent_report_sha256']:
            raise ValueError('Pinned parent result differs')
        shutil.copyfile(archive,root/'universe.db');shutil.copyfile(news,root/'news.db')
        shutil.copyfile(registration_path,root/'preregistration.json')
        envelope=seal(root/'universe.db',root/'news.db',root/'preregistration.json',memory/'audit.json',
                      specifications/f'fly-market-study-{parent}-plan.json',root/'plan.json')
        atomic_json(root/'source-hashes.json',{name:digest(Path(__file__).with_name(name)) for name in SOURCE_FILES})
        receipt.update(status='pending',plan_sha256=envelope['sha256'],request_sha256=signature({'plan':envelope,'registration':r}))
        atomic_json(receipt_path,receipt);commit()
        print(json.dumps({'event':'paper_checkpoint_started',**receipt}),flush=True)
        output=state/'fly-debugger'/receipt['run_id']
        summary=run(envelope,memory,root/'news.db',state/'fly-data',output)
        if summary.get('status')!='paper_checkpoint_study_completed':raise ValueError('Unexpected checkpoint study result')
        result={'status':summary['status'],'run_id':receipt['run_id'],'remote_path':str(output),'report':summary,'budget':settle_budget()}
        atomic_json(output/'cloud-result.json',result);atomic_json(root/'cloud-result.json',result)
        receipt.update(status='completed',budget=result['budget']);atomic_json(receipt_path,receipt);commit()
        print(json.dumps({'event':'paper_checkpoint_completed','run_id':receipt['run_id'],'budget':result['budget']}),flush=True)
        return result
    except Exception as exc:
        receipt.update(status='failed',error_type=type(exc).__name__,error=str(exc)[:300]);atomic_json(receipt_path,receipt);commit()
        raise
