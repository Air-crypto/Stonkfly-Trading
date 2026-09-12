"""Once-only scheduled execution of registered paper-checkpoint study 09."""
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


def execute_due(state,discovery,specifications,memory,*,call_id,input_id,commit,run,settle_budget,now=None):
    state,discovery,specifications,memory=map(Path,(state,discovery,specifications,memory))
    root=state/DIRECTORY;receipt_path=root/'cloud-call.json'
    if receipt_path.exists():return None
    registration_path=specifications/'fly-market-study-09-preregistration.json'
    r=json.loads(registration_path.read_text());now=time.time() if now is None else now
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
             'run_id':'assay-paper-09-'+digest(registration_path)[:16],
             'registration_sha256':digest(registration_path),'dispatch':'scheduled-worker-once',
             'dispatch_source_sha256':digest(__file__)}
    atomic_json(receipt_path,receipt);commit()
    try:
        if digest(specifications/'fly-market-study-08.json')!=r['parent_report_sha256']:
            raise ValueError('Pinned parent result differs')
        shutil.copyfile(archive,root/'universe.db');shutil.copyfile(news,root/'news.db')
        shutil.copyfile(registration_path,root/'preregistration.json')
        envelope=seal(root/'universe.db',root/'news.db',root/'preregistration.json',memory/'audit.json',
                      specifications/'fly-market-study-08-plan.json',root/'plan.json')
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
