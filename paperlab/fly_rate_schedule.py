"""Once-only study 12 chunks under the shared paper worker lease."""
import json
import math
from pathlib import Path
import shutil
import sqlite3
import time

from .core import atomic_json, digest
from .fly_market_study import signature
from .fly_rate_protocol import ARMS, chunk_name, chunk_order
from .fly_rate_study import SOURCE_FILES, select_development
from .fly_rate_inputs import seal, validate
from .fly_rate_protocol import validate_registration

DIRECTORY = 'registered-rate-12'
REGISTRATION = 'fly-rate-market-registration-12.json'
from .fly_rate_protocol import EXECUTION
from .fly_rate_inputs import verify_reference


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    files = {f'paperlab/{name}': root/'paperlab'/name for name in (*SOURCE_FILES,
        'fly_rate_schedule.py', 'fly_rate_runtime.py', 'fly_rate_projection.py', 'fly_rate_cloud.py',
        'fly_recording_download.py', 'budget.py', 'universe.py')}
    files['rate_market_cloud.py'] = root/'rate_market_cloud.py'
    files['cloud.py'] = root/'cloud.py'
    for path in (root/'vendor/stonkfly').rglob('*'):
        if path.suffix in ('.py','.cpp') and '__pycache__' not in path.parts:
            files[str(path.relative_to(root))] = path
    return {name: digest(path) for name,path in sorted(files.items())}


def verify_inputs(r, specifications, memory):
    a=json.loads((memory/'audit.json').read_text());validate_registration(r,a)
    if r['study']!='12' or r.get('execution_protocol')!=EXECUTION:
        raise ValueError('Study 12 execution protocol differs')
    verify_reference(r,a,memory)


def completed_chunks(root, envelope):
    """Read committed results in order; uncertain/failed work cannot be skipped."""
    completed={}; owners=set()
    for stage,pool,arm in chunk_order():
        name=chunk_name(stage,pool,arm);folder=root/'chunks'/name;path=folder/'receipt.json'
        if not path.exists():
            if folder.exists():raise ValueError('Unclaimed chunk directory exists')
            if any((root/'chunks'/chunk_name(*later)).exists() for later in chunk_order()[len(completed)+1:]):
                raise ValueError('Chunks exist beyond an unclaimed condition')
            return completed,(stage,pool,arm),None
        receipt=json.loads(path.read_text())
        if receipt.get('chunk')!=name or receipt.get('plan_sha256')!=envelope['sha256'] or not receipt.get('call_id') or not receipt.get('input_id'):
            raise ValueError('Chunk receipt identity differs')
        if receipt['call_id'] in owners:raise ValueError('Different chunks reused an owning cloud call')
        owners.add(receipt['call_id'])
        if receipt['status']!='completed':
            return completed,None,receipt
        if (any(type(receipt.get(k)) not in (int,float) or not math.isfinite(receipt[k])
                for k in ('claimed_at','completed_at')) or receipt['completed_at']<receipt['claimed_at']):
            raise ValueError('Completed chunk receipt clock differs')
        result_path=folder/'artifacts/summary.json'
        if digest(result_path)!=receipt['summary_sha256']:raise ValueError('Committed chunk summary differs')
        result=json.loads(result_path.read_text())
        if (result.get('chunk')!=name or result.get('plan_sha256')!=envelope['sha256']
                or result.get('registration')!=envelope['plan']['registration']
                or result.get('status')!='paper_rate_chunk_completed'):
            raise ValueError('Committed chunk provenance differs')
        completed[name]=result
    return completed,None,None


def aggregate(envelope, completed, selection):
    if len(completed)!=16:raise ValueError('All sixteen conditions are required')
    equities={};coverage={}
    for arm in ARMS:
        equities[arm]={};coverage[arm]={}
        for stage in ('development','test'):
            rows=[completed[chunk_name(stage,pool,arm)]['outcome'] for pool in range(2)]
            equities[arm][stage]=500+sum(row['equity'] for row in rows)
            coverage[arm][stage]=[sum(r['event'] is not None for r in row['rows'][:-1]) for row in rows]
    metrics={arm:{stage:[completed[chunk_name(stage,pool,arm)]['trading_metrics'] for pool in range(2)]
                  for stage in ('development','test')} for arm in ARMS}
    return {'status':'paper_rate_study_completed','plan_sha256':envelope['sha256'],
        'registration':envelope['plan']['registration'],'selection':selection,
        'total_equity':equities,'observations':coverage,'trading_metrics':metrics,
        'test_coverage_sufficient':all(min(coverage[arm]['test'])>=18 for arm in ARMS),
        'chunk_sha256':{name:signature(result) for name,result in completed.items()},
        'audited':False,'interpretation':'All chunks captured; independent full-bin and raw-price audits are required before publication. No policy promotion.'}


def execute_due(state, discovery, specifications, memory, *, call_id, input_id, commit, run,
                deadline, allow_compute=True, now=None, seal_inputs=None):
    if seal_inputs is None:
        from .fly_rate_runtime import seal_seeded
        seal_inputs=seal_seeded
    state,discovery,specifications,memory=map(Path,(state,discovery,specifications,memory))
    registration=specifications/REGISTRATION
    if not registration.exists():return None
    root=state/DIRECTORY;now=time.time() if now is None else now
    r=json.loads(registration.read_text())
    if not call_id or not input_id:raise ValueError('Study execution requires its owning call and input')
    if (root/'missed.json').exists():return {'status':'paper_rate_registration_missed'}
    # Completed studies retain their recorded sources; later work may evolve the
    # app without turning every normal trading call into a source-drift failure.
    if (root/'summary.json').exists():
        summary=json.loads((root/'summary.json').read_text())
        if summary.get('status')!='paper_rate_study_completed' or summary.get('registration')!=r:
            raise ValueError('Completed study belongs to another registration')
        return {'status':'paper_rate_study_completed','summary_sha256':digest(root/'summary.json')}
    verify_inputs(r,specifications,memory)
    sources=source_hashes();sha=digest(registration);armed_path=root/'armed.json'
    if not armed_path.exists():
        root.mkdir(parents=True,exist_ok=True)
        if now>=r['development_start']:
            atomic_json(root/'missed.json',{'status':'registration_missed','observed_at':now,
                'registration_sha256':sha,'call_id':call_id,'input_id':input_id})
            commit();return {'status':'paper_rate_registration_missed'}
        if now<r['recorded_at']:raise ValueError('Cloud clock precedes registration')
        shutil.copyfile(registration,root/'preregistration.json')
        atomic_json(root/'source-hashes.json',sources)
        atomic_json(armed_path,{'armed_at':now,'registration_sha256':sha,
            'registration_signature':signature(r),'source_sha256':signature(sources),
            'call_id':call_id,'input_id':input_id})
        commit()
        print(json.dumps({'event':'paper_rate_armed','start':r['development_start'],'end':r['end'],'call_id':call_id}),flush=True)
    armed=json.loads(armed_path.read_text())
    if (armed['registration_sha256']!=sha or armed['registration_signature']!=signature(r)
            or not r['recorded_at']<=armed['armed_at']<r['development_start']
            or armed['source_sha256']!=signature(sources)
            or json.loads((root/'source-hashes.json').read_text())!=sources
            or json.loads((root/'preregistration.json').read_text())!=r):
        raise ValueError('Prospective registration or executed source changed')
    if now<r['end']:
        return {'status':'paper_rate_collecting','development_start':r['development_start'],'test_start':r['test_start'],'end':r['end']}
    if not allow_compute or deadline-time.monotonic()<450:
        return {'status':'paper_rate_waiting_for_worker_time'}
    claim=root/'sealing.json'
    if not claim.exists():
        archive=discovery/'universe-snapshot.db';news=state/'meme-pools-v1/news.db'
        if not archive.exists() or not news.exists():return {'status':'paper_rate_waiting_for_snapshot'}
        db=sqlite3.connect(f'file:{archive.resolve()}?mode=ro',uri=True)
        try:last=db.execute("SELECT max(json_extract(payload,'$.observed')) FROM observations").fetchone()[0]
        finally:db.close()
        if last is None or last<r['end']:return {'status':'paper_rate_waiting_for_snapshot'}
        receipt={'status':'claimed','call_id':call_id,'input_id':input_id,'claimed_at':now,'registration_sha256':sha}
        atomic_json(claim,receipt);commit()
        try:
            shutil.copyfile(archive,root/'universe.db');shutil.copyfile(news,root/'news.db')
            envelope=seal_inputs(root/'universe.db',root/'news.db',root/'preregistration.json',memory/'audit.json',
                memory,root/'plan.json')
            receipt.update(status='completed',plan_sha256=envelope['sha256']);atomic_json(claim,receipt);commit()
        except BaseException as exc:
            receipt.update(status='failed',error_type=type(exc).__name__,error=str(exc)[:300])
            atomic_json(claim,receipt);commit();raise
    sealed=json.loads(claim.read_text())
    if sealed['status']!='completed':
        return {'status':'paper_rate_seal_unresolved','receipt':sealed}
    envelope=json.loads((root/'plan.json').read_text());validate(envelope)
    if sealed['plan_sha256']!=envelope['sha256'] or envelope['plan']['registration']!=r:
        raise ValueError('Sealed plan differs from cloud registration')
    completed,next_chunk,unresolved=completed_chunks(root,envelope)
    if unresolved:return {'status':'paper_rate_chunk_unresolved','receipt':unresolved,'completed_chunks':len(completed)}
    if next_chunk is not None and any(json.loads((root/'chunks'/name/'receipt.json').read_text())['call_id']==call_id
                                     for name in completed):
        raise ValueError('Do not capture two chunks in the same owning cloud call')
    development={k:v for k,v in completed.items() if v['stage']=='development'}
    selection=None
    if len(development)==8:
        selection=select_development(envelope,development);path=root/'selection.json'
        if not path.exists():
            atomic_json(path,selection)
            atomic_json(root/'selection-receipt.json',{'selected_at':time.time(),'call_id':call_id,'input_id':input_id,
                'selection_sha256':digest(path),'development_chunk_sha256':selection['development_chunk_sha256']})
            commit()
        if json.loads(path.read_text())!=selection:raise ValueError('Persisted development selection differs')
        receipt_path=root/'selection-receipt.json'
        if not receipt_path.exists():raise ValueError('Uncertain selection receipt; do not start test')
        selection_receipt=json.loads(receipt_path.read_text())
        selected_at=selection_receipt.get('selected_at')
        dev_receipts=[json.loads((root/'chunks'/name/'receipt.json').read_text()) for name in development]
        test_receipts=[json.loads((root/'chunks'/name/'receipt.json').read_text())
                       for name,result in completed.items() if result['stage']=='test']
        if (selection_receipt['selection_sha256']!=digest(path)
                or selection_receipt['development_chunk_sha256']!=selection['development_chunk_sha256']
                or not selection_receipt.get('call_id') or not selection_receipt.get('input_id')
                or type(selected_at) not in (int,float) or not math.isfinite(selected_at)
                or selected_at<max(v['completed_at'] for v in dev_receipts)
                or any(selected_at>v['claimed_at'] for v in test_receipts)):
            raise ValueError('Selection receipt differs')
    if next_chunk is None:
        summary=aggregate(envelope,completed,selection);atomic_json(root/'summary.json',summary);commit()
        return {'status':summary['status'],'completed_chunks':16,'summary_sha256':digest(root/'summary.json')}
    if deadline-time.monotonic()<450:return {'status':'paper_rate_waiting_for_worker_time','completed_chunks':len(completed)}
    stage,pool,arm=next_chunk;name=chunk_name(stage,pool,arm);folder=root/'chunks'/name;output=folder/'artifacts'
    receipt={'status':'claimed','chunk':name,'plan_sha256':envelope['sha256'],'call_id':call_id,'input_id':input_id,
        'claimed_at':time.time(),'source_sha256':signature(sources),'dispatch':'dedicated-rate-worker-shared-lease',
        'remote_path':str(output)}
    atomic_json(folder/'receipt.json',receipt);commit()
    print(json.dumps({'event':'paper_rate_chunk_started',**receipt}),flush=True)
    try:
        result=run(envelope,memory,root/'news.db',state/'fly-data',output,stage=stage,pool_index=pool,arm=arm,
            development=development if stage=='test' else None,seconds=min(480,deadline-time.monotonic()-30))
        if (result.get('status')!='paper_rate_chunk_completed' or result.get('chunk')!=name
                or result.get('plan_sha256')!=envelope['sha256'] or result.get('registration')!=r
                or result.get('code_sha256')!={file:sources['paperlab/'+file] for file in SOURCE_FILES}
                or json.loads((output/'summary.json').read_text())!=result):
            raise ValueError('Chunk result or executed source differs')
        receipt.update(status='completed',completed_at=time.time(),summary_sha256=digest(output/'summary.json'))
        atomic_json(folder/'receipt.json',receipt);commit()
        return {'status':'paper_rate_chunk_completed','chunk':name,'completed_chunks':len(completed)+1,'receipt':receipt}
    except BaseException as exc:
        receipt.update(status='failed',error_type=type(exc).__name__,error=str(exc)[:300])
        atomic_json(folder/'receipt.json',receipt);commit();raise
