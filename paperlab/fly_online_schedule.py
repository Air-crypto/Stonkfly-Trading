"""Once-only study 11 chunks, called after the normal paper cycle."""
import json
from pathlib import Path
import shutil
import sqlite3
import time

from .core import atomic_json, digest
from .fly_market_study import signature
from .fly_online_protocol import ARMS, chunk_name, chunk_order
from .fly_online_study import SOURCE_FILES, select_development
from .fly_paper_inputs import seal, validate
from .fly_paper_protocol import validate_registration

DIRECTORY = 'registered-paper-11'
REGISTRATION = 'fly-market-study-11-preregistration.json'
EXECUTION = ('Witness this registration and pin execution sources in the cloud before development starts. '
    'Seal the full four-hour price/news window before evaluation. Run at most one complete pool/arm/phase per '
    'scheduled worker call after committing its normal paper cycle, within the existing 600-second reservation '
    'and unchanged worker $25 / collector $15 monthly caps. Run eight development chunks before eight test chunks; '
    'persist development selection before any test chunk. Claim each chunk durably with its owning Modal call/input '
    'before compute. Never retry or skip failed/uncertain chunks. Refuse source or registration drift until completion. '
    'Require independent raw-input and full-bin audits before publishing results; no automatic policy promotion.')
PINNED_FILES = {
    'fly-market-study-10.json': 'parent_report_sha256',
    'fly-credit-reset-study-01.json': 'mechanism_report_sha256',
    'fly-credit-reset-audit-01.json': 'mechanism_audit_sha256',
}


def source_hashes(specifications):
    root = Path(__file__).resolve().parents[1]
    files = {f'paperlab/{name}': root/'paperlab'/name for name in (*SOURCE_FILES,'fly_online_schedule.py')}
    files['cloud.py'] = Path(specifications)/'cloud-source.py'
    for path in (root/'vendor/stonkfly').rglob('*'):
        if path.suffix in ('.py','.cpp') and '__pycache__' not in path.parts:
            files[str(path.relative_to(root))] = path
    return {name: digest(path) for name,path in sorted(files.items())}


def verify_inputs(r, specifications, memory):
    a=json.loads((memory/'audit.json').read_text());validate_registration(r,a)
    if r['study']!='11' or r.get('execution_protocol')!=EXECUTION or digest(memory/'audit.json')!=r['training_audit_sha256']:
        raise ValueError('Study 11 training provenance differs')
    for file,field in PINNED_FILES.items():
        if digest(specifications/file)!=r[field]:raise ValueError('Pinned study evidence differs: '+file)
    parent=json.loads((specifications/'fly-market-study-10-plan.json').read_text());validate(parent)
    if (parent['sha256']!=r['parent_plan_sha256'] or parent['plan']['registration']['cohort']!=r['cohort']
            or parent['plan']['registration']['study']!='10'
            or r['development_start']<=parent['plan']['registration']['end']):
        raise ValueError('Parent study or chronological separation differs')
    if json.loads((specifications/'fly-market-study-10.json').read_text())['plan_sha256']!=parent['sha256']:
        raise ValueError('Parent report belongs to another plan')
    for i,key in enumerate(r['cohort']):
        if digest(memory/f'pool{i}-memory.npz')!=r['source_memories'][key]['memory_file_sha256']:
            raise ValueError('Pinned memory file differs')


def completed_chunks(root, envelope):
    """Read committed results in order; uncertain/failed work cannot be skipped."""
    completed={}
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
        if receipt['status']!='completed':
            return completed,None,receipt
        result_path=folder/'artifacts/summary.json'
        if digest(result_path)!=receipt['summary_sha256']:raise ValueError('Committed chunk summary differs')
        result=json.loads(result_path.read_text())
        if (result.get('chunk')!=name or result.get('plan_sha256')!=envelope['sha256']
                or result.get('registration')!=envelope['plan']['registration']
                or result.get('status')!='paper_online_chunk_completed'):
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
    return {'status':'paper_online_study_completed','plan_sha256':envelope['sha256'],
        'registration':envelope['plan']['registration'],'selection':selection,
        'total_equity':equities,'observations':coverage,
        'chunk_sha256':{name:signature(result) for name,result in completed.items()},
        'audited':False,'interpretation':'All chunks captured; independent full-bin and raw-price audits are required before publication. No policy promotion.'}


def execute_due(state, discovery, specifications, memory, *, call_id, input_id, commit, run,
                deadline, allow_compute=True, now=None):
    state,discovery,specifications,memory=map(Path,(state,discovery,specifications,memory))
    registration=specifications/REGISTRATION
    if not registration.exists():return None
    root=state/DIRECTORY;now=time.time() if now is None else now
    r=json.loads(registration.read_text())
    if not call_id or not input_id:raise ValueError('Study execution requires its owning call and input')
    if (root/'missed.json').exists():return {'status':'paper_online_registration_missed'}
    # Completed studies retain their recorded sources; later work may evolve the
    # app without turning every normal trading call into a source-drift failure.
    if (root/'summary.json').exists():
        summary=json.loads((root/'summary.json').read_text())
        if summary.get('status')!='paper_online_study_completed' or summary.get('registration')!=r:
            raise ValueError('Completed study belongs to another registration')
        return {'status':'paper_online_study_completed','summary_sha256':digest(root/'summary.json')}
    verify_inputs(r,specifications,memory)
    sources=source_hashes(specifications);sha=digest(registration);armed_path=root/'armed.json'
    if not armed_path.exists():
        root.mkdir(parents=True,exist_ok=True)
        if now>=r['development_start']:
            atomic_json(root/'missed.json',{'status':'registration_missed','observed_at':now,
                'registration_sha256':sha,'call_id':call_id,'input_id':input_id})
            commit();return {'status':'paper_online_registration_missed'}
        if now<r['recorded_at']:raise ValueError('Cloud clock precedes registration')
        shutil.copyfile(registration,root/'preregistration.json')
        atomic_json(root/'source-hashes.json',sources)
        atomic_json(armed_path,{'armed_at':now,'registration_sha256':sha,
            'registration_signature':signature(r),'source_sha256':signature(sources),
            'call_id':call_id,'input_id':input_id})
        commit()
        print(json.dumps({'event':'paper_online_armed','start':r['development_start'],'end':r['end'],'call_id':call_id}),flush=True)
    armed=json.loads(armed_path.read_text())
    if (armed['registration_sha256']!=sha or armed['registration_signature']!=signature(r)
            or not r['recorded_at']<=armed['armed_at']<r['development_start']
            or armed['source_sha256']!=signature(sources)
            or json.loads((root/'source-hashes.json').read_text())!=sources
            or json.loads((root/'preregistration.json').read_text())!=r):
        raise ValueError('Prospective registration or executed source changed')
    if now<r['end']:
        return {'status':'paper_online_collecting','development_start':r['development_start'],'test_start':r['test_start'],'end':r['end']}
    if not allow_compute or deadline-time.monotonic()<450:
        return {'status':'paper_online_waiting_for_worker_time'}
    claim=root/'sealing.json'
    if not claim.exists():
        archive=discovery/'universe-snapshot.db';news=state/'meme-pools-v1/news.db'
        if not archive.exists() or not news.exists():return {'status':'paper_online_waiting_for_snapshot'}
        db=sqlite3.connect(f'file:{archive.resolve()}?mode=ro',uri=True)
        try:last=db.execute("SELECT max(json_extract(payload,'$.observed')) FROM observations").fetchone()[0]
        finally:db.close()
        if last is None or last<r['end']:return {'status':'paper_online_waiting_for_snapshot'}
        receipt={'status':'claimed','call_id':call_id,'input_id':input_id,'claimed_at':now,'registration_sha256':sha}
        atomic_json(claim,receipt);commit()
        try:
            shutil.copyfile(archive,root/'universe.db');shutil.copyfile(news,root/'news.db')
            envelope=seal(root/'universe.db',root/'news.db',root/'preregistration.json',memory/'audit.json',
                specifications/'fly-market-study-10-plan.json',root/'plan.json')
            receipt.update(status='completed',plan_sha256=envelope['sha256']);atomic_json(claim,receipt);commit()
        except Exception as exc:
            receipt.update(status='failed',error_type=type(exc).__name__,error=str(exc)[:300])
            atomic_json(claim,receipt);commit();raise
    sealed=json.loads(claim.read_text())
    if sealed['status']!='completed':
        return {'status':'paper_online_seal_unresolved','receipt':sealed}
    envelope=json.loads((root/'plan.json').read_text());validate(envelope)
    if sealed['plan_sha256']!=envelope['sha256'] or envelope['plan']['registration']!=r:
        raise ValueError('Sealed plan differs from cloud registration')
    completed,next_chunk,unresolved=completed_chunks(root,envelope)
    if unresolved:return {'status':'paper_online_chunk_unresolved','receipt':unresolved,'completed_chunks':len(completed)}
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
        if (selection_receipt['selection_sha256']!=digest(path)
                or selection_receipt['development_chunk_sha256']!=selection['development_chunk_sha256']
                or not selection_receipt.get('call_id') or not selection_receipt.get('input_id')):
            raise ValueError('Selection receipt differs')
    if next_chunk is None:
        summary=aggregate(envelope,completed,selection);atomic_json(root/'summary.json',summary);commit()
        return {'status':summary['status'],'completed_chunks':16,'summary_sha256':digest(root/'summary.json')}
    if deadline-time.monotonic()<450:return {'status':'paper_online_waiting_for_worker_time','completed_chunks':len(completed)}
    stage,pool,arm=next_chunk;name=chunk_name(stage,pool,arm);folder=root/'chunks'/name;output=folder/'artifacts'
    receipt={'status':'claimed','chunk':name,'plan_sha256':envelope['sha256'],'call_id':call_id,'input_id':input_id,
        'claimed_at':time.time(),'source_sha256':signature(sources),'dispatch':'scheduled-worker-after-paper-cycle',
        'remote_path':str(output)}
    atomic_json(folder/'receipt.json',receipt);commit()
    print(json.dumps({'event':'paper_online_chunk_started',**receipt}),flush=True)
    try:
        result=run(envelope,memory,root/'news.db',state/'fly-data',output,stage=stage,pool_index=pool,arm=arm,
            development=development if stage=='test' else None,seconds=min(480,deadline-time.monotonic()-30))
        if (result.get('status')!='paper_online_chunk_completed' or result.get('chunk')!=name
                or result.get('plan_sha256')!=envelope['sha256'] or result.get('registration')!=r
                or result.get('code_sha256')!={file:sources['paperlab/'+file] for file in SOURCE_FILES}
                or json.loads((output/'summary.json').read_text())!=result):
            raise ValueError('Chunk result or executed source differs')
        receipt.update(status='completed',completed_at=time.time(),summary_sha256=digest(output/'summary.json'))
        atomic_json(folder/'receipt.json',receipt);commit()
        return {'status':'paper_online_chunk_completed','chunk':name,'completed_chunks':len(completed)+1,'receipt':receipt}
    except Exception as exc:
        receipt.update(status='failed',error_type=type(exc).__name__,error=str(exc)[:300])
        atomic_json(folder/'receipt.json',receipt);commit();raise
