"""Prepare a future study 12 registration and read its existing cloud state."""
import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
import zipfile

from .core import atomic_json,digest
from .fly_market_study import signature
from .fly_paper_protocol import MEMORY_FIELDS,NEWS
from .fly_rate_inputs import REFERENCE_FILES,verify_reference
from .fly_rate_protocol import ARMS,COHORT_POLICY,HYPOTHESIS,INFERENCE,SELECTION,EXECUTION,chunk_name,chunk_order
from .fly_rate_schedule import DIRECTORY,REGISTRATION,source_hashes
from .multi import DEX_COSTS

APP='fly-paper-rate-market-12'
ROOT=Path(__file__).resolve().parents[1]


def prepare(registration,reference,*,start=None,now=None):
    registration,reference=Path(registration),Path(reference);now=time.time() if now is None else now
    start=math.ceil((now+900)/300)*300 if start is None else start
    if type(start) not in (int,float) or not math.isfinite(start) or start%300 or start<now+600:
        raise ValueError('Allow at least ten minutes to witness a future aligned registration')
    if registration.exists() or reference.exists():raise ValueError('Preserve an existing registration or reference')
    archive=ROOT/'reports/fly-rate-checkpoint-12.zip'
    completed=json.loads((ROOT/'reports/fly-rate-checkpoint-execution-12.json').read_text())
    if digest(archive)!=completed['archive']['sha256']:
        raise ValueError('Published checkpoint archive changed')
    with zipfile.ZipFile(archive) as z:
        if set(z.namelist())!=set(completed['archive']['members']):raise ValueError('Checkpoint archive members differ')
        reference.mkdir(parents=True)
        for name in z.namelist():
            if Path(name).name!=name:raise ValueError('Unsafe checkpoint member path')
            (reference/name).write_bytes(z.read(name))
            if digest(reference/name)!=completed['archive']['members'][name]:raise ValueError('Checkpoint member digest differs')
    shutil.copyfile(ROOT/'reports/fly-online-result-11.json',reference/'parent-report.json')
    shutil.copyfile(ROOT/'reports/fly-learning-scale-audit-01.json',reference/'mechanism-audit.json')
    audit=json.loads((reference/'audit.json').read_text());capture=json.loads((reference/'capture.json').read_text())
    parent=json.loads((reference/'parent-report.json').read_text())
    r={'schema':4,'kind':'paper_checkpoint_comparison','study':'12','arms':ARMS,
       'phase_steps':24,'decision_seconds':300,'costs':asdict(DEX_COSTS),
       'hypothesis':HYPOTHESIS,'inference_protocol':INFERENCE,'selection_rule':SELECTION,
       'news_protocol':NEWS,'cohort_policy':COHORT_POLICY,'python_hash_seed':'0','execution_protocol':EXECUTION,
       'recorded_at':now,'development_start':start,'test_start':start+7200,'end':start+14400,
       'training_cutoff':max((p['last_slot']+1)*300 for p in audit['pools'].values()),
       'checkpoint_capture_at':audit['captured_at'],'parent_end':parent['registration']['end'],
       'cohort':capture['cohort'],'source_memories':{k:{f:v[f] for f in MEMORY_FIELDS} for k,v in audit['pools'].items()},
       'preparation_source_sha256':digest(__file__)}
    r.update({field:digest(reference/name) for name,field in REFERENCE_FILES.items()})
    verify_reference(r,audit,reference);atomic_json(registration,r)
    return {'status':'rate_registration_prepared_not_yet_cloud_witnessed','registration':r,
            'registration_sha256':digest(registration),'source_sha256':source_hashes()}


def verify_arming(files,file_hashes,registration,expected_sources):
    """Compare cloud witness to exact locally prepared bytes and execution sources."""
    path=Path(registration)
    if not path.is_file():return {'verified':False,'reason':'Local registration is unavailable'}
    r=json.loads(path.read_text());armed=files.get('armed.json',{});sources=files.get('source-hashes.json')
    at=armed.get('armed_at')
    checks={'registration_content':files.get('preregistration.json')==r,
            'registration_bytes':file_hashes.get('preregistration.json')==digest(path)==armed.get('registration_sha256'),
            'registration_signature':armed.get('registration_signature')==signature(r),
            'source_content':sources==expected_sources,
            'source_signature':armed.get('source_sha256')==signature(expected_sources),
            'prospective':type(at) in (int,float) and math.isfinite(at) and r['recorded_at']<=at<r['development_start'],
            'owner':bool(armed.get('call_id') and armed.get('input_id'))}
    return {'verified':all(checks.values()),'checks':checks}


async def _read(volume,path):
    async with asyncio.timeout(25):
        pieces=[]
        async for block in volume.read_file.aio('/'+DIRECTORY+'/'+path):pieces.append(block)
        data=b''.join(pieces)
        return json.loads(data),hashlib.sha256(data).hexdigest()


def status(output):
    """Read existing metadata and owning handles; never dispatch or restart work."""
    import modal
    volume=modal.Volume.from_name('fly-paper-lab-state',environment_name='main')
    async def collect():
        names=('armed.json','preregistration.json','source-hashes.json','sealing.json',
               'selection.json','selection-receipt.json','summary.json','last-call.json','costs.json','halt.json','missed.json')
        found=await asyncio.gather(*(_read(volume,n) for n in names),return_exceptions=True)
        result={};errors={};hashes={}
        for name,value in zip(names,found):
            if isinstance(value,(FileNotFoundError,modal.exception.NotFoundError)):continue
            if isinstance(value,BaseException):errors[name]={'type':type(value).__name__,'message':str(value)[:200]}
            else:result[name],hashes[name]=value
        if result.get('sealing.json',{}).get('status')=='completed':
            receipts=['chunks/'+chunk_name(*chunk)+'/receipt.json' for chunk in chunk_order()]
            values=await asyncio.gather(*(_read(volume,n) for n in receipts),return_exceptions=True)
            for name,value in zip(receipts,values):
                if isinstance(value,(FileNotFoundError,modal.exception.NotFoundError)):continue
                if isinstance(value,BaseException):errors[name]={'type':type(value).__name__,'message':str(value)[:200]}
                else:result[name],hashes[name]=value
        return result,errors,hashes
    files,errors,hashes=asyncio.run(collect())
    fn=modal.Function.from_name(APP,'worker',environment_name='main');stats=fn.get_current_stats()
    owner=modal.Dict.from_name('fly-paper-lab-writers',environment_name='main').get('worker')
    result={'observed_at':time.time(),'files':files,'file_sha256':hashes,'read_errors':errors,
            'worker_stats':{'runners':stats.num_total_runners,'running_inputs':stats.num_running_inputs,'backlog':stats.backlog},
            'shared_owner':owner,'cloud_submissions':0}
    if 'armed.json' in files:
        result['arming_verification']=verify_arming(files,hashes,ROOT/'reports'/REGISTRATION,source_hashes())
        result['arming_verified']=result['arming_verification']['verified']
    atomic_json(output,result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','status'))
    p.add_argument('--registration',type=Path,default=ROOT/'reports'/REGISTRATION)
    p.add_argument('--reference',type=Path,default=ROOT/'runs/rate-market-reference-12')
    p.add_argument('--out',type=Path,required=True);p.add_argument('--start',type=float)
    a=p.parse_args()
    result=prepare(a.registration,a.reference,start=a.start) if a.action=='prepare' else status(a.out)
    if a.action=='prepare':atomic_json(a.out,result)
    if a.action=='status':
        print(json.dumps({k:result.get(k) for k in ('observed_at','arming_verified','worker_stats','read_errors')},indent=2))
    else:print(json.dumps({k:result[k] for k in ('status','registration_sha256')},indent=2))


if __name__=='__main__':main()
