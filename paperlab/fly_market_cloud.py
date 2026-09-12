"""Submit or resume a registered market comparison and audit its saved results."""
import argparse
import copy
import json
from pathlib import Path

from .core import atomic_json,digest
from .fly_market_study import signature,validate
from .fly_market_memory_audit import audit as memory_audit
from .fly_market_input import source_files, development_choice

FILES=('fly_market_study.py','fly_trace.py','fly.py','fly_visual.py','fly_market_restoration.py')
PHASES=('training','development','test')


def validate_registration(envelope,registration):
    plan=validate(envelope['plan'])
    if plan['schema'] not in (4,5,6) or signature(plan)!=envelope['sha256']:raise ValueError('Expected a sealed inference comparison')
    fields=['start','phase_steps','cohort','arms','parent_plan_sha256']
    if plan['schema']==5:fields+=['costs','restoration_timing']
    if plan['schema']==6:fields+=['schema','costs','activity_reset_timing','promotion_rule','recorded_at','previous_test_end','mechanism_report_sha256','parent_report_sha256']
    if any(plan[k]!=registration.get(k) for k in fields):raise ValueError('Plan differs from registration')
    if registration.get('end')!=plan['start']+plan['phase_steps']*900:raise ValueError('Registered endpoint differs')
    if not 0<registration.get('recorded_at',0)<plan['start']+plan['phase_steps']*600:raise ValueError('Registration must precede the test interval')
    return plan


def build_report(envelope,summary,raw,selection):
    plan=envelope['plan'];summary=copy.deepcopy(summary)
    if summary['plan_sha256']!=envelope['sha256'] or set(raw)!=set(plan['cohort']):raise ValueError('Result plan or cohort differs')
    idle=summary['initial_capital']-len(raw)*summary['costs']['capital']
    for pool in raw.values():
        if set(pool)!=set(plan['arms']):raise ValueError('Missing result arm')
        for arm,phases in pool.items():
            if set(phases)!=set(PHASES):raise ValueError('Missing result phase')
            for j,phase in enumerate(PHASES):
                p=phases[phase];rows=p['rows'];start=plan['start']+j*plan['phase_steps']*300
                if [r['decision_ts'] for r in rows]!=[start+i*300 for i in range(plan['phase_steps']+1)]:raise ValueError('Result timeline differs')
                if abs(rows[-1]['equity']-p['equity'])>1e-8:raise ValueError('Terminal equity does not reconcile')
                fills=[r['fill'] for r in rows if r['fill']['status']=='filled']
                if len(fills)!=p['fills'] or abs(sum(float(f['fee']) for f in fills)-p['fees'])>1e-8:raise ValueError('Fills or fees do not reconcile')
    for arm in plan['arms']:
        for phase in PHASES:
            expected=idle+sum(raw[k][arm][phase]['equity'] for k in raw)
            if abs(expected-summary['total_equity'][arm][phase])>1e-8:raise ValueError('Aggregate equity does not reconcile')
    development={arm:summary['total_equity'][arm]['development'] for arm in plan['arms']}
    selected=development_choice(plan,development)
    if selection!=summary['selection'] or selection['selected']!=selected or selection['development_equity']!=development or selection['test_simulated_before_selection']:
        raise ValueError('Selection differs from the development gate')
    summary['phase_diagnostics']={key:{arm:{phase:{**{k:v for k,v in p.items() if k!='rows'},'decisions':[
        {'decision_ts':r['decision_ts'],'quote_ts':r['quote_ts'],'available':r['available'],'terminal':r['terminal'],
         'equity':r['equity'],'fill':r['fill'],'neural':r['event']} for r in p['rows']]} for phase,p in phases.items()}
        for arm,phases in arms.items()} for key,arms in raw.items()}
    summary['verification']={'selection_matches_development_gate':True,'terminal_and_aggregate_equity_reconciled':True,'fills_and_fees_reconciled':True,
        'test_fills_per_arm':{arm:sum(raw[k][arm]['test']['fills'] for k in raw) for arm in plan['arms']},
        'test_fees_per_arm':{arm:sum(raw[k][arm]['test']['fees'] for k in raw) for arm in plan['arms']},
        'test_decision_slots_observed_per_arm':{arm:sum(r['event'] is not None for k in raw for r in raw[k][arm]['test']['rows'] if not r['terminal']) for arm in plan['arms']}}
    return summary


def cloud_run(envelope,registration,output,*,allow_submit=True):
    import modal
    import uuid
    plan=validate_registration(envelope,registration)
    root=Path(output);root.mkdir(parents=True,exist_ok=True);receipt_path=root/'cloud-call.json'
    request_hash=signature({'plan':envelope,'registration':registration})
    for name,value in (('plan.json',envelope),('preregistration.json',registration)):
        path=root/name
        if path.exists() and json.loads(path.read_text())!=value:raise ValueError('Existing output metadata belongs to a different request')
    if not receipt_path.exists():
        if not allow_submit:raise RuntimeError('Missing scheduled receipt; observer cannot submit work')
        receipt={'run_id':'assay-market-'+uuid.uuid4().hex,'status':'submitting','request_sha256':request_hash,'plan_sha256':envelope['sha256']}
        atomic_json(receipt_path,receipt);atomic_json(root/'source-hashes.json',{name:digest(Path(__file__).with_name(name)) for name in source_files(plan['schema'])})
        if not (root/'plan.json').exists():atomic_json(root/'plan.json',envelope)
        if not (root/'preregistration.json').exists():atomic_json(root/'preregistration.json',registration)
        call=modal.Function.from_name('fly-paper-lab','worker',environment_name='main').spawn(debug={'run_id':receipt['run_id'],'market_plan':envelope})
        receipt.update(status='pending',call_id=call.object_id);atomic_json(receipt_path,receipt)
    receipt=json.loads(receipt_path.read_text())
    if receipt['request_sha256']!=request_hash:raise ValueError('Output directory belongs to a different request')
    if 'call_id' not in receipt:raise RuntimeError('Uncertain submission; inspect Modal before any retry')
    print(f"Observing saved call {receipt['call_id']}",flush=True)
    try:result=modal.FunctionCall.from_id(receipt['call_id']).get(timeout=50)
    except TimeoutError:
        print('Same call is pending. Repeat this command to observe it; no resubmission.',flush=True);return
    atomic_json(root/'cloud-result.json',result)
    if result.get('status')!='market_study_completed' or result.get('run_id')!=receipt['run_id']:raise RuntimeError('Unexpected result; no automatic resubmission')
    if result['report']['code_sha256']!=json.loads((root/'source-hashes.json').read_text()):raise ValueError('Executed source differs from submission')
    base='/state/fly-debugger/'+receipt['run_id']
    if result['remote_path']!=base:raise ValueError('Unexpected artifact directory')
    volume=modal.Volume.from_name('fly-paper-lab-state',environment_name='main');artifacts=root/'artifacts'
    def download(name):
        target=artifacts/name;target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            partial=target.with_suffix(target.suffix+'.partial')
            with partial.open('wb') as f:
                for block in volume.read_file(base.removeprefix('/state')+'/'+name):f.write(block)
            partial.replace(target)
        return target
    for name in ('results.json','selection.json'):download(name)
    raw=json.loads((artifacts/'results.json').read_text());selection=json.loads((artifacts/'selection.json').read_text())
    summary=build_report(envelope,result['report'],raw,selection)
    if plan['schema']==5:download('plastic-map.npz')
    if plan['schema']==6:
        download('initial-dynamics.npz');download('neuron-ids.npz')
    for i,key in enumerate(plan['cohort']):
        for arm in plan['arms']:
            download(f'pool{i}-{arm}-training-memory.npz');download(f'pool{i}-{arm}/initial-memory.npz')
            if plan['schema']==5:download(f'pool{i}-{arm}-inference-memory.npz')
            if plan['schema']==6:
                for phase in ('development','test'):
                    events=[r['event'] for r in raw[key][arm][phase]['rows'] if r['event'] is not None]
                    for j in range(1,len(events)+1):download(f'boundaries/pool{i}-{arm}-{phase}/boundary-{j:02}.npz')
            if any(r['event'] is not None for r in raw[key][arm]['test']['rows']):
                view=download(f'pool{i}-{arm}/view.json');data=json.loads(view.read_text())
                if data['report']['native_build']!=summary['native_build'] or data['report']['initial_memory_sha256']!=raw[key][arm]['test']['initial_memory_sha256']:
                    raise ValueError('View does not match its phase provenance')
                folder=root/f'pool{i}-{arm}';folder.mkdir(exist_ok=True)
                atomic_json(folder/'view.json',data);atomic_json(folder/'report.json',data['report'])
                atomic_json(folder/'remote.json',{'remote_path':base+f'/pool{i}-{arm}','call_id':receipt['call_id']})
    audited=memory_audit(envelope,summary,artifacts)
    if plan['schema']==6:
        from .fly_market_input import audit_market_boundaries
        atomic_json(root/'activity-audit.json',audit_market_boundaries(envelope,summary,artifacts))
        summary['verification']['activity_boundaries_audited']=True
    summary['verification'].update(plan_matches_preregistration=True,code_hashes_match_submission=True,memory_checkpoints_audited=True)
    atomic_json(root/'report.json',summary);atomic_json(root/'memory-audit.json',audited)
    receipt.update(status='completed',budget=result['budget']);atomic_json(receipt_path,receipt)
    print(json.dumps({'equity':summary['total_equity'],'selection':selection,'verification':summary['verification'],'budget':result['budget']},indent=2))


def observe_scheduled(registration, output):
    """Attach to the cloud-owned study 08; this path can never spawn a worker."""
    import modal
    from .fly_market_schedule import DIRECTORY
    volume=modal.Volume.from_name('fly-paper-lab-state',environment_name='main')
    def read(name):return json.loads(b''.join(volume.read_file('/'+DIRECTORY+'/'+name)))
    receipt=read('cloud-call.json')
    if receipt.get('dispatch')!='scheduled-worker-once' or not receipt.get('call_id') or receipt.get('status') not in ('pending','completed'):
        raise RuntimeError('Scheduled call is unready, failed or uncertain; inspect its saved receipt, never resubmit')
    envelope=read('plan.json');registered=read('preregistration.json');hashes=read('source-hashes.json')
    if registered!=registration:raise ValueError('Scheduled registration differs from the local registration')
    plan=validate_registration(envelope,registration)
    if receipt['plan_sha256']!=envelope['sha256'] or receipt['request_sha256']!=signature({'plan':envelope,'registration':registration}):
        raise ValueError('Scheduled receipt differs from its sealed request')
    if hashes!={name:digest(Path(__file__).with_name(name)) for name in source_files(plan['schema'])}:
        raise ValueError('Local audit source differs from scheduled execution source')
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    saved=root/'cloud-call.json'
    if saved.exists():
        old=json.loads(saved.read_text())
        if any(old.get(k)!=receipt.get(k) for k in ('call_id','run_id','request_sha256')):
            raise ValueError('Local directory belongs to another call; do not mix results')
    for name,value in (('plan.json',envelope),('preregistration.json',registered),('source-hashes.json',hashes)):
        path=root/name
        if path.exists() and json.loads(path.read_text())!=value:raise ValueError('Existing scheduled metadata differs')
        atomic_json(path,value)
    if not saved.exists():atomic_json(saved,receipt)
    return cloud_run(envelope,registration,root,allow_submit=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--plan',type=Path);mode.add_argument('--scheduled',action='store_true')
    for field in ('registration','out'):p.add_argument('--'+field,type=Path,required=True)
    a=p.parse_args();registration=json.loads(a.registration.read_text())
    if a.scheduled:observe_scheduled(registration,a.out)
    else:cloud_run(json.loads(a.plan.read_text()),registration,a.out)


if __name__=='__main__':main()
