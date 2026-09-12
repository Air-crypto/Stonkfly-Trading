"""Controlled activity-state resets over recorded market inputs; no account or orders."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .core import atomic_json, digest
from .fly import UPSTREAM_COMMIT
from .fly_market_pulse import SOURCE, input_sequences
from .fly_market_restoration import matches
from .fly_market_study import learned_state, memory_signature, restore_learned, signature, validate as validate_market
from .fly_trace import TraceLab

MODES=('carry','full','visual_filters','adaptation')
# Reproduce both controls before running any changed state boundary.
ARMS={**{m+'_carry':{'memory':m,'state_reset':'carry'} for m in ('pristine','trained')},
      **{m+'_'+s:{'memory':m,'state_reset':s} for s in MODES[1:] for m in ('pristine','trained')}}
RESET_FIELDS={'visual_filters':('luminance','r8_light'),'adaptation':('adaptation',)}
FILES=('fly_market_activity.py','fly_market_activity_audit.py','fly_market_pulse.py','fly_market_restoration.py',
       'fly_market_study.py','fly_trace.py','fly.py','cloud_debug.py')
SCALARS=('cursor','sim_ms','total_spikes')
MEMORY_FIELDS=('memory_u','memory_w')


def validate(payload):
    if not isinstance(payload,dict) or set(payload)!={'protocol','reference_json','market_plan'}:
        raise ValueError('Expected protocol, reference JSON and sealed market plan')
    p=payload['protocol'];raw=payload['reference_json'];env=payload['market_plan']
    if not isinstance(p,dict) or p.get('schema')!=1 or p.get('kind')!='post_hoc_market_activity_reset' or p.get('arms')!=ARMS:
        raise ValueError('Requires all eight fixed activity-reset conditions')
    if p.get('inference')!={'learning':False,'reinforcement':'none','extra_current':0,'steps':3}:
        raise ValueError('Inference must be frozen, unstimulated and three observations long')
    if p.get('training_source')!='trained_frozen' or p.get('test_source')!='trained_frozen':
        raise ValueError('Changed reference source')
    if p.get('reset_fields')!={k:list(v) for k,v in RESET_FIELDS.items()} or p.get('reset_timing')!='Between observations only; first observation starts from the same fresh dynamics in every arm.':
        raise ValueError('Changed reset fields or timing')
    if not isinstance(raw,str) or len(raw.encode())>2_000_000 or hashlib.sha256(raw.encode()).hexdigest()!=p.get('reference_report_sha256'):
        raise ValueError('Reference report differs from protocol')
    r=json.loads(raw)
    if not isinstance(env,dict) or set(env)!={'plan','sha256'}:raise ValueError('Expected sealed market envelope')
    plan=validate_market(env['plan'])
    if signature(plan)!=env['sha256'] or r.get('plan_sha256')!=env['sha256'] or p.get('market_plan_sha256')!=env['sha256']:
        raise ValueError('Market provenance mismatch')
    if plan['schema']!=5 or plan['phase_steps']!=3:raise ValueError('Requires a three-step schema-five reference')
    pool=p.get('pool')
    if pool not in plan['cohort'] or pool not in r.get('phase_diagnostics',{}):raise ValueError('Missing registered pool')
    rows=r['phase_diagnostics'][pool]
    for arm,phase in (('trained_frozen','training'),('pristine_frozen','test'),('trained_frozen','test')):
        ds=rows[arm][phase]['decisions'];start=plan['start']+(1800 if phase=='test' else 0)
        if len(ds)!=4 or [d['decision_ts'] for d in ds]!=[start+i*300 for i in range(4)] or not ds[-1]['terminal']:
            raise ValueError('Incomplete reference timeline')
        for i,d in enumerate(ds[:3]):
            if d['terminal'] or not d['available'] or d['neural'] is None:raise ValueError('All three observed reference decisions are required')
            e=d['neural']
            if e['stimulus'] not in ('none','reward','aversive'):raise ValueError('Invalid reference stimulus')
            if phase=='test':
                other=rows['trained_frozen']['test']['decisions'][i]
                if d['quote_ts']!=other['quote_ts'] or e['input_sha256']!=other['neural']['input_sha256']:raise ValueError('Controls saw different inputs')
                if e['stimulus']!='none' or e['plasticity_enabled'] or e['weight_delta_l2']!=0:raise ValueError('Reference is not frozen')
    return p,r,plan


def array_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def dynamic_state(brain):
    return {k:getattr(brain,k).copy() for k in brain.initial if k not in MEMORY_FIELDS}


def apply_boundary(brain, mode, observation, path):
    """Save actual before/after arrays, checking target and non-target state separately."""
    if mode not in MODES or observation not in (1,2,3):raise ValueError('Invalid boundary mode or observation')
    before=dynamic_state(brain);memory_before=learned_state(brain)
    clock_before={k:getattr(brain,k) for k in SCALARS};all_weights_before=array_hash(brain.weight)
    effective='initial' if observation==1 else mode
    targets=tuple(before) if effective=='full' else RESET_FIELDS.get(effective,())
    if any(k not in before for k in targets):raise ValueError('Native reset field is missing')
    if effective=='full':brain.reset(keep_memory=True)
    elif targets:
        for k in targets:getattr(brain,k)[:]=brain.initial[k]
    after=dynamic_state(brain);memory_after=learned_state(brain);clock_after={k:getattr(brain,k) for k in SCALARS}
    for k in before:
        expected=brain.initial[k] if k in targets else before[k]
        if not np.array_equal(after[k],expected):raise AssertionError('Boundary changed an undeclared field: '+k)
    if any(not np.array_equal(memory_before[k],memory_after[k]) for k in memory_before) or array_hash(brain.weight)!=all_weights_before:
        raise AssertionError('Activity reset changed synaptic memory or weights')
    if clock_after!=({k:0 for k in SCALARS} if effective=='full' else clock_before):raise AssertionError('Boundary clock semantics differ')
    values={prefix+k:v for prefix,state in (('before__',before),('after__',after),('memory_before__',memory_before),('memory_after__',memory_after)) for k,v in state.items()}
    np.savez_compressed(path,**values)
    return {'mode':effective,'requested_mode':mode,'observation':observation,'target_fields':list(targets),
        'before_clock':clock_before,'after_clock':clock_after,'artifact_sha256':digest(path),
        'all_weight_sha256':all_weights_before,'memory_sha256':memory_signature(memory_before),
        'changed_fields':[k for k in before if not np.array_equal(before[k],after[k])],
        'before_sha256':{k:array_hash(v) for k,v in before.items()},'after_sha256':{k:array_hash(v) for k,v in after.items()}}


def run(payload,data,output):
    p,reference,plan=validate(payload);root=Path(output);root.mkdir(parents=True,exist_ok=False);atomic_json(root/'protocol.json',p)
    lab=TraceLab(data);b=lab.brain
    if b.build!=reference['native_build']:raise ValueError('Native build differs from reference')
    seq=input_sequences(plan,reference,p['pool'],'trained_frozen');rows=reference['phase_diagnostics'][p['pool']]
    deadline=time.monotonic()+420
    def check():
        if time.monotonic()>deadline:raise TimeoutError('Bounded activity assay expired')
    b.reset();pristine=learned_state(b);np.savez_compressed(root/'pristine-memory.npz',**pristine)
    initial=dynamic_state(b);np.savez_compressed(root/'initial-dynamics.npz',**initial)
    if not all(k in initial for fields in RESET_FIELDS.values() for k in fields):raise ValueError('Missing native state fields')
    b.eta=.001;b.weights_frozen=False;lab.fly.controller.s=replace(lab.fly.controller.s,learning=True);training=[]
    for rgb,row in seq['training']:
        check();e=lab.fly.controller.observe(rgb,row['neural']['stimulus'])
        if not matches(e,row['neural']):raise AssertionError('Original training did not reproduce')
        training.append(e)
    trained=learned_state(b)
    if memory_signature(trained)!=rows['trained_frozen']['training']['final_memory_sha256']:raise AssertionError('Training memory did not reproduce')
    np.savez_compressed(root/'trained-memory.npz',**trained)
    reports={};summary={}
    for name,arm in ARMS.items():
        check();memory=pristine if arm['memory']=='pristine' else trained
        restore_learned(b,memory);b.eta=.001;b.weights_frozen=True;lab.fly.controller.s=replace(lab.fly.controller.s,learning=False)
        if any(not np.array_equal(v,getattr(b,k)) for k,v in initial.items()):raise AssertionError('Initial dynamics differ between arms')
        folder=root/name;folder.mkdir();np.savez_compressed(folder/'initial-memory.npz',**memory);events=[];started=time.monotonic()
        for i,(rgb,row) in enumerate(seq['test'],1):
            check();boundary=apply_boundary(b,arm['state_reset'],i,folder/f'boundary-{i:02}.npz')
            e=lab.capture(rgb,'none',folder,i)
            if e['input_sha256']!=row['neural']['input_sha256'] or memory_signature(learned_state(b))!=memory_signature(memory):raise AssertionError('Input or frozen memory changed')
            e.update(phase='replay',phase_step=i,market_decision_ts=row['decision_ts'],input_preset='recorded_market',input_news='none',activity_boundary=boundary)
            events.append(e)
        source=arm['memory']+'_frozen'
        if arm['state_reset']=='carry' and any(not matches(e,d['neural']) for e,d in zip(events,rows[source]['test']['decisions'])):raise AssertionError('Original control did not reproduce: '+name)
        if arm['state_reset']!='carry' and not matches(events[0],reports[arm['memory']+'_carry']['events'][0]):raise AssertionError('First observation changed before any intervention')
        report={'schema':1,'source':SOURCE,'config':{'preset':'recorded_market','view':'original','news':'none','eta':.001,'learning':False,'pulses':'none',**arm},
            'graph':{'neurons':b.n,'edges':len(b.post),'plastic_edges':len(b.circuit['edges'])},'native_build':b.build,'upstream_commit':UPSTREAM_COMMIT,
            'seconds':time.monotonic()-started,'events':events,'initial_memory_sha256':memory_signature(memory),'final_memory_sha256':memory_signature(learned_state(b)),
            'training_memory_sha256':memory_signature(trained),'reference_report_sha256':p['reference_report_sha256'],'plan_sha256':p['market_plan_sha256'],
            'pool':p['pool'],'interpretation':p['interpretation']}
        atomic_json(folder/'report.json',report);atomic_json(folder/'view.json',lab.export_view(folder,report));reports[name]=report
        summary[name]={'actions':[e['side'] for e in events],'gate_spikes':[e['gate_spikes'] for e in events],'difference_hz':[e['difference_hz'] for e in events]}
        print(f"activity_study arm={name} {json.dumps(summary[name])}",flush=True)
    result={'status':'market_activity_study_completed','protocol':p,'training_events':training,'reports':reports,'summary':summary,
        'initial_dynamics_sha256':digest(root/'initial-dynamics.npz'),'code_sha256':{n:digest(Path(__file__).with_name(n)) for n in FILES}}
    atomic_json(root/'summary.json',result);return result


def cloud_run(payload,output):
    import modal
    import uuid
    validate(payload);root=Path(output);root.mkdir(parents=True,exist_ok=True);receipt_path=root/'cloud-call.json'
    if not receipt_path.exists():
        receipt={'run_id':'assay-activity-'+uuid.uuid4().hex,'status':'submitting','payload_sha256':signature(payload)}
        atomic_json(receipt_path,receipt);atomic_json(root/'source-hashes.json',{n:digest(Path(__file__).with_name(n)) for n in FILES})
        call=modal.Function.from_name('fly-paper-lab','worker',environment_name='main').spawn(debug={'run_id':receipt['run_id'],'activity_plan':payload})
        receipt.update(status='pending',call_id=call.object_id);atomic_json(receipt_path,receipt)
    receipt=json.loads(receipt_path.read_text())
    if receipt['payload_sha256']!=signature(payload):raise ValueError('Output directory belongs to another payload')
    if 'call_id' not in receipt:raise RuntimeError('Uncertain submission; inspect Modal before retrying')
    print(f"Observing saved call {receipt['call_id']}",flush=True)
    try:result=modal.FunctionCall.from_id(receipt['call_id']).get(timeout=50)
    except TimeoutError:print('Same call pending; repeat this observer without resubmission.',flush=True);return
    atomic_json(root/'cloud-result.json',result)
    if result.get('status')!='market_activity_study_completed' or result.get('run_id')!=receipt['run_id']:raise RuntimeError('Unexpected result; no automatic resubmission')
    report=result['report'];base='/state/fly-debugger/'+receipt['run_id']
    if report['protocol']!=payload['protocol'] or report['code_sha256']!=json.loads((root/'source-hashes.json').read_text()) or result['remote_path']!=base:raise ValueError('Executed protocol, source or path differs')
    volume=modal.Volume.from_name('fly-paper-lab-state',environment_name='main')
    def download(name):
        path=root/'artifacts'/name;path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            partial=path.with_suffix('.partial')
            with partial.open('wb') as f:
                for block in volume.read_file(base.removeprefix('/state')+'/'+name):f.write(block)
            partial.replace(path)
        return path
    for name in ('initial-dynamics.npz','pristine-memory.npz','trained-memory.npz'):download(name)
    for arm in ARMS:
        folder=root/arm;folder.mkdir(exist_ok=True);download(arm+'/initial-memory.npz')
        view=json.loads(download(arm+'/view.json').read_text())
        if view['report']!=report['reports'][arm] or [f['event'] for f in view['frames']]!=view['report']['events']:raise ValueError('View differs from reported provenance')
        for i in range(1,4):download(arm+f'/boundary-{i:02}.npz')
        atomic_json(folder/'view.json',view);atomic_json(folder/'report.json',view['report']);atomic_json(folder/'remote.json',{'remote_path':base+'/'+arm,'call_id':receipt['call_id']})
    from .fly_market_activity_audit import audit
    atomic_json(root/'audit.json',audit(report,payload,root/'artifacts'))
    atomic_json(root/'summary.json',report);receipt.update(status='completed',budget=result['budget']);atomic_json(receipt_path,receipt)
    print(json.dumps({'summary':report['summary'],'budget':result['budget']},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    pack=sub.add_parser('pack')
    for field in ('protocol','reference','plan','out'):pack.add_argument('--'+field,type=Path,required=True)
    for name in ('run','cloud'):
        cmd=sub.add_parser(name);cmd.add_argument('--payload',type=Path,required=True);cmd.add_argument('--out',type=Path,required=True)
        if name=='run':cmd.add_argument('--fly-data',type=Path,required=True)
    a=p.parse_args()
    if a.command=='pack':
        payload={'protocol':json.loads(a.protocol.read_text()),'reference_json':a.reference.read_text(),'market_plan':json.loads(a.plan.read_text())};validate(payload);atomic_json(a.out,payload)
    else:
        payload=json.loads(a.payload.read_text());cloud_run(payload,a.out) if a.command=='cloud' else run(payload,a.fly_data,a.out)


if __name__=='__main__':main()
