"""Restore specified synaptic memory in a matched market replay; no trading account."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .core import atomic_json, digest
from .fly import UPSTREAM_COMMIT
from .fly_market_pulse import SOURCE, input_sequences, same_memory
from .fly_market_study import learned_state, memory_signature, restore_learned, signature, validate as validate_market
from .fly_trace import TraceLab

ARMS={
    'pristine_frozen':{'memory':'pristine','restore_post_ids':[]},
    'trained_frozen':{'memory':'trained','restore_post_ids':[]},
    'restore_10704':{'memory':'trained','restore_post_ids':['10704']},
    'restore_11402':{'memory':'trained','restore_post_ids':['11402']},
    'restore_both':{'memory':'trained','restore_post_ids':['10704','11402']},
}
FILES=('fly_market_restoration.py','fly_market_pulse.py','fly_market_study.py','fly_trace.py','fly.py')
DECODER=('side','left_hz','right_hz','difference_hz','gate_spikes','spike_sha256')


def validate(payload):
    if not isinstance(payload,dict) or set(payload)!={'protocol','reference_json','market_plan'}:
        raise ValueError('Expected protocol, reference JSON, and sealed market plan')
    protocol=payload['protocol'];raw=payload['reference_json']
    if not isinstance(protocol,dict) or protocol.get('schema')!=1 or protocol.get('kind')!='post_hoc_market_memory_restoration' or protocol.get('arms')!=ARMS:
        raise ValueError('Expected the five registered restoration controls')
    if protocol.get('inference')!={'learning':False,'reinforcement':'none','extra_current':0,'steps':3}:
        raise ValueError('Restoration inference must remain frozen and unstimulated')
    if protocol.get('training_source')!='online_original' or protocol.get('test_source')!='trained_frozen':
        raise ValueError('Changed restoration reference arms')
    if not isinstance(raw,str) or len(raw.encode())>2_000_000 or hashlib.sha256(raw.encode()).hexdigest()!=protocol.get('reference_report_sha256'):
        raise ValueError('Reference report differs from registration')
    reference=json.loads(raw);envelope=payload['market_plan']
    if not isinstance(envelope,dict) or set(envelope)!={'plan','sha256'}:raise ValueError('Expected sealed market envelope')
    plan=validate_market(envelope['plan'])
    if signature(plan)!=envelope['sha256'] or reference.get('plan_sha256')!=envelope['sha256'] or protocol.get('market_plan_sha256')!=envelope['sha256']:
        raise ValueError('Market plan provenance mismatch')
    if plan['schema']!=4 or plan['phase_steps']!=3:raise ValueError('Requires three-step frozen-inference reference')
    pool=protocol.get('pool')
    if pool not in plan['cohort'] or pool not in reference.get('phase_diagnostics',{}):raise ValueError('Registered pool missing')
    rows=reference['phase_diagnostics'][pool]
    for arm,phase in (('online_original','training'),('online_original','test'),('pristine_frozen','test'),('trained_frozen','test')):
        decisions=rows[arm][phase]['decisions'];start=plan['start']+(1800 if phase=='test' else 0)
        if len(decisions)!=4 or [d['decision_ts'] for d in decisions]!=[start+i*300 for i in range(4)] or not decisions[-1]['terminal']:
            raise ValueError('Incomplete reference timeline')
        for i,d in enumerate(decisions[:3]):
            if d['terminal'] or not d['available'] or d['neural'] is None:raise ValueError('All three reference observations are required')
            e=d['neural']
            if e['stimulus'] not in ('none','reward','aversive'):raise ValueError('Invalid training stimulus')
            if phase=='test':
                original=rows['online_original']['test']['decisions'][i]
                if d['quote_ts']!=original['quote_ts'] or e['input_sha256']!=original['neural']['input_sha256']:
                    raise ValueError('Reference controls saw different inputs')
                if arm!='online_original' and (e['stimulus']!='none' or e['plasticity_enabled'] or e['weight_delta_l2']!=0):
                    raise ValueError('Frozen reference reported stimulation or updates')
    return protocol,reference,plan


def restore_posts(pristine, trained, post_ids, targets):
    """Return an isolated memory copy; restore every plastic input to target body IDs."""
    if len(set(targets))!=len(targets) or any(t not in set(post_ids) for t in targets):
        raise ValueError('Duplicate or absent postsynaptic target')
    mask=np.isin(post_ids,targets)
    if set(pristine)!=set(trained) or set(trained)!={'weights','u','w'}:raise ValueError('Incomplete learned memory')
    result={}
    for key,value in trained.items():
        if value.shape!=mask.shape or pristine[key].shape!=value.shape:raise ValueError('Memory dimensions differ')
        result[key]=value.copy();result[key][mask]=pristine[key][mask]
    return result,mask


def matches(event, expected):
    return all(event[key]==expected[key] for key in DECODER)


def run(payload,data,output):
    protocol,reference,plan=validate(payload)
    root=Path(output);root.mkdir(parents=True,exist_ok=False);atomic_json(root/'protocol.json',protocol)
    lab=TraceLab(data);b=lab.brain
    if b.build!=reference['native_build']:raise ValueError('Native build differs from the reference')
    sequences=input_sequences(plan,reference,protocol['pool']);rows=reference['phase_diagnostics'][protocol['pool']]
    deadline=time.monotonic()+420
    def check_time():
        if time.monotonic()>deadline:raise TimeoutError('Bounded restoration assay expired')
    b.reset();pristine=learned_state(b);np.savez_compressed(root/'pristine-memory.npz',**pristine)
    b.eta=.001;b.weights_frozen=False;lab.fly.controller.s=replace(lab.fly.controller.s,learning=True)
    training=[]
    for rgb,row in sequences['training']:
        check_time();event=lab.fly.controller.observe(rgb,row['neural']['stimulus'])
        if not matches(event,row['neural']):raise AssertionError('Reference training did not reproduce')
        training.append(event)
    trained=learned_state(b)
    if memory_signature(trained)!=rows['online_original']['training']['final_memory_sha256']:
        raise AssertionError('Reconstructed trained memory differs from reference')
    np.savez_compressed(root/'trained-memory.npz',**trained)
    posts=b.post[b.circuit['edges']];post_ids=np.array([str(b.ids[i]) for i in posts])
    for target in ('10704','11402'):
        indices=np.flatnonzero(b.ids==int(target))
        if len(indices)!=1 or str(lab.types[int(indices[0])])!='MBON11':raise ValueError('Expected annotated MBON11 target')
    reports={};summary={}
    for name,arm in ARMS.items():
        check_time();initial,mask=restore_posts(pristine,trained if arm['memory']=='trained' else pristine,post_ids,arm['restore_post_ids'])
        restore_learned(b,initial);b.eta=.001;b.weights_frozen=True
        lab.fly.controller.s=replace(lab.fly.controller.s,learning=False)
        if not same_memory(b,initial):raise AssertionError('Restoration did not preserve the requested memory')
        folder=root/name;folder.mkdir();np.savez_compressed(folder/'initial-memory.npz',**initial)
        started=time.monotonic();events=[]
        for i,(rgb,row) in enumerate(sequences['test']):
            check_time();event=lab.capture(rgb,'none',folder,i+1)
            event.update(phase='replay',phase_step=i+1,market_decision_ts=row['decision_ts'],input_preset='recorded_market',input_news='none')
            if event['input_sha256']!=row['neural']['input_sha256'] or not same_memory(b,initial):
                raise AssertionError('Frozen replay changed input or synaptic memory')
            events.append(event)
        if name in ('pristine_frozen','trained_frozen'):
            expected=rows[name]['test']['decisions'][:3]
            if any(not matches(e,r['neural']) for e,r in zip(events,expected)):raise AssertionError(f'Reference control did not reproduce: {name}')
        restoration={'post_ids':arm['restore_post_ids'],'edge_ids':[str(e) for e in b.circuit['edges'][mask]],
                     'edge_count':int(mask.sum()),'changed_weights_from_trained':int(np.count_nonzero(initial['weights']!=trained['weights'])) if arm['memory']=='trained' else None}
        report={'schema':1,'source':SOURCE,'config':{'preset':'recorded_market','view':'original','news':'none','eta':.001,'learning':False,'pulses':'none',**arm},
                'restoration':restoration,'graph':{'neurons':b.n,'edges':len(b.post),'plastic_edges':len(b.circuit['edges'])},
                'native_build':b.build,'upstream_commit':UPSTREAM_COMMIT,'seconds':time.monotonic()-started,'events':events,
                'initial_memory_sha256':memory_signature(initial),'final_memory_sha256':memory_signature(learned_state(b)),
                'training_memory_sha256':memory_signature(trained),'reference_report_sha256':protocol['reference_report_sha256'],
                'plan_sha256':protocol['market_plan_sha256'],'pool':protocol['pool'],'interpretation':protocol['interpretation']}
        atomic_json(folder/'report.json',report);atomic_json(folder/'view.json',lab.export_view(folder,report));reports[name]=report
        summary[name]={'actions':[e['side'] for e in events],'gate_spikes':[e['gate_spikes'] for e in events],
                       'difference_hz':[e['difference_hz'] for e in events],'restored_edges':int(mask.sum())}
        print(f"restoration_study arm={name} actions={summary[name]['actions']}",flush=True)
    result={'status':'market_restoration_study_completed','protocol':protocol,'training_events':training,'reports':reports,'summary':summary,
            'verification':{'same_native_build':True,'training_and_memory_reproduced':True,'both_reference_controls_reproduced':True,
                            'all_frozen_memory_preserved':True,'identical_test_images':True},
            'code_sha256':{name:digest(Path(__file__).with_name(name)) for name in FILES}}
    atomic_json(root/'summary.json',result);return result


def cloud_run(payload,output):
    import modal
    import uuid
    validate(payload);root=Path(output);root.mkdir(parents=True,exist_ok=True);receipt_path=root/'cloud-call.json'
    if not receipt_path.exists():
        receipt={'run_id':'assay-restore-'+uuid.uuid4().hex,'status':'submitting','payload_sha256':signature(payload)}
        atomic_json(receipt_path,receipt);atomic_json(root/'source-hashes.json',{name:digest(Path(__file__).with_name(name)) for name in FILES})
        call=modal.Function.from_name('fly-paper-lab','worker',environment_name='main').spawn(debug={'run_id':receipt['run_id'],'restoration_plan':payload})
        receipt.update(status='pending',call_id=call.object_id);atomic_json(receipt_path,receipt)
    receipt=json.loads(receipt_path.read_text())
    if receipt['payload_sha256']!=signature(payload):raise ValueError('Output directory belongs to another payload')
    if 'call_id' not in receipt:raise RuntimeError('Uncertain submission; inspect Modal before retrying')
    print(f"Observing saved call {receipt['call_id']}",flush=True)
    try:result=modal.FunctionCall.from_id(receipt['call_id']).get(timeout=50)
    except TimeoutError:
        print('Same call is pending. Repeat this observer command; it will not resubmit.',flush=True);return
    atomic_json(root/'cloud-result.json',result)
    if result.get('status')!='market_restoration_study_completed' or result.get('run_id')!=receipt['run_id']:raise RuntimeError('Unexpected result; no automatic resubmission')
    report=result['report']
    if report['protocol']!=payload['protocol'] or report['code_sha256']!=json.loads((root/'source-hashes.json').read_text()):raise ValueError('Executed protocol or source differs from submission')
    volume=modal.Volume.from_name('fly-paper-lab-state',environment_name='main')
    for arm in ARMS:
        folder=root/arm;folder.mkdir(exist_ok=True);remote=result['remote_path']+'/'+arm;view=folder/'view.json'
        if not view.exists():
            partial=view.with_suffix('.partial')
            with partial.open('wb') as f:
                for block in volume.read_file(remote.removeprefix('/state')+'/view.json'):f.write(block)
            partial.replace(view)
        atomic_json(folder/'report.json',report['reports'][arm]);atomic_json(folder/'remote.json',{'remote_path':remote,'call_id':receipt['call_id']})
    atomic_json(root/'summary.json',report);receipt.update(status='completed',budget=result['budget']);atomic_json(receipt_path,receipt)
    print(json.dumps({'summary':report['summary'],'verification':report['verification'],'budget':result['budget']},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    pack=sub.add_parser('pack')
    for field in ('protocol','reference','plan','out'):pack.add_argument('--'+field,type=Path,required=True)
    for command in ('run','cloud'):
        cmd=sub.add_parser(command);cmd.add_argument('--payload',type=Path,required=True);cmd.add_argument('--out',type=Path,required=True)
        if command=='run':cmd.add_argument('--fly-data',type=Path,required=True)
    a=p.parse_args()
    if a.command=='pack':
        payload={'protocol':json.loads(a.protocol.read_text()),'reference_json':a.reference.read_text(),'market_plan':json.loads(a.plan.read_text())}
        validate(payload);atomic_json(a.out,payload)
    else:
        payload=json.loads(a.payload.read_text())
        if a.command=='cloud':cloud_run(payload,a.out)
        else:run(payload,a.fly_data,a.out)


if __name__=='__main__':main()
