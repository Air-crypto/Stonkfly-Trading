"""Audit real saved activity-boundary arrays and frozen synaptic memory offline."""
import argparse
import json
from pathlib import Path

import numpy as np

from .core import atomic_json, digest
from .fly_market_activity import ALL_RESET_FIELDS, REFINEMENT_FIELDS, GATE_IDS, SCALARS, array_hash, catalog, validate
from .fly_market_memory_audit import read_memory
from .fly_market_restoration import matches
from .fly_market_study import memory_signature

DYNAMIC_FIELDS={'v','g','refractory','drive','previous_drive','queue','queue_count','counts','luminance',
    'active','active_flag','nactive','last','eligibility','eligibility_last','modulation','modulation_last',
    'adaptation','rate_kc','rate_dan','r8_light'}


def read_arrays(path):
    with np.load(path,allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}


def audit_boundary(path, metadata, initial, memory, mode, observation, neuron_ids=None):
    if digest(path)!=metadata['artifact_sha256']:raise ValueError('Boundary artifact hash differs')
    data=read_arrays(path);fields=set(initial);expected={p+k for p in ('before__','after__') for k in fields}|{p+k for p in ('memory_before__','memory_after__') for k in memory}
    if set(data)!=expected:raise ValueError('Boundary arrays are incomplete')
    effective='initial' if observation==1 else mode
    targets=fields if effective=='full' else set(ALL_RESET_FIELDS.get(effective,()))
    indices=[]
    if mode=='gate_voltage_conductance':
        if neuron_ids is None:raise ValueError('Gate reset requires the actual neuron identity map')
        ids=[str(i) for i in neuron_ids]
        if len(set(ids))!=len(ids) or any(ids.count(i)!=1 for i in GATE_IDS):raise ValueError('Invalid gate identity map')
        indices=[ids.index(i) for i in GATE_IDS]
    if metadata.get('target_ids',[])!=(GATE_IDS if indices else []) or metadata.get('target_indices',[])!=indices:raise ValueError('Boundary reset targeted different neurons')
    if mode in REFINEMENT_FIELDS and not np.all(data['before__last']==metadata['before_clock']['cursor']-1):raise ValueError('Boundary state was not materialized before reset')
    if metadata['mode']!=effective or metadata['requested_mode']!=mode or metadata['observation']!=observation or set(metadata['target_fields'])!=targets or len(metadata['target_fields'])!=len(targets):raise ValueError('Boundary does not match its declared intervention')
    changed=[]
    for k in fields:
        before,after=data['before__'+k],data['after__'+k]
        if before.shape!=initial[k].shape or after.shape!=before.shape or before.dtype!=initial[k].dtype or after.dtype!=before.dtype or not np.isfinite(before).all() or not np.isfinite(after).all():raise ValueError('Invalid boundary state arrays')
        if array_hash(before)!=metadata['before_sha256'][k] or array_hash(after)!=metadata['after_sha256'][k]:raise ValueError('Boundary state hash differs')
        if observation==1 and not np.array_equal(before,initial[k]):raise ValueError('First observation did not start fresh')
        desired=before.copy()
        if k in targets:
            if indices:
                if desired.shape!=(len(neuron_ids),):raise ValueError('Target field does not map one-to-one to neurons')
                desired[indices]=initial[k][indices]
            else:desired=initial[k]
        if not np.array_equal(after,desired):raise ValueError('Target or untouched state changed incorrectly: '+k)
        if not np.array_equal(before,after):changed.append(k)
    if set(changed)!=set(metadata['changed_fields']) or len(changed)!=len(metadata['changed_fields']):raise ValueError('Reported changed fields differ')
    for k,value in memory.items():
        if not np.array_equal(data['memory_before__'+k],value) or not np.array_equal(data['memory_after__'+k],value):raise ValueError('Boundary altered frozen memory')
    if metadata['memory_sha256']!=memory_signature(memory):raise ValueError('Boundary memory identity differs')
    before_clock,after_clock=metadata['before_clock'],metadata['after_clock']
    if set(before_clock)!=set(SCALARS) or after_clock!=({k:0 for k in SCALARS} if effective=='full' else before_clock):raise ValueError('Boundary changed clock outside its scope')
    return {'mode':effective,'changed_fields':sorted(changed),'reset_fields':sorted(targets),
        'target_ids':GATE_IDS if indices else [],'target_indices':indices,
        'preserved_fields':sorted(fields-targets),'artifact_sha256':digest(path),'before_count_sha256':array_hash(data['before__counts'])}


def audit_view_boundaries(view, folder, neuron_ids):
    indices=np.asarray([n['index'] for n in view['nodes']])
    if indices.dtype.kind not in 'iu' or np.any(indices<0) or np.any(indices>=len(neuron_ids)) or len(set(indices.tolist()))!=len(indices):raise ValueError('Invalid displayed neuron indices')
    if [str(neuron_ids[i]) for i in indices]!=[str(n['id']) for n in view['nodes']]:raise ValueError('Displayed neuron identities differ from the full graph')
    for i,f in enumerate(view['frames'],1):
        with np.load(Path(folder)/f'boundary-{i:02}.npz',allow_pickle=False) as a:
            state=f.get('activity_state',{})
            if set(state)!={p+'_'+k for p in ('before','after') for k in ('v','g')}:raise ValueError('Missing boundary state in view')
            for prefix in ('before','after'):
                for field in ('v','g'):
                    if not np.array_equal(state[prefix+'_'+field],a[prefix+'__'+field][indices]):raise ValueError('Displayed boundary values differ from saved arrays')


def audit(study,payload,artifacts):
    p,ref,plan=validate(payload);arms=catalog(p);root=Path(artifacts);parent=json.loads(payload['activity_reference_json']) if p['schema']==2 else None
    if study['protocol']!=p or set(study['reports'])!=set(arms):raise ValueError('Activity study controls differ')
    rows=ref['phase_diagnostics'][p['pool']]
    if len(study['training_events'])!=3 or any(not matches(e,d['neural']) for e,d in zip(study['training_events'],rows['trained_frozen']['training']['decisions'])):raise ValueError('Training reference differs')
    memories={m:read_memory(root/(m+'-memory.npz')) for m in ('pristine','trained')}
    for m,mem in memories.items():
        expected=rows[m+'_frozen']['test']['initial_memory_sha256']
        if memory_signature(mem)!=expected:raise ValueError('Saved reference memory differs')
    initial=read_arrays(root/'initial-dynamics.npz')
    if set(initial)!=DYNAMIC_FIELDS or digest(root/'initial-dynamics.npz')!=study['initial_dynamics_sha256']:raise ValueError('Initial native dynamics differ')
    neuron_ids=None
    if p['schema']==2:
        ids_file=root/'neuron-ids.npz'
        if digest(ids_file)!=study['neuron_ids_sha256']:raise ValueError('Neuron identity map hash differs')
        saved_ids=read_arrays(ids_file)
        if set(saved_ids)!={'neuron_ids'}:raise ValueError('Neuron identity checkpoint differs')
        neuron_ids=saved_ids['neuron_ids']
        if neuron_ids.ndim!=1 or len(neuron_ids)!=len(initial['v']) or len(np.unique(neuron_ids))!=len(neuron_ids):raise ValueError('Neuron identity map does not match dynamic state')
    verified={};weights_by_memory={}
    for name,arm in arms.items():
        report=study['reports'][name];events=report['events'];memory=memories[arm['memory']]
        if report['native_build']!=ref['native_build'] or report['plan_sha256']!=p['market_plan_sha256'] or report['reference_report_sha256']!=p['reference_report_sha256']:raise ValueError('Replay provenance differs')
        if report['config']!={'preset':'recorded_market','view':'original','news':'none','eta':.001,'learning':False,'pulses':'none',**arm}:raise ValueError('Replay settings differ')
        if report['initial_memory_sha256']!=memory_signature(memory) or report['final_memory_sha256']!=memory_signature(memory) or len(events)!=3:raise ValueError('Incomplete or unfrozen replay')
        saved=read_memory(root/name/'initial-memory.npz')
        if any(not np.array_equal(saved[k],v) for k,v in memory.items()):raise ValueError('Initial arm checkpoint differs')
        if p['schema']==2:
            view=json.loads((root/name/'view.json').read_text())
            if view['report']!=report or [f['event'] for f in view['frames']]!=events:raise ValueError('Boundary view provenance differs')
            audit_view_boundaries(view,root/name,neuron_ids)
        boundaries=[];expected_clock={k:0 for k in SCALARS}
        for i,(event,row) in enumerate(zip(events,rows['trained_frozen']['test']['decisions']),1):
            if event['market_decision_ts']!=row['decision_ts'] or event['input_sha256']!=row['neural']['input_sha256'] or event['stimulus']!='none' or event['diagnostics']['plasticity_enabled'] or event['diagnostics']['weight_delta_l2']!=0:raise ValueError('Input, learning or stimulation differs')
            boundary=event['activity_boundary'];result=audit_boundary(root/name/f'boundary-{i:02}.npz',boundary,initial,memory,arm['state_reset'],i,neuron_ids)
            if boundary['before_clock']!=expected_clock:raise ValueError('Boundary is not linked to the preceding observation')
            if i>1 and result['before_count_sha256']!=events[i-2]['spike_sha256']:raise ValueError('Boundary spike counts differ from preceding observation')
            if event['brain_ms']!=boundary['after_clock']['sim_ms']+500:raise ValueError('Observation clock does not follow boundary')
            expected_clock={'sim_ms':event['brain_ms'],'cursor':boundary['after_clock']['cursor']+5000,
                'total_spikes':boundary['after_clock']['total_spikes']+event['total_spikes']}
            previous=weights_by_memory.setdefault(arm['memory'],boundary['all_weight_sha256'])
            if previous!=boundary['all_weight_sha256']:raise ValueError('Frozen whole-graph weights differ between boundaries')
            boundaries.append(result)
        if arm['state_reset']=='carry' and any(not matches(e,d['neural']) for e,d in zip(events,rows[arm['memory']+'_frozen']['test']['decisions'])):raise ValueError('Carry control did not reproduce')
        if parent is not None and name in ('pristine_carry','trained_carry','pristine_full','trained_full'):
            old=parent['reports'][name]
            if report['initial_memory_sha256']!=old['initial_memory_sha256'] or any(not matches(a,b) for a,b in zip(events,old['events'])):raise ValueError('Earlier activity control did not reproduce')
        control=study['reports'][arm['memory']+'_carry']['events']
        if not matches(events[0],control[0]):raise ValueError('First observation changed before intervention')
        verified[name]={'boundaries':boundaries,'actions':[e['side'] for e in events],'gate_spikes':[e['gate_spikes'] for e in events],
            'difference_hz':[e['difference_hz'] for e in events],
            'action_matches_carry':[a['side']==b['side'] for a,b in zip(events,control)],
            'whole_count_matches_carry':[a['spike_sha256']==b['spike_sha256'] for a,b in zip(events,control)]}
        if study['summary'][name]!={k:verified[name][k] for k in ('actions','gate_spikes','difference_hz')}:raise ValueError('Summary differs from recorded events')
    return {'protocol':p,'reference_controls':['pristine_carry','trained_carry','pristine_full','trained_full'] if parent is not None else ['pristine_carry','trained_carry'],'initial_dynamics_sha256':digest(root/'initial-dynamics.npz'),'memory_sha256':{m:memory_signature(v) for m,v in memories.items()},'arms':verified,
        'interpretation':'Actual saved before/after arrays verify every targeted and untouched dynamic field and synaptic memory. Both carry controls and all first observations reproduce. Full resets change multiple dynamic fields and the neural clock; partial resets preserve every other recorded dynamic field. No trading account or returns are computed.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study','payload','artifacts','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();atomic_json(a.out,audit(json.loads(a.study.read_text()),json.loads(a.payload.read_text()),a.artifacts))


if __name__=='__main__':main()
