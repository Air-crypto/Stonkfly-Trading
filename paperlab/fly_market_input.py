"""Registered market evaluation of native synaptic-input resets; paper only."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3

import numpy as np

from .core import atomic_json, digest

INPUT_ARMS = {'pristine_frozen': {'eta': 0.001, 'train': False, 'online': False, 'view': 'original', 'activity_reset': 'carry'}, 'trained_frozen': {'eta': 0.001, 'train': True, 'online': False, 'view': 'original', 'activity_reset': 'carry'}, 'pristine_input_reset': {'eta': 0.001, 'train': False, 'online': False, 'view': 'original', 'activity_reset': 'conductance'}, 'trained_input_reset': {'eta': 0.001, 'train': True, 'online': False, 'view': 'original', 'activity_reset': 'conductance'}}
INPUT_TIMING = 'Training carries all activity normally and shares identical training recipes. Development and test start separately from fresh dynamics and the saved training memory. Immediately before each eligible observation after the first, restore only native synaptic-input state g to its initial values in the reset arms. Carry controls preserve all activity. Missing decision slots do not advance or reset the brain. All inference weights/u/w remain frozen; no reinforcement or extra current. Preserve every other field and the neural clock at each partial reset.'
INPUT_PROMOTION = 'Only trained_input_reset can qualify. Its development total equity must strictly exceed cash and each of pristine_frozen, trained_frozen, and pristine_input_reset. Fix selection before test simulation. Report all test arms without reselection or automatic deployment.'

BASE_FILES=('fly_market_study.py','fly_trace.py','fly.py','fly_visual.py','fly_market_restoration.py')

def source_files(schema):
    return BASE_FILES + ('fly_market_input.py',) + (('fly_market_activity.py','fly_market_activity_audit.py','fly_market_pulse.py') if schema==6 else ())


def validate_input(plan):
    if plan.get('phase_steps')!=3 or plan.get('activity_reset_timing')!=INPUT_TIMING or plan.get('promotion_rule')!=INPUT_PROMOTION:
        raise ValueError('Input reset timing, phase length or promotion rule differs')
    start=plan['start'];previous=plan.get('previous_test_end',0)
    if not 0<plan.get('recorded_at',0)<start or not 0<previous<=start or previous%300:
        raise ValueError('Register the input comparison before training, after the parent test')
    for key in ('mechanism_report_sha256','parent_report_sha256'):
        value=plan.get(key)
        if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Missing pinned input-reset provenance')


def development_choice(plan,equity):
    if plan['schema']==6:
        return 'trained_input_reset' if equity['trained_input_reset']>max(1000,*(equity[k] for k in INPUT_ARMS if k!='trained_input_reset'))+1e-9 else None
    candidate=max(plan['arms'],key=lambda name:(equity[name],name))
    return candidate if equity[candidate]>max(1000,equity['pristine_frozen'])+1e-9 else None


def seal_registered(archive,previous,registration,output):
    from .fly_market_study import validate,signature
    from .multi import read_archive
    from .fly_market_cloud import validate_registration
    previous=json.loads(Path(previous).read_text());registration=json.loads(Path(registration).read_text())
    old=validate(previous['plan'])
    if previous['sha256']!=signature(old) or registration['parent_plan_sha256']!=previous['sha256'] or registration['previous_test_end']!=old['start']+old['phase_steps']*900:
        raise ValueError('Registered parent interval differs')
    if registration['cohort']!=old['cohort'] or registration['costs']!=old['costs']:
        raise ValueError('Registered cohort or costs differ from parent')
    output=Path(output)
    if output.exists():raise ValueError('Refuse to overwrite a sealed input-reset plan')
    archive=Path(archive);db=sqlite3.connect(f'file:{archive.resolve()}?mode=ro',uri=True)
    try:last=db.execute("SELECT max(json_extract(payload,'$.observed')) FROM observations").fetchone()[0]
    finally:db.close()
    end=registration['end']
    if last is None or last<end:raise ValueError('Snapshot has not reached the registered endpoint')
    # Read only observations through the declared endpoint, even if the snapshot is newer.
    _,series=read_archive(archive,end)
    if any(k not in series for k in old['cohort']):raise ValueError('An original cohort pool is missing; never replace it with a survivor')
    # Keep enough context for the unchanged 100-observation visual window and admission.
    # Selection is chronological and does not depend on returns or later availability.
    selected={}
    for key in old['cohort']:
        ticks=series[key];past=[t for t in ticks if t.ts<=registration['start']]
        future=[t for t in ticks if registration['start']<t.ts<=end]
        selected[key]=[asdict(t) for t in past[-(512-len(future)):]+future]
    plan={**old,**registration,'snapshot_sha256':digest(archive),'snapshot_end':last,'series':selected}
    plan.pop('restoration_timing',None);plan.pop('visual_encoding',None);plan.pop('end',None)
    envelope={'plan':plan,'sha256':signature(plan)};validate_registration(envelope,registration)
    atomic_json(output,envelope);return envelope


def audit_market_boundaries(envelope,report,artifacts):
    """Verify actual reset arrays, fresh phase starts, observed slots and viewer values."""
    import hashlib
    import sys
    from .fly_market_activity_audit import DYNAMIC_FIELDS, read_arrays, audit_boundary, audit_view_boundaries
    from .fly_market_memory_audit import read_memory
    from .fly_market_study import validate, signature, memory_signature, quote_at
    from .fly import frame
    from .news import News
    from .core import Tick
    plan=validate(envelope['plan']);root=Path(artifacts)
    if plan['schema']!=6 or signature(plan)!=envelope['sha256'] or report['plan_sha256']!=envelope['sha256'] or report['activity_reset_timing']!=INPUT_TIMING:
        raise ValueError('Boundary audit plan or timing differs')
    initial=read_arrays(root/'initial-dynamics.npz');ids=read_arrays(root/'neuron-ids.npz')
    if set(initial)!=DYNAMIC_FIELDS or set(ids)!={'neuron_ids'} or digest(root/'initial-dynamics.npz')!=report['initial_dynamics_sha256'] or digest(root/'neuron-ids.npz')!=report['neuron_ids_sha256']:
        raise ValueError('Initial state or neuron identity file differs')
    identities=ids['neuron_ids']
    if identities.ndim!=1 or identities.shape!=initial['v'].shape or len(np.unique(identities))!=len(identities):raise ValueError('Invalid neuron identity map')
    vendor=str(Path(__file__).resolve().parents[1]/'vendor/stonkfly')
    if vendor not in sys.path:sys.path.insert(0,vendor)
    news=News(enabled=False);verified={};weights={}
    try:
        for i,key in enumerate(plan['cohort']):
            ticks=[Tick(**t) for t in plan['series'][key]];verified[key]={}
            for name,arm in plan['arms'].items():
                saved=read_memory(root/f'pool{i}-{name}-training-memory.npz');memory=memory_signature(saved);verified[key][name]={}
                for j,phase in enumerate(('development','test'),1):
                    rows=report['phase_diagnostics'][key][name][phase]['decisions'];folder=root/f'boundaries/pool{i}-{name}-{phase}'
                    events=[];clock={'cursor':0,'sim_ms':0.,'total_spikes':0};previous=None;last_quote=0;boundaries=[]
                    for slot,d in enumerate(rows):
                        if d['decision_ts']!=plan['start']+(j*3+slot)*300 or d['terminal']!=(slot==3):raise ValueError('Decision or terminal timeline differs')
                        index,quote=quote_at(ticks,d['decision_ts']);e=d['neural']
                        if d['quote_ts']!=quote.ts or d['available']!=quote.available:raise ValueError('Recorded quote eligibility differs from the sealed input')
                        expected=slot<3 and quote.available and quote.ts>last_quote
                        if (e is not None)!=expected:raise ValueError('Brain advanced in an ineligible or missing slot')
                        if e is None:continue
                        if e['input_sha256']!=hashlib.sha256(frame(ticks,index,news).tobytes()).hexdigest():raise ValueError('Recorded sensory input differs from the sealed market image')
                        last_quote=quote.ts;events.append(e)
                        if e['stimulus']!='none' or e['plasticity_enabled'] or e['weight_delta_l2']!=0 or e['equity_reward_usd']!=0:raise ValueError('Activity inference changed learning or reinforcement')
                        boundary=e['activity_boundary']
                        if boundary['before_clock']!=clock:raise ValueError('Phase carried or reset an undeclared neural clock')
                        result=audit_boundary(folder/f'boundary-{len(events):02}.npz',boundary,initial,saved,arm['activity_reset'],len(events),identities)
                        if previous is not None and result['before_count_sha256']!=previous['spike_sha256']:raise ValueError('Boundary counts do not match the preceding observation')
                        if e['brain_ms']!=clock['sim_ms']+500:raise ValueError('Partial reset changed observation time')
                        clock={'cursor':clock['cursor']+5000,'sim_ms':e['brain_ms'],'total_spikes':clock['total_spikes']+e['total_spikes']};previous=e
                        old=weights.setdefault((key,arm['train']),boundary['all_weight_sha256'])
                        if old!=boundary['all_weight_sha256'] or boundary['memory_sha256']!=memory:raise ValueError('Frozen memory differs between inference conditions')
                        boundaries.append(result)
                    if len(rows)!=4:raise ValueError('Incomplete phase timeline')
                    if phase=='test' and events:
                        view=json.loads((root/f'pool{i}-{name}/view.json').read_text())
                        if view['report']['events']!=events or [f['event'] for f in view['frames']]!=events or view['report']['config']['activity_reset']!=arm['activity_reset']:raise ValueError('Viewer events or reset mode differ')
                        audit_view_boundaries(view,folder,identities)
                    verified[key][name][phase]={'observations':len(events),'boundaries':boundaries,'final_neural_ms':clock['sim_ms']}
    finally:news.db.close()
    return {'plan_sha256':envelope['sha256'],'pools':verified,'initial_dynamics_sha256':report['initial_dynamics_sha256'],'neuron_ids_sha256':report['neuron_ids_sha256'],
        'interpretation':'Every observed inference boundary uses actual before/after arrays. Verify only the declared input state resets, all other fields and memory remain fixed at the boundary, and phases start fresh. Missing slots do not trigger observations or resets. Sensory input hashes reproduce the sealed prices. This audit does not establish profitability.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('archive','previous','registration','out'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();r=seal_registered(a.archive,a.previous,a.registration,a.out);print(r['sha256'])


if __name__=='__main__':main()
