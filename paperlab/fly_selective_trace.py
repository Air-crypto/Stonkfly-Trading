"""Separate KC/DAN boundary interventions without changing the running study.

The CLI prepares and checks recorded arrays only. Native execution is an internal
entrypoint for a future budgeted worker integration, after study 11 is audited.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from .core import atomic_json, digest
from .fly_credit_reset import FILES, validate as validate_parent, compare_trace
from .fly_market_activity import SCALARS, array_hash, attach_boundary_state, dynamic_state
from .fly_market_activity_audit import DYNAMIC_FIELDS, read_arrays
from .fly_market_memory_audit import read_memory
from .fly_market_pulse import input_sequences, SOURCE
from .fly_market_study import learned_state, memory_signature, restore_learned, signature
from .fly_trace import TraceLab
from .fly_trace_memory import enrich_frame
from .fly import UPSTREAM_COMMIT

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SHA = '4497f1fc904f6fef548ed623e623ef5d57b6527283e4dc9a910572edc4bd9d6e'
TARGETS = {'carry': (), 'reset_rates': ('rate_kc', 'rate_dan'),
           'reset_kc': ('rate_kc',), 'reset_dan': ('rate_dan',)}
NEW_FILES = ('paperlab/fly_selective_trace.py', 'paperlab/fly_selective_trace_audit.py',
             'paperlab/fly_online_cloud.py', 'paperlab/fly_online_figure.py',
             'paperlab/fly_credit_divergence.py','paperlab/fly_view_projection.py',
             'paperlab/fly_study_evidence_bundle.py', 'paperlab/fly_selective_trace_cloud.py')


def source_hashes():
    return {name: digest(ROOT/name) for name in (*FILES, *NEW_FILES)}


def validate(payload):
    if not isinstance(payload, dict) or set(payload) != {'protocol_json', 'parent_payload', 'parent_audit_json'}:
        raise ValueError('Expected the selective protocol and original parent evidence')
    raw = payload['protocol_json']; raw_audit = payload['parent_audit_json']
    if not isinstance(raw, str) or hashlib.sha256(raw.encode()).hexdigest() != PROTOCOL_SHA:
        raise ValueError('Selective protocol differs from specification')
    p = json.loads(raw)
    if digest(ROOT/'reports/fly-first-drive-01.json') != p['first_drive_sha256']:
        raise ValueError('First-drive evidence differs from specification')
    if (not isinstance(raw_audit, str) or hashlib.sha256(raw_audit.encode()).hexdigest() != p['parent_audit_sha256']
            or signature(payload['parent_payload']) != p['parent_payload_signature']):
        raise ValueError('Parent evidence differs from specification')
    _, _, market, plan = validate_parent(payload['parent_payload'])
    audit = json.loads(raw_audit)
    if audit['status'] != 'credit_reset_audited' or audit['verification']['observations'] != 18:
        raise ValueError('Expected the completed six-condition parent audit')
    if list(p['arms']) != p['execution_order'] or p['reset_fields'] != {k: list(v) for k,v in TARGETS.items()}:
        raise ValueError('Intervention fields or order differ')
    return p, audit, market, plan


def verify_reference(audit, root):
    root = Path(root)
    for name, sha in audit['artifact_sha256'].items():
        if digest(root/name) != sha:
            raise ValueError('Audited parent artifact differs: '+name)


def apply_boundary(brain, mode, observation, path):
    """Reset only designated zero-initialized rate arrays; record every field."""
    if mode not in TARGETS or type(observation) is not int or not 1 <= observation <= 3:
        raise ValueError('Invalid selective boundary mode or observation')
    path = Path(path)
    if path.exists():
        raise ValueError('Refuse to overwrite a boundary')
    before = dynamic_state(brain); memory = learned_state(brain)
    clock = {k: getattr(brain, k) for k in SCALARS}; weights = array_hash(brain.weight)
    if set(before) != DYNAMIC_FIELDS:
        raise ValueError('Expected every native dynamic field')
    for key, value in before.items():
        initial = brain.initial[key]
        if value.shape != initial.shape or value.dtype != initial.dtype or not np.isfinite(value).all():
            raise ValueError('Invalid native boundary array: '+key)
        if observation == 1 and not np.array_equal(value, initial):
            raise ValueError('First observation must start with fresh dynamics')
    for key in ('rate_kc', 'rate_dan'):
        if np.any(brain.initial[key]):
            raise ValueError('Initial learning traces must be zero')
    fields = () if observation == 1 else TARGETS[mode]
    for key in fields:
        getattr(brain, key)[:] = brain.initial[key]
    after = dynamic_state(brain); memory_after = learned_state(brain)
    for key in before:
        if not np.array_equal(after[key], brain.initial[key] if key in fields else before[key]):
            raise ValueError('Boundary changed an undeclared field: '+key)
    if (any(not np.array_equal(memory[k], memory_after[k]) for k in memory)
            or array_hash(brain.weight) != weights or {k:getattr(brain,k) for k in SCALARS} != clock):
        raise ValueError('Boundary changed weights, memory or clock')
    arrays = {prefix+k:v for prefix, state in (('before__',before), ('after__',after),
        ('memory_before__',memory), ('memory_after__',memory_after)) for k,v in state.items()}
    np.savez_compressed(path, **arrays)
    return {'mode':'initial' if observation == 1 else mode, 'requested_mode':mode,
        'observation':observation, 'target_fields':list(fields), 'target_ids':[], 'target_indices':[],
        'before_clock':clock, 'after_clock':clock.copy(), 'all_weight_sha256':weights,
        'memory_sha256':memory_signature(memory), 'artifact_sha256':digest(path),
        'changed_fields':[k for k in before if not np.array_equal(before[k],after[k])],
        'before_sha256':{k:array_hash(v) for k,v in before.items()},
        'after_sha256':{k:array_hash(v) for k,v in after.items()}}


def compare_recording(path, reference, *, rates=True):
    compare_trace(path, reference, rates=rates)
    # The original checker verifies counts and memory; also retain exact sampled
    # voltage agreement so hidden state divergence cannot pass as reproduction.
    with np.load(path, allow_pickle=False) as a, np.load(reference, allow_pickle=False) as b:
        if a['voltage'].dtype != b['voltage'].dtype or not np.array_equal(a['voltage'], b['voltage']):
            raise ValueError('Full reference control does not reproduce: voltage')


def require_completed_study(root):
    from .fly_online_figure import evidence
    report, _, _, fixture = evidence(root)
    registration = json.loads((ROOT/'reports/fly-market-study-11-preregistration.json').read_text())
    if fixture or report['registration'] != registration or time.time() < registration['end']:
        raise ValueError('Native assay requires completion and audits of the actual study 11')
    return digest(Path(root)/'report.json')


def run(payload, reference_root, data, output, *, completed_study):
    """Bounded full-network capture; no submission, account or policy promotion."""
    p, parent, market, plan = validate(payload)
    output = Path(output); reference_root = Path(reference_root)
    if output.exists():
        raise ValueError('Refuse to overwrite or resume a neural assay')
    release = require_completed_study(completed_study)
    verify_reference(parent, reference_root)
    deadline = time.monotonic()+p['limits']['inner_timeout_seconds']
    def check():
        if time.monotonic() > deadline:
            raise TimeoutError('Selective trace assay time limit exceeded')
    lab = TraceLab(data); b = lab.brain
    if b.build != market['native_build'] or (b.n,len(b.post),len(b.circuit['edges'])) != (166700,25582938,7835):
        raise ValueError('Full native graph or build differs from reference')
    memory = read_memory(reference_root/'trained-memory.npz')
    initial = dynamic_state(b); baseline = learned_state(b)
    saved_initial = read_arrays(reference_root/'initial-dynamics.npz')
    if set(initial) != set(saved_initial) or any(not np.array_equal(v,saved_initial[k]) for k,v in initial.items()):
        raise ValueError('Fresh native state differs from original controls')
    if not np.array_equal((b.baseline_plastic*(1+memory['w'])).astype(b.weight.dtype),memory['weights']):
        raise ValueError('Imported memory differs from the full native baseline')
    reference = json.loads((reference_root/'trained_online_recorded_carry/view.json').read_text())['report']
    seq = input_sequences(plan,market,reference['pool'])['test']
    output.mkdir(parents=True); atomic_json(output/'protocol.json',p)
    np.savez_compressed(output/'initial-dynamics.npz',**initial)
    np.savez_compressed(output/'trained-memory.npz',**memory)
    np.savez_compressed(output/'pristine-memory.npz',**baseline)
    np.savez_compressed(output/'full-weight-reference.npz',weight=b.weight)
    np.savez_compressed(output/'neuron-ids.npz',neuron_ids=b.ids)
    np.savez_compressed(output/'circuit.npz',edges=b.circuit['edges'],pre=b.circuit['pre'],
        post=b.post[b.circuit['edges']],dan=b.circuit['dan'],gain=b.circuit['gain'],baseline=b.baseline_plastic)
    reports = {}; controls = []; summary = {}
    expected_controls = list(parent['protocol']['arms'])
    for name, arm in p['arms'].items():
        check()
        if name not in expected_controls and controls != expected_controls:
            raise ValueError('All six controls must reproduce before selective intervention')
        restore_learned(b,memory); b.eta=.001; b.weights_frozen=not arm['learning']
        lab.fly.controller.s=replace(lab.fly.controller.s,learning=arm['learning'])
        folder=output/name; folder.mkdir(); np.savez_compressed(folder/'initial-memory.npz',**memory)
        events=[]; started=time.monotonic()
        for i,(rgb,row) in enumerate(seq,1):
            check(); boundary=apply_boundary(b,arm['boundary'],i,folder/f'boundary-{i:02}.npz')
            pulse=row['neural']['stimulus'] if arm['pulses']=='recorded' else 'none'
            event=lab.capture(rgb,pulse,folder,i)
            if event['input_sha256'] != row['neural']['input_sha256']:
                raise ValueError('Sealed image did not reproduce')
            event.update(phase='replay',phase_step=i,market_decision_ts=row['decision_ts'],
                input_preset='recorded_market',input_news='none',activity_boundary=boundary,
                all_weight_sha256=array_hash(b.weight))
            if i<3:
                path=folder/f'end-{i:02}.npz';np.savez_compressed(path,**dynamic_state(b))
                event['end_state_sha256']=digest(path)
            source=name if name in expected_controls else arm['reference_arm']+'_carry'
            if name in expected_controls or i==1 or not arm['learning']:
                compare_recording(folder/f'step-{i:02}.npz',reference_root/source/f'step-{i:02}.npz',
                    rates=name in expected_controls or i==1)
            events.append(event)
        if name in expected_controls: controls.append(name)
        report={'schema':1,'source':SOURCE,'config':{'preset':'recorded_market','view':'original','news':'none',
            'eta':.001,'memory':'trained','state_reset':arm['boundary'],**arm},
            'graph':{'neurons':b.n,'edges':len(b.post),'plastic_edges':len(b.circuit['edges'])},
            'native_build':b.build,'upstream_commit':UPSTREAM_COMMIT,'seconds':time.monotonic()-started,
            'events':events,'initial_memory_sha256':memory_signature(memory),
            'final_memory_sha256':memory_signature(learned_state(b)),
            'reference_report_sha256':p['parent_audit_sha256'],'plan_sha256':reference['plan_sha256'],
            'pool':reference['pool'],'interpretation':p['interpretation']}
        atomic_json(folder/'report.json',report);view=lab.export_view(folder,report)
        attach_boundary_state(view,folder)
        for i,frame in enumerate(view['frames'],1):
            frame.update(enrich_frame(view,frame,folder/f'step-{i:02}.npz'))
        atomic_json(folder/'view.json',view);reports[name]=report
        summary[name]={'actions':[e['side'] for e in events],'gate_spikes':[e['gate_spikes'] for e in events],
            'weight_update_l2':[e['diagnostics']['weight_delta_l2'] for e in events]}
        atomic_json(output/'progress.json',{'completed':list(reports),'controls_verified':controls})
        print('selective_trace '+name+' '+json.dumps(summary[name]),flush=True)
    result={'status':'selective_trace_captured_pending_audit','protocol':p,'reports':reports,'summary':summary,
        'controls_verified':controls,'code_sha256':source_hashes(),'completed_study_sha256':release,
        'artifact_sha256':{name:digest(output/name) for name in ('initial-dynamics.npz','trained-memory.npz',
            'pristine-memory.npz','full-weight-reference.npz','neuron-ids.npz','circuit.npz')}}
    atomic_json(output/'summary.json',result);return result


def preflight(payload, reference_root, output):
    """Apply boundary operations to copies of saved arrays, with no propagation."""
    from .fly_selective_trace_audit import audit_boundary
    p,parent,_,_=validate(payload);root=Path(reference_root);output=Path(output)
    if output.exists():raise ValueError('Preserve earlier preflight evidence')
    verify_reference(parent,root)
    initial=read_arrays(root/'initial-dynamics.npz');circuit=read_arrays(root/'circuit.npz')
    full_weight=read_arrays(root/'full-weight-reference.npz')['weight']
    output.mkdir(parents=True);checks=[]
    for name,arm in parent['protocol']['arms'].items():
        if arm['boundary']!='carry':continue
        view=json.loads((root/name/'view.json').read_text())
        for i in (2,3):
            original=read_arrays(root/name/f'boundary-{i:02}.npz')
            memory={k:original['memory_before__'+k] for k in ('weights','u','w')}
            full_weight[circuit['edges']]=memory['weights']
            for mode in TARGETS:
                b=SimpleNamespace(initial=initial,weight=full_weight,circuit=circuit,
                    memory_u=memory['u'].copy(),memory_w=memory['w'].copy(),
                    **view['frames'][i-1]['event']['activity_boundary']['before_clock'],
                    **{k:original['before__'+k].copy() for k in initial})
                path=output/f'{name}-{i}-{mode}.npz';metadata=apply_boundary(b,mode,i,path)
                checked=audit_boundary(path,metadata,initial,memory,mode,i,full_weight)
                checks.append({'recorded_parent':name,'observation':i,'mode':mode,**checked})
    result={'status':'selective_trace_preflight_passed','protocol_sha256':PROTOCOL_SHA,
        'parent_audit_sha256':p['parent_audit_sha256'],'parent_artifacts_verified':len(parent['artifact_sha256']),
        'boundary_checks':checks,'dynamic_fields':len(initial),'full_weights':len(full_weight),
        'source_sha256':source_hashes(),'neural_observations':0,'cloud_submissions':0,
        'interpretation':'Boundary transformations of copies of recorded native arrays only. These are not selective neural trajectories, decisions, or returns. Native execution and its independent audit remain pending.'}
    atomic_json(output/'report.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('parent-payload','parent-audit','reference-recordings','out'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    payload={'protocol_json':(ROOT/'reports/fly-selective-trace-protocol-01.json').read_text(),
        'parent_payload':json.loads(args.parent_payload.read_text()),'parent_audit_json':args.parent_audit.read_text()}
    report=preflight(payload,args.reference_recordings,args.out)
    atomic_json(args.out/'payload.json',payload)
    print(json.dumps({k:report[k] for k in ('status','parent_artifacts_verified','dynamic_fields',
        'full_weights','neural_observations','cloud_submissions')},indent=2))


if __name__=='__main__':main()
