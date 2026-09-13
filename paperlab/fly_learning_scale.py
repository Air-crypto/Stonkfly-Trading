"""Bounded tenfold learning-rate comparison with both activity traces retained.

Capture layout derives from the preserved selective-trace runner. This module
has its own registered inputs, execution hashes, cloud claim and audit.
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .core import atomic_json, digest
from .fly import UPSTREAM_COMMIT
from .fly_selective_trace import (apply_boundary, compare_recording, require_completed_study,
    verify_reference, validate as validate_selective, source_hashes as selective_sources)
from .fly_market_activity import dynamic_state, array_hash, attach_boundary_state
from .fly_market_activity_audit import read_arrays
from .fly_market_memory_audit import read_memory
from .fly_market_pulse import input_sequences, SOURCE
from .fly_market_study import learned_state, restore_learned, memory_signature
from .fly_trace import TraceLab
from .fly_trace_memory import enrich_frame
ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SHA = '18a918669a6520b6e5ef2bbcabd4a162924394e58723f92645d098ad77eb9fb3'
NEW_FILES = ('paperlab/fly_learning_scale.py','paperlab/fly_learning_scale_audit.py',
    'paperlab/fly_learning_scale_credit.py','paperlab/fly_learning_scale_cloud.py','scale_cloud.py',
    'paperlab/fly_selective_memory.py','reports/fly-learning-scale-protocol-01.json',
    'reports/fly-selective-audit-01.json','reports/fly-selective-memory-01.json')


def source_hashes():
    return {**selective_sources(), **{n:digest(ROOT/n) for n in NEW_FILES}}


def validate(payload):
    if not isinstance(payload,dict) or set(payload)!={'protocol_json','selective_payload'}:
        raise ValueError('Expected the registered scale protocol and original selective payload')
    raw=payload['protocol_json']
    if not isinstance(raw,str) or hashlib.sha256(raw.encode()).hexdigest()!=PROTOCOL_SHA:
        raise ValueError('Learning-rate protocol differs from registration')
    p=json.loads(raw)
    prior_raw=json.dumps(payload['selective_payload'],indent=2)+'\n'
    if hashlib.sha256(prior_raw.encode()).hexdigest()!=p['selective_payload_sha256']:
        raise ValueError('Original selective payload differs')
    _,parent,market,plan=validate_selective(payload['selective_payload'])
    for file,key in (('fly-selective-audit-01.json','selective_audit_sha256'),
                     ('fly-selective-memory-01.json','memory_analysis_sha256')):
        if digest(ROOT/'reports'/file)!=p[key]:raise ValueError('Required completed evidence differs')
    previous=json.loads((ROOT/'reports/fly-selective-audit-01.json').read_text())
    memory=json.loads((ROOT/'reports/fly-selective-memory-01.json').read_text())
    if (previous['status']!='selective_trace_audited' or previous['verification']['observations']!=36
        or memory['status']!='selective_memory_components_reconstructed'
        or memory['audit_sha256']!=p['selective_audit_sha256']):
        raise ValueError('Previous selective assay and memory analysis must be complete')
    if list(p['arms'])!=p['execution_order'] or list(p['arms'])[:3]!=p['control_arms']:
        raise ValueError('All original controls must precede new conditions')
    return p,parent,market,plan


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
            raise TimeoutError('Learning-rate assay time limit exceeded')
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
    expected_controls = p['control_arms']
    for name, arm in p['arms'].items():
        check()
        if name not in expected_controls and controls != expected_controls:
            raise ValueError('All three controls must reproduce before the lower-rate intervention')
        restore_learned(b,memory); b.eta=arm['eta']; b.weights_frozen=not arm['learning']
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
            if name in expected_controls:
                compare_recording(folder/f'step-{i:02}.npz',reference_root/source/f'step-{i:02}.npz',
                    rates=True)
            events.append(event)
        if name in expected_controls: controls.append(name)
        report={'schema':1,'source':SOURCE,'config':{'preset':'recorded_market','view':'original','news':'none',
            'eta':arm['eta'],'memory':'trained','state_reset':arm['boundary'],**arm},
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
        print('learning_scale '+name+' '+json.dumps(summary[name]),flush=True)
    result={'status':'learning_scale_captured_pending_audit','protocol':p,'reports':reports,'summary':summary,
        'controls_verified':controls,'code_sha256':source_hashes(),'completed_study_sha256':release,
        'artifact_sha256':{name:digest(output/name) for name in ('initial-dynamics.npz','trained-memory.npz',
            'pristine-memory.npz','full-weight-reference.npz','neuron-ids.npz','circuit.npz')}}
    atomic_json(output/'summary.json',result);return result

