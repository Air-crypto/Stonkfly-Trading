"""Registered, bounded intervention on learning traces in the full fly graph."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .core import atomic_json, digest
from .fly import UPSTREAM_COMMIT
from .fly_market_activity import apply_boundary, attach_boundary_state, dynamic_state, array_hash
from .fly_market_memory_audit import read_memory
from .fly_market_pulse import input_sequences, validate as validate_pulse, SOURCE
from .fly_market_study import learned_state, memory_signature, restore_learned
from .fly_pulse_audit import audit as audit_pulse
from .fly_trace import TraceLab
from .fly_trace_memory import enrich_frame

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SHA = 'f16fb1857e2b5099105077e02ea1cd039903d58e55a08d60593370305bcbd33f'
FILES = tuple('paperlab/'+n+'.py' for n in ('fly_credit_reset','fly_credit_reset_audit',
    'fly_credit_reset_cloud','fly_credit_audit','fly_market_activity','fly_market_activity_audit',
    'fly_market_memory_audit','fly_market_pulse','fly_market_study','fly_market_restoration',
    'fly_pulse_audit','fly_trace','fly_trace_memory','fly','fly_visual','news','core','cloud_debug')) + tuple(
    'vendor/stonkfly/stonkfly/neural/'+n for n in ('rule.py','brain.py','controller.py','circuit.py',
    'visual.py','state.py','common.py','arrays.lock.json','sources.lock.json'))


def source_hashes():
    return {name: digest(ROOT/name) for name in FILES}


def validate(payload):
    if not isinstance(payload, dict) or set(payload) != {'protocol_json','pulse_payload_json','reference_json','audit_json'}:
        raise ValueError('Expected the registered protocol and three pinned reference documents')
    if any(not isinstance(v, str) or len(v.encode()) > 2_000_000 for v in payload.values()):
        raise ValueError('Each pinned document must be text below 2 MB')
    if hashlib.sha256(payload['protocol_json'].encode()).hexdigest() != PROTOCOL_SHA:
        raise ValueError('Credit-reset protocol differs from registration')
    p = json.loads(payload['protocol_json'])
    for key, pin in (('pulse_payload_json','reference_pulse_payload_sha256'),
                     ('reference_json','reference_study_sha256'),('audit_json','diagnostic_audit_sha256')):
        if hashlib.sha256(payload[key].encode()).hexdigest() != p[pin]:
            raise ValueError('Pinned reference differs: '+key)
    pulse = json.loads(payload['pulse_payload_json']); _, market, plan = validate_pulse(pulse)
    reference = json.loads(payload['reference_json']); audit_pulse(reference, pulse['reference_json'])
    if hashlib.sha256(pulse['reference_json'].encode()).hexdigest() != p['reference_market_sha256']:
        raise ValueError('Market reference differs')
    audit = json.loads(payload['audit_json'])
    if audit['artifact_sha256'] != p['reference_artifact_sha256'] or not audit['verification']['all_weights_exact']:
        raise ValueError('Original recording audit differs')
    return p, reference, market, plan


def verify_reference(p, root):
    root = Path(root)
    for name, sha in p['reference_artifact_sha256'].items():
        if digest(root/name) != sha:
            raise ValueError('Original native reference artifact differs: '+name)


def compare_trace(path, reference, *, rates=True):
    """Every neuron/bin and plastic state, not only aggregate decoder outputs."""
    keys = ('ms','counts','weights','u','w','initial_weights','neuron_ids','plastic_edges','plastic_pre','plastic_post')
    if rates: keys += ('kc','dan')
    with np.load(path, allow_pickle=False) as a, np.load(reference, allow_pickle=False) as b:
        for key in keys:
            if a[key].dtype != b[key].dtype or not np.array_equal(a[key], b[key]):
                raise ValueError('Full reference control does not reproduce: '+key)


def run(payload, reference_root, data, output):
    p, reference, market, plan = validate(payload)
    reference_root = Path(reference_root); output = Path(output)
    if output.exists(): raise ValueError('Refuse to overwrite an experiment')
    verify_reference(p, reference_root)
    deadline = time.monotonic()+p['limits']['inner_timeout_seconds']
    def check():
        if time.monotonic() > deadline: raise TimeoutError('Registered trace-reset time limit exceeded')
    lab = TraceLab(data); b = lab.brain
    if b.build != market['native_build']:
        raise ValueError('The original native binary and source must match')
    seq = input_sequences(plan, market, reference['protocol']['pool'])['test']
    initial = dynamic_state(b); baseline = learned_state(b)
    memory = read_memory(reference_root/'trained_online_recorded/initial-memory.npz')
    if any(r['initial_memory_sha256'] != memory_signature(memory) for r in
           (reference['reports'][a['reference_arm']] for a in p['arms'].values())):
        raise ValueError('All conditions require the original identical training memory')
    if not np.array_equal((b.baseline_plastic*(1+memory['w'])).astype(b.weight.dtype), memory['weights']):
        raise ValueError('Training memory does not match prepared baseline weights')
    output.mkdir(parents=True)
    atomic_json(output/'protocol.json', p)
    np.savez_compressed(output/'initial-dynamics.npz', **initial)
    np.savez_compressed(output/'trained-memory.npz', **memory)
    np.savez_compressed(output/'pristine-memory.npz', **baseline)
    np.savez_compressed(output/'full-weight-reference.npz', weight=b.weight)
    np.savez_compressed(output/'neuron-ids.npz', neuron_ids=b.ids)
    np.savez_compressed(output/'circuit.npz', edges=b.circuit['edges'], pre=b.circuit['pre'],
        post=b.post[b.circuit['edges']], dan=b.circuit['dan'], gain=b.circuit['gain'], baseline=b.baseline_plastic)
    reports = {}; controls = []; summary = {}
    for name, arm in p['arms'].items():
        check(); restore_learned(b, memory); b.eta = .001; b.weights_frozen = not arm['learning']
        lab.fly.controller.s = replace(lab.fly.controller.s, learning=arm['learning'])
        if any(not np.array_equal(v, getattr(b,k)) for k,v in initial.items()):
            raise ValueError('An arm did not begin with fresh native dynamics')
        if arm['boundary'] != 'carry' and len(controls) != 3:
            raise ValueError('All three controls must finish before intervention')
        folder = output/name; folder.mkdir(); np.savez_compressed(folder/'initial-memory.npz', **memory)
        events = []; started = time.monotonic()
        for i, (rgb, row) in enumerate(seq, 1):
            check(); boundary = apply_boundary(b, arm['boundary'], i, folder/f'boundary-{i:02}.npz')
            pulse = row['neural']['stimulus'] if arm['pulses'] == 'recorded' else 'none'
            event = lab.capture(rgb, pulse, folder, i)
            if event['input_sha256'] != row['neural']['input_sha256']:
                raise ValueError('Sealed image did not reproduce')
            event.update(phase='replay', phase_step=i, market_decision_ts=row['decision_ts'],
                input_preset='recorded_market', input_news='none', activity_boundary=boundary,
                all_weight_sha256=array_hash(b.weight))
            if i < 3:
                path = folder/f'end-{i:02}.npz'; np.savez_compressed(path, **dynamic_state(b))
                event['end_state_sha256'] = digest(path)
            original = reference_root/arm['reference_arm']/f'step-{i:02}.npz'
            if arm['boundary'] == 'carry' or i == 1:
                compare_trace(folder/f'step-{i:02}.npz', original)
            elif not arm['learning']:
                compare_trace(folder/f'step-{i:02}.npz', original, rates=False)
            events.append(event)
        if arm['boundary'] == 'carry': controls.append(name)
        report = {'schema':1, 'source':SOURCE, 'config':{'preset':'recorded_market','view':'original',
            'news':'none','eta':.001,'memory':'trained','state_reset':arm['boundary'],**arm},
            'graph':{'neurons':b.n,'edges':len(b.post),'plastic_edges':len(b.circuit['edges'])},
            'native_build':b.build,'upstream_commit':UPSTREAM_COMMIT,'seconds':time.monotonic()-started,
            'events':events,'initial_memory_sha256':memory_signature(memory),
            'final_memory_sha256':memory_signature(learned_state(b)),
            'reference_report_sha256':p['reference_study_sha256'],
            'plan_sha256':reference['reports'][arm['reference_arm']]['plan_sha256'],
            'pool':reference['protocol']['pool'],'interpretation':p['interpretation']}
        atomic_json(folder/'report.json', report); view = lab.export_view(folder, report)
        attach_boundary_state(view, folder)
        for i, frame in enumerate(view['frames'], 1):
            frame.update(enrich_frame(view, frame, folder/f'step-{i:02}.npz'))
        atomic_json(folder/'view.json', view); reports[name] = report
        summary[name] = {'actions':[e['side'] for e in events], 'gate_spikes':[e['gate_spikes'] for e in events],
            'weight_update_l2':[e['diagnostics']['weight_delta_l2'] for e in events]}
        atomic_json(output/'progress.json', {'completed':list(reports), 'controls_verified':controls})
        print('credit_reset '+name+' '+json.dumps(summary[name]), flush=True)
    result = {'status':'credit_reset_completed','protocol':p,'reports':reports,'summary':summary,
        'controls_verified':controls,'code_sha256':source_hashes(),
        'artifact_sha256':{name:digest(output/name) for name in ('initial-dynamics.npz','trained-memory.npz',
            'pristine-memory.npz','full-weight-reference.npz','neuron-ids.npz','circuit.npz')}}
    atomic_json(output/'summary.json', result)
    return result
