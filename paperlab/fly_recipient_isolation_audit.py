"""Audit selective recipient stimulation and reproduce all original control bins."""
import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .core import digest
from .fly_market_activity_audit import DYNAMIC_FIELDS, audit_boundary, audit_view_boundaries, read_arrays
from .fly_market_memory_audit import read_memory
from .fly_market_restoration import matches
from .fly_market_study import memory_signature
from .fly_paper_inputs import audit_news
from .fly_paper_memory import array_hash
from .fly_paper_stimulation import INFERENCE, SOURCE, TARGETS, sequences
from .fly_recipient_isolation import SETS, validate


def require(condition, message):
    if not condition: raise ValueError(message)


def audit_current(event, currents, path, arm, lookup):
    targets = [lookup[x] for x in arm['target_ids']]
    require(event['stimulation'] == {'target_ids':arm['target_ids'],'current':arm['current'],
            'duration_ms':500,'artifact_sha256':digest(path)}, 'Current provenance differs')
    require(set(currents) == {'durations_ms','currents','target_indices'} and
            np.array_equal(currents['target_indices'],targets), 'Applied current targeted different cells')
    require(np.array_equal(currents['durations_ms'],np.full(50,10)) and
            np.array_equal(currents['currents'],np.full((50,len(targets)),arm['current'])), 'Applied current or duration differs')


def audit(study, payload, artifacts, reference_artifacts):
    p, parent, parent_audit, plan = validate(payload); root = Path(artifacts); reference_root = Path(reference_artifacts)
    ref = json.loads(payload['paper_payload']['reference_json']); seq = sequences(payload['paper_payload'])
    require(study['status'] == 'paper_recipient_isolation_completed' and study['protocol'] == p, 'Study protocol differs')
    require(list(study['reports']) == list(p['arms']), 'Missing or reordered stimulation conditions')
    require(study['controls_verified'] == [n for n,a in p['arms'].items() if a['target_set']=='both'], 'Missing controls')
    require(study['news_audit'] == audit_news(payload['paper_payload']['paper_plan'], root/'news.db'), 'News audit differs')
    require(digest(root/'imported/audit.json') == plan['registration']['training_audit_sha256'], 'Training audit differs')
    initial = read_arrays(root/'initial-dynamics.npz'); ids = read_arrays(root/'neuron-ids.npz')['neuron_ids']
    require(set(initial) == DYNAMIC_FIELDS and digest(root/'initial-dynamics.npz') == study['initial_dynamics_sha256'], 'Initial dynamics differ')
    require(study['initial_dynamics_sha256'] == parent['initial_dynamics_sha256'], 'Fresh dynamics differ from parent')
    require(digest(root/'neuron-ids.npz') == study['neuron_ids_sha256'], 'Neuron identity artifact differs')
    metadata = next(iter(plan['training_audit']['pools'].values()))['source_metadata']
    require(ids.ndim == 1 and len(ids) == 166700 and len(np.unique(ids)) == len(ids) and array_hash(ids) == metadata['graph_ids_sha256'], 'Full neuron identity map differs')
    lookup = {str(v): i for i,v in enumerate(ids)}
    pristine = read_memory(root/'pristine-memory.npz')
    require(memory_signature(pristine) == ref['pristine_memory_sha256'], 'Pristine memory differs')
    imported = {}
    for i,key in enumerate(plan['registration']['cohort']):
        path = root/f'imported/pool{i}-memory.npz'; mem = read_memory(path); pin = plan['registration']['source_memories'][key]
        require(digest(path) == pin['memory_file_sha256'] and memory_signature(mem) == pin['memory_sha256'], 'Imported memory differs')
        imported[key] = mem
    outcomes = {}; hashes = {}; counts_by_arm = {}; reference_hashes = {}
    for name, arm in p['arms'].items():
        folder = root/name; report = study['reports'][name]; memory = pristine if arm['memory']=='pristine' else imported[arm['pool']]
        pool_index = plan['registration']['cohort'].index(arm['pool'])
        old_name = f'pool{pool_index}-{arm["memory"]}-current10'
        require(report['all_weight_sha256'] == parent['reports'][old_name]['all_weight_sha256'], 'Full graph weights differ from parent')
        require(report['config'] == {'preset':'recorded_market','view':'original','news':'sealed_timestamped','pulses':'none',**INFERENCE,**arm,'neurons':TARGETS}, 'Inference configuration differs')
        require(report['source'] == SOURCE and report['native_build'] == ref['native_build'] and report['plan_sha256'] == p['plan_sha256'] and report['reference_report_sha256'] == p['parent_report_sha256'], 'Recording provenance differs')
        require(report['initial_memory_sha256'] == report['final_memory_sha256'] == memory_signature(memory), 'Frozen memory differs')
        require(memory_signature(read_memory(folder/'initial-memory.npz')) == memory_signature(memory), 'Saved arm memory differs')
        view = json.loads((folder/'view.json').read_text()); events = report['events']
        require(view['report'] == report and len(events) == len(view['frames']) == 3 and [f['event'] for f in view['frames']] == events, 'Incomplete or mismatched view')
        audit_view_boundaries(view, folder, ids)
        selected = np.array([n['index'] for n in view['nodes']]); plastic_selected = view['plastic_selection']
        rows = []; previous_counts = initial['counts']; previous_voltage = initial['v']; previous_clock = 0
        counts_by_arm[name] = []
        for j,(event,(rgb,row),f) in enumerate(zip(events,seq[arm['pool']],view['frames']),1):
            require(event['market_decision_ts'] == row['decision_ts'] and event['quote_ts'] == row['quote_ts'] and event['news_features'] == row['neural']['news_features'], 'Recorded market clocks or news differ')
            require(event['input_sha256'] == row['neural']['input_sha256'] and event['stimulus'] == 'none' and event['stimulus_ms'] == 0, 'Input or reinforcement differs')
            require(event['diagnostics']['plasticity_enabled'] is False and event['diagnostics']['changed_edges'] == 0 and event['diagnostics']['weight_delta_l2'] == 0, 'Learning occurred during frozen inference')
            require(np.array_equal(np.asarray(Image.open(folder/f'input-{j:02}.png').convert('RGB')),rgb), 'Saved input differs')
            require(np.array_equal(np.asarray(Image.open(io.BytesIO(base64.b64decode(f['input_png']))).convert('RGB')),rgb), 'Displayed input differs')
            current_path = folder/f'current-{j:02}.npz'; currents = read_arrays(current_path)
            audit_current(event, currents, current_path, arm, lookup)
            boundary = event['activity_boundary']; audit_boundary(folder/f'boundary-{j:02}.npz',boundary,initial,memory,'carry',j,ids)
            before = read_arrays(folder/f'boundary-{j:02}.npz')
            require(np.array_equal(before['before__counts'],previous_counts) and np.array_equal(before['before__v'],previous_voltage), 'Activity was reset between observations')
            require(boundary['before_clock']['sim_ms'] == previous_clock and boundary['all_weight_sha256'] == report['all_weight_sha256'], 'Clock or graph weights differ')
            path = folder/f'step-{j:02}.npz'; a = read_arrays(path); counts = a['counts']
            require(np.array_equal(a['neuron_ids'],ids) and array_hash(a['plastic_edges']) == metadata['plastic_edges_sha256'], 'Trace graph identities differ')
            require(counts.shape == (50,len(ids)) and counts.dtype.kind in 'iu' and np.all(counts>=0), 'Invalid full-neuron spike array')
            require(a['voltage'].shape == counts.shape and np.isfinite(a['voltage']).all(), 'Invalid voltage array')
            total = counts.sum(axis=0).astype(initial['counts'].dtype)
            require(array_hash(total) == event['spike_sha256'] and int(total.sum()) == event['total_spikes'], 'Full-neuron counts differ from report')
            left = int(total[lookup['10162']])*2; right = int(total[lookup['10059']])*2
            gate = int(total[lookup['10527']]+total[lookup['555871']]); diff = right-left
            side = 'HOLD' if gate==0 or abs(diff)<2 else 'BUY' if diff>0 else 'SELL'
            require((event['left_hz'],event['right_hz'],event['difference_hz'],event['gate_spikes'],event['side']) == (left,right,diff,gate,side), 'Decoder differs from full-neuron counts')
            require(np.array_equal(a['ms'],np.arange(10,501,10)+previous_clock) and event['brain_ms']==previous_clock+500, 'Observation duration differs')
            for array_name, mem_name in (('weights','weights'),('u','u'),('w','w')):
                require(a[array_name].shape == (50,7835) and np.array_equal(a[array_name],np.broadcast_to(memory[mem_name],(50,7835))), 'Plastic memory changed inside an observation')
            require(np.array_equal(a['initial_weights'],memory['weights']), 'Initial recording weights differ')
            require(np.array_equal(f['counts'],counts[:,selected]) and np.array_equal(f['voltage'],a['voltage'][:,selected].round(3)), 'Displayed neural activity differs')
            for displayed, native in (('plastic_weights','weights'),('plastic_u','u'),('plastic_w','w')):
                require(np.array_equal(f[displayed],a[native][:,plastic_selected]), 'Displayed memory differs')
            if arm['target_set']=='both':
                require(matches(event,parent['reports'][old_name]['events'][j-1]), 'Original control decoder did not reproduce')
                relative = old_name+f'/step-{j:02}.npz'; old_path = reference_root/relative
                require(digest(old_path)==parent_audit['artifact_sha256'][relative], 'Original native control evidence differs')
                old = read_arrays(old_path)
                require(np.array_equal(old['neuron_ids'],ids) and np.array_equal(old['counts'],counts), 'Original full-neuron control bins did not reproduce')
                reference_hashes[relative] = digest(old_path)
            rows.append({'observation':j,'decision_ts':row['decision_ts'],'input_sha256':event['input_sha256'],
                'side':side,'difference_hz':diff,'gate_spikes':gate,'total_spikes':int(total.sum()),
                'recipient_counts':{identity:int(total[lookup[identity]]) for identity in TARGETS},
                'recipient_sampled_voltage_max':{identity:float(a['voltage'][:,lookup[identity]].max()) for identity in TARGETS},
                'spike_sha256':event['spike_sha256']})
            counts_by_arm[name].append(total)
            previous_counts = total; previous_voltage = a['voltage'][-1]; previous_clock += 500
            for file in (path,current_path,folder/f'boundary-{j:02}.npz'): hashes[str(file.relative_to(root))] = digest(file)
        outcomes[name] = rows; hashes[name+'/view.json'] = digest(folder/'view.json')
    comparisons = []
    for label in SETS:
        for i,key in enumerate(plan['registration']['cohort']):
            a=f'pool{i}-pristine-{label}'; b=f'pool{i}-trained-{label}'
            comparisons.append({'pool':key,'target_set':label,'pristine':a,'trained':b,
                'changed_neurons':[int(np.count_nonzero(x!=y)) for x,y in zip(counts_by_arm[a],counts_by_arm[b])],
                'different_actions':[x['side']!=y['side'] for x,y in zip(outcomes[a],outcomes[b])]})
    isolation = []
    for i,key in enumerate(plan['registration']['cohort']):
        for memory in ('pristine','trained'):
            baseline=f'pool{i}-{memory}-both'
            for label in ('only_10704','only_11402'):
                variant=f'pool{i}-{memory}-{label}'
                isolation.append({'pool':key,'memory':memory,'baseline':baseline,'variant':variant,
                    'changed_neurons':[int(np.count_nonzero(x!=y)) for x,y in zip(counts_by_arm[baseline],counts_by_arm[variant])],
                    'different_actions':[x['side']!=y['side'] for x,y in zip(outcomes[baseline],outcomes[variant])]})
    require(len(reference_hashes)==12,'Missing original native control evidence')
    return {'status':'verified','protocol':p,'full_observations_verified':36,'native_bins_verified':1800,
            'controls_verified':study['controls_verified'],'rows':outcomes,'comparisons':comparisons,
            'artifact_sha256':hashes,'reference_control_artifact_sha256':reference_hashes,
            'isolation_comparisons':isolation,'interpretation':p['interpretation']}
