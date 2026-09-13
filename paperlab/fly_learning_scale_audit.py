"""Independent full-recording audit at the registered learning rate."""
import copy
import json
from pathlib import Path

import numpy as np

from .core import digest
from .fly_credit_audit import rule_module
from .fly_learning_scale_credit import reconstruct
from .fly_learning_scale import compare_recording, validate, verify_reference, source_hashes, require_completed_study
from .fly_market_activity import array_hash
from .fly_market_activity_audit import DYNAMIC_FIELDS, audit_view_boundaries, read_arrays
from .fly_market_memory_audit import read_memory
from .fly_market_pulse import input_sequences
from .fly_market_study import memory_signature
from .fly_trace_memory import enrich_frame
from .fly_view_projection import load_graph, audit_view, audit_image


from .fly_selective_trace_audit import require, audit_boundary, audit_update


def audit(study, payload, artifacts, reference_root, data, *, completed_study):
    p, parent, market, plan = validate(payload); root = Path(artifacts); reference_root = Path(reference_root)
    references={n:json.loads((reference_root/n/"view.json").read_text())["report"] for n in parent["protocol"]["arms"]}
    verify_reference(parent, reference_root)
    require(study['status']=='learning_scale_captured_pending_audit' and study['protocol']==p, 'Protocol/result differs')
    require(list(study['reports'])==list(p['arms']), 'Missing or reordered conditions')
    require(study['controls_verified']==p['control_arms'], 'Missing controls')
    require(study['code_sha256']==source_hashes(), 'Executed source differs')
    require(study['completed_study_sha256']==require_completed_study(completed_study), 'Study 11 release evidence differs')
    graph=load_graph(data);ids=graph.ids;circuit=graph.circuit
    baseline=graph.weight[circuit['edges']].copy();rule=rule_module()
    require(set(study['artifact_sha256'])=={'neuron-ids.npz','circuit.npz','trained-memory.npz',
        'pristine-memory.npz','initial-dynamics.npz','full-weight-reference.npz'}, 'Incomplete root artifact manifest')
    for name, sha in study['artifact_sha256'].items(): require(digest(root/name)==sha, 'Study artifact differs: '+name)
    require(np.array_equal(read_arrays(root/'neuron-ids.npz')['neuron_ids'],ids), 'Neuron identity map differs')
    saved_circuit=read_arrays(root/'circuit.npz')
    for key in ('edges','pre','post','dan','gain'):
        require(np.array_equal(saved_circuit[key],circuit[key]), 'Recorded learning circuit differs: '+key)
    require(np.array_equal(saved_circuit['baseline'],baseline), 'Pristine plastic weights differ')
    initial=read_arrays(root/'initial-dynamics.npz'); require(set(initial)==DYNAMIC_FIELDS, 'Missing initial native fields')
    parent_initial=read_arrays(reference_root/'initial-dynamics.npz')
    require(set(parent_initial)==set(initial) and all(initial[k].dtype==parent_initial[k].dtype
        and np.array_equal(v,parent_initial[k]) for k,v in initial.items()), 'Initial native dynamics differ from parent')
    require(not np.any(initial['rate_kc']) and not np.any(initial['rate_dan']), 'Initial rate traces must be zero')
    memory=read_memory(root/'trained-memory.npz')
    original_memory=read_memory(reference_root/'trained-memory.npz')
    require(memory_signature(memory)==memory_signature(original_memory), 'Changed imported memory')
    pristine=read_memory(root/'pristine-memory.npz')
    require(np.array_equal(pristine['weights'],baseline) and not np.any(pristine['u']) and not np.any(pristine['w']), 'Invalid pristine memory')
    full_weight=read_arrays(root/'full-weight-reference.npz')['weight']
    require(full_weight.shape==(25582938,) and full_weight.dtype==np.float32 and np.isfinite(full_weight).all(), 'Invalid full weight reference')
    require(np.array_equal(full_weight,read_arrays(reference_root/'full-weight-reference.npz')['weight']), 'Full weight baseline differs from parent')
    require(np.array_equal(full_weight[circuit['edges']],baseline), 'Reference plastic weights differ')
    seq=input_sequences(plan,market,references['trained_online_recorded_carry']['pool'])['test']
    lookup={str(v):i for i,v in enumerate(ids)}; outcomes={}; views={}; projection_checks={}; hashes=dict(study['artifact_sha256'])
    for name, arm in p['arms'].items():
        folder=root/name; report=study['reports'][name]; original=references.get(name,references[arm['reference_arm']+'_carry'])
        require(report['graph']=={'neurons':166700,'edges':25582938,'plastic_edges':7835}, 'Changed graph size')
        require(report['reference_report_sha256']==p['parent_audit_sha256'] and report['pool']==original['pool']
            and report['upstream_commit']==original['upstream_commit'], 'Reference identity differs')
        require(json.loads((folder/'report.json').read_text())==report, 'Saved report differs')
        require(report['native_build']==original['native_build'] and report['plan_sha256']==original['plan_sha256'], 'Native build or plan differs')
        require(report['initial_memory_sha256']==memory_signature(memory), 'Wrong initial memory signature')
        require(memory_signature(read_memory(folder/'initial-memory.npz'))==memory_signature(memory), 'Wrong arm memory')
        require(report['config']=={'preset':'recorded_market','view':'original','news':'none','eta':arm['eta'],
            'memory':'trained','state_reset':arm['boundary'],**arm}, 'Arm configuration differs')
        view=json.loads((folder/'view.json').read_text()); events=report['events']
        projection_checks[name]=audit_view(view,folder,graph)
        require(len(events)==len(view['frames'])==3 and view['report']==report and
                [f['event'] for f in view['frames']]==events, 'Mismatched recording report')
        audit_view_boundaries(view,folder,ids); expanded=copy.deepcopy(view)
        state={**{k:v.copy() for k,v in memory.items()}, 'kc':initial['rate_kc'].copy(), 'dan':initial['rate_dan'].copy()}
        previous=initial; clock={'cursor':0,'sim_ms':0.,'total_spikes':0}; outcomes[name]=[]
        for i,(event,frame,(rgb,row)) in enumerate(zip(events,view['frames'],seq),1):
            require(event['market_decision_ts']==row['decision_ts'] and event['input_sha256']==row['neural']['input_sha256'], 'Changed image/timestamp')
            pulse=row['neural']['stimulus'] if arm['pulses']=='recorded' else 'none'
            require(event['stimulus']==pulse and event['stimulus_ms']==(0 if pulse=='none' else 200), 'Changed pulse')
            audit_image(frame,folder/f'input-{i:02}.png',rgb)
            boundary=event['activity_boundary']; boundary_path=folder/f'boundary-{i:02}.npz'
            full_weight[circuit['edges']]=state['weights']
            audit_boundary(boundary_path,boundary,initial,{k:state[k] for k in memory},arm['boundary'],i,full_weight)
            b=read_arrays(boundary_path)
            for key in initial:
                require(np.array_equal(b['before__'+key],previous[key]), 'Boundary disconnected from previous state: '+key)
            require(boundary['before_clock']==clock, 'Boundary clock differs')
            full_weight[circuit['edges']]=state['weights']
            require(boundary['all_weight_sha256']==array_hash(full_weight), 'Boundary changed full graph weights')
            state['kc']=b['after__rate_kc'].copy(); state['dan']=b['after__rate_dan'].copy()
            path=folder/f'step-{i:02}.npz'; a=read_arrays(path)
            require(np.array_equal(a['neuron_ids'],ids), 'Recorded neuron order differs')
            selected_memory=enrich_frame(view,frame,path)
            for key,value in selected_memory.items(): require(np.array_equal(frame[key],value), 'Displayed memory differs')
            require(np.array_equal(a['ms'],np.arange(10,501,10)+clock['sim_ms']), 'Native time grid differs')
            credit,metrics=reconstruct(a,state,circuit,baseline,rule,arm['learning'],np.asarray(view['plastic_selection'],dtype=np.int64),eta=arm['eta'])
            # The rule was checked through all 50 bins. Fingerprints and boundary
            # continuity use the actual recorded values, not roundoff from a
            # different platform's BLAS implementation of that reconstruction.
            for key in ('weights','u','w'):state[key]=a[key][-1].copy()
            total=a['counts'].sum(axis=0).astype(initial['counts'].dtype)
            require(array_hash(total)==event['spike_sha256'] and int(total.sum())==event['total_spikes'], 'Full spike totals differ')
            left=int(total[lookup['10162']])*2; right=int(total[lookup['10059']])*2
            gate=int(total[lookup['10527']]+total[lookup['555871']]); difference=right-left
            side='HOLD' if not gate or abs(difference)<2 else 'BUY' if difference>0 else 'SELL'
            require((event['side'],event['left_hz'],event['right_hz'],event['difference_hz'],event['gate_spikes'])==
                    (side,left,right,difference,gate), 'Fixed decoder differs from counts')
            require(event['cell_ids']==original['events'][i-1]['cell_ids'], 'Decoder identities differ')
            require(event['diagnostics']['plasticity_enabled']==arm['learning'], 'Learning flag differs')
            delta=a['weights'][-1]-a['initial_weights']
            norm_check=audit_update(delta,event['diagnostics'])
            require(event['memory']['sha256']==array_hash(a['weights'][-1]), 'Event memory differs')
            full_weight[circuit['edges']]=a['weights'][-1]
            require(event['all_weight_sha256']==array_hash(full_weight), 'Observation changed nonplastic weights')
            selected=np.asarray([n['index'] for n in view['nodes']])
            require(np.array_equal(frame['voltage'],a['voltage'][:,selected].round(3)), 'Displayed voltage differs')
            source=name if name in parent['protocol']['arms'] else arm['reference_arm']+'_carry'
            if name in p['control_arms']:
                compare_recording(path,reference_root/source/path.name,rates=True)
            clock={'cursor':clock['cursor']+5000,'sim_ms':clock['sim_ms']+500,
                   'total_spikes':clock['total_spikes']+int(total.sum())}
            require(event['brain_ms']==clock['sim_ms'], 'Event clock differs')
            if i<3:
                end=folder/f'end-{i:02}.npz'; require(digest(end)==event['end_state_sha256'], 'End state hash differs')
                previous=read_arrays(end); require(set(previous)==DYNAMIC_FIELDS, 'End state fields differ')
                for key, expected in (('counts',total),('v',a['voltage'][-1]),('rate_kc',a['kc'][-1]),('rate_dan',a['dan'][-1])):
                    require(np.array_equal(previous[key],expected), 'End state does not match native trace: '+key)
                hashes[str(end.relative_to(root))]=digest(end)
            outcomes[name].append({'observation':i,'side':side,'gate_spikes':gate,'difference_hz':difference,
                'weight_update_l2':norm_check['reported'], 'weight_norm_audit':norm_check,
                'spike_sha256':event['spike_sha256'],
                'changed_boundary_fields':boundary['changed_fields'], 'rule':metrics})
            expanded['frames'][i-1]['plastic_credit']=credit
            for file in (path,boundary_path,folder/f'input-{i:02}.png'):
                hashes[str(file.relative_to(root))]=digest(file)
        require(report['final_memory_sha256']==memory_signature(state), 'Final memory signature differs')
        require(study['summary'][name]=={'actions':[r['side'] for r in outcomes[name]],
            'gate_spikes':[r['gate_spikes'] for r in outcomes[name]],
            'weight_update_l2':[r['weight_update_l2'] for r in outcomes[name]]}, 'Summary differs')
        hashes[name+'/view.json']=digest(folder/'view.json'); hashes[name+'/initial-memory.npz']=digest(folder/'initial-memory.npz')
        hashes[name+'/report.json']=digest(folder/'report.json')
        views[name]=expanded
    comparisons=[]
    from .fly_credit_divergence import compare_arrays
    for name,arm in p['arms'].items():
        if name in parent['protocol']['arms']:continue
        for control_mode in ('carry',):
            control=arm['reference_arm']+'_'+control_mode;rows=[]
            for i in range(1,4):
                with np.load(root/control/f'step-{i:02}.npz',allow_pickle=False) as left, np.load(root/name/f'step-{i:02}.npz',allow_pickle=False) as right:
                    rows.append({'observation':i,**compare_arrays(left,right)})
            comparisons.append({'control':control,'intervention':name,'observations':rows})
    result={'status':'learning_scale_audited','protocol':p,'arms':outcomes,'comparisons':comparisons,
        'view_projections':projection_checks,
        'artifact_sha256':hashes,'code_sha256':study['code_sha256'],'audit_source_sha256':digest(__file__),
        'verification':{'observations':15,'bins':750,'plastic_edges_per_bin':7835,'all_weights_exact':True,
            'all_three_original_controls_reproduced':True,'all_boundaries_verified':True,'all_rate_histories_carried':True,
            'all_decoders_reconstructed':True,'rule_rtol':1e-11,'rule_atol':1e-12,
            'all_displayed_topology_and_series_verified':True,
            'reported_norm_tolerance_float32_ulps':4},'interpretation':p['interpretation']}
    for name,view in views.items():
        view['credit_audit']={'schema':1,'source_sha256':study['code_sha256'],
            'audit_source_sha256':result['audit_source_sha256'],'verification':result['verification'],
            'artifact_sha256':{k:v for k,v in hashes.items() if k.startswith(name+'/')},
            'interpretation':'Learning drive reconstructed at each registered eta with both rate histories carried. Components are algebraic; full trajectories come from separate native runs.'}
    return result, views
