"""Replay a complete online paper ledger and every recorded plasticity bin offline."""
import argparse
import base64
import copy
import io
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from .core import Broker, Tick, atomic_json, digest
from .fly import frame
from .fly_credit_audit import graph_arrays, rule_module
from .fly_learning_scale_credit import reconstruct
from .fly_rate_metrics import trading_metrics
from .fly_credit_reset_audit import audit_update
from .fly_market_activity import array_hash
from .fly_market_activity_audit import DYNAMIC_FIELDS, audit_boundary, audit_view_boundaries, read_arrays
from .fly_market_memory_audit import read_memory
from .fly_market_study import decision_timeline, memory_signature, quote_at
from .fly_rate_protocol import ARMS, chunk_name
from .fly_rate_inputs import SealedNews, audit_news, validate
from .fly_trace_memory import enrich_frame
from .multi import DEX_COSTS


def require(condition, message):
    if not condition:raise ValueError(message)


def audit_chunk(envelope, summary, root, data):
    from .fly_rate_study import SOURCE_FILES, select_development
    root = Path(root); p = validate(envelope); r = p['registration']
    require(r['study']=='12' and summary['registration']==r and summary['plan_sha256']==envelope['sha256'], 'Online protocol differs')
    stage, pool, name = summary['stage'], summary['pool_index'], summary['arm']
    require(summary['status']=='paper_rate_chunk_completed' and summary['chunk']==chunk_name(stage,pool,name), 'Chunk identity differs')
    key = r['cohort'][pool]; arm = ARMS[name]
    require(summary['pool']==key, 'Pool identity differs')
    require(set(summary['code_sha256'])==set(SOURCE_FILES), 'Missing executed source provenance')
    hashes = summary['artifact_sha256']
    for file, sha in hashes.items():
        relative = Path(file)
        require(not relative.is_absolute() and '..' not in relative.parts and digest(root/relative)==sha, 'Artifact differs: '+file)
    required = {'protocol.json', 'news.db', 'pristine-memory.npz', 'initial-dynamics.npz', 'neuron-ids.npz',
        'full-weight-reference.npz', 'circuit.npz', 'imported/audit.json', 'imported/pool0-memory.npz',
        'imported/pool1-memory.npz', 'final-memory.npz', 'final-dynamics.npz', 'trace/initial-memory.npz'}
    require(required <= set(hashes), 'Missing chunk provenance artifacts')
    require(json.loads((root/'protocol.json').read_text())==envelope, 'Saved plan differs')
    if stage=='test':
        require({'development.json','selection.json'}<=set(hashes), 'Missing pre-test development selection')
        selection = select_development(envelope,json.loads((root/'development.json').read_text()))
        require(selection==summary['selection']==json.loads((root/'selection.json').read_text()), 'Test selection differs from development')
    else:require(summary['selection'] is None, 'Development imported selection state')
    news_check = audit_news(envelope,root/'news.db'); news = SealedNews(p['news_features'])
    require(news_check==summary['news_audit'], 'Reported news audit differs')
    require(digest(root/'imported/audit.json')==r['training_audit_sha256'] and
            json.loads((root/'imported/audit.json').read_text())==p['training_audit'], 'Training provenance differs')
    ids, circuit, baseline = graph_arrays(data); rule = rule_module()
    require(summary['graph']=={'neurons':len(ids),'edges':25582938,'plastic_edges':len(baseline)}, 'Graph dimensions differ')
    require(np.array_equal(read_arrays(root/'neuron-ids.npz')['neuron_ids'],ids), 'Neuron identity map differs')
    recorded_circuit = read_arrays(root/'circuit.npz')
    require(set(recorded_circuit)=={'edges','pre','post','dan','gain','baseline'}, 'Learning circuit fields differ')
    for field in ('edges','pre','post','dan','gain'):
        require(np.array_equal(recorded_circuit[field],circuit[field]), 'Learning circuit differs: '+field)
    require(np.array_equal(recorded_circuit['baseline'],baseline), 'Plastic baseline differs')
    full_weight = read_arrays(root/'full-weight-reference.npz')['weight']
    # The unchanged visual adapter makes existing R8 -> aMe12 contacts positive.
    # Reconstruct that declared baseline from locked graph/annotations, without
    # constructing a brain or accepting the recording's own list of corrections.
    from stonkfly.neural.common import annotations
    types=annotations(ids).type.fillna('')
    with np.load(Path(data)/'graph.npz',allow_pickle=False) as prepared:
        expected_weight=prepared['weight'].copy();ptr=prepared['ptr'];post=prepared['post']
        for source in np.flatnonzero(types.str.startswith('R8')):
            edges=np.arange(ptr[source],ptr[source+1]);edges=edges[types.iloc[post[edges]].eq('aMe12').to_numpy()]
            expected_weight[edges]=np.abs(expected_weight[edges])
    require(np.array_equal(full_weight,expected_weight), 'Full graph reference differs from the declared visual model')
    pristine = read_memory(root/'pristine-memory.npz')
    require(np.array_equal(pristine['weights'],baseline) and not np.any(pristine['u']) and not np.any(pristine['w']), 'Pristine memory differs')
    for i,k in enumerate(r['cohort']):
        imported = read_memory(root/f'imported/pool{i}-memory.npz'); source=r['source_memories'][k]
        metadata = p['training_audit']['pools'][k]['source_metadata']
        require(digest(root/f'imported/pool{i}-memory.npz')==source['memory_file_sha256'] and
            memory_signature(imported)==source['memory_sha256'], 'Imported memory differs')
        require(metadata['graph_ids_sha256']==array_hash(ids) and metadata['plastic_edges_sha256']==array_hash(circuit['edges']), 'Imported graph identity differs')
        require(all(summary['native_build'][v]==metadata['build'][v] for v in ('model','source_sha256','flags')), 'Native model differs from imported memory')
        require(np.array_equal(imported['weights'],(baseline*(1+imported['w'])).astype(np.float32)), 'Imported efficacy differs')
    memory = read_memory(root/f'imported/pool{pool}-memory.npz') if arm['memory']=='paper_trained' else pristine
    require(memory_signature(read_memory(root/'trace/initial-memory.npz'))==memory_signature(memory), 'Initial phase memory differs')
    initial = read_arrays(root/'initial-dynamics.npz')
    require(set(initial)==DYNAMIC_FIELDS and not np.any(initial['rate_kc']) and not np.any(initial['rate_dan']), 'Initial dynamics are incomplete or imported traces')
    state = {**{k:v.copy() for k,v in memory.items()}, 'kc':initial['rate_kc'].copy(), 'dan':initial['rate_dan'].copy()}
    previous = initial; clock = {'cursor':0,'sim_ms':0.,'total_spikes':0}
    lookup = {str(v):i for i,v in enumerate(ids)}
    outcome = summary['outcome']; rows = outcome['rows']; events = [row['event'] for row in rows if row['event'] is not None]
    require(len(rows)==r['phase_steps']+1, 'Incomplete decision timeline')
    require(outcome['initial_memory_sha256']==memory_signature(memory), 'Outcome initial memory differs')
    view = json.loads((root/'trace/view.json').read_text()) if events else None
    expanded = copy.deepcopy(view)
    if events:
        report=view['report']; config=report['config']
        require({'trace/view.json','trace/report.json'}<=set(hashes), 'Missing viewer artifacts')
        require(json.loads((root/'trace/report.json').read_text())==report and report['events']==events and
            [f['event'] for f in view['frames']]==events and report['market_timeline']==decision_timeline(rows), 'Viewer events differ')
        require(config=={'preset':'market_replay','view':'original','news':'sealed_timestamped','eta':arm['eta'],
            'learning':arm['learning'],'reinforcement_only':False,'restore_post_ids':[],
            'activity_reset':arm['activity_reset']}, 'Viewer learning configuration differs')
        require(report['evaluation_phase']==stage and report['memory_origin']==arm['memory'] and
            report['plan_sha256']==envelope['sha256'] and report['chunk']==summary['chunk'] and report['pool']==key and
            report['study']=='12' and report['native_build']==summary['native_build'], 'Viewer provenance differs')
        require(report['training_exposure']==(r['source_memories'][key] if arm['memory']=='paper_trained' else None), 'Viewer training exposure differs')
        audit_view_boundaries(view,root/'boundaries',ids)
    else:require(not (root/'trace/view.json').exists(), 'Unobserved phase has a fabricated viewer')
    vendor=Path(__file__).resolve().parents[1]/'vendor/stonkfly'
    if str(vendor) not in sys.path:sys.path.insert(0,str(vendor))
    broker=Broker(DEX_COSTS); pending=None; anchor=250.; unpriced=False; last_quote=0; checked=[]
    ticks=[Tick(**t) for t in p['series'][key]]
    for slot,row in enumerate(rows):
        stamp=r[stage+'_start']+slot*300; index,t=quote_at(ticks,stamp); event=row['event']
        require(row['decision_ts']==stamp and row['quote_ts']==t.ts and row['available']==t.available and row['terminal']==(slot==r['phase_steps']), 'Quote or timeline differs')
        fill=broker.execute(*pending,t) if pending else {'status':'hold'}; pending=None; equity=broker.equity(t)
        require(row['fill']==fill and row['broker']==broker.state() and abs(row['equity']-equity)<1e-8, 'Paper ledger does not replay')
        expected=slot<r['phase_steps'] and t.available and t.ts>last_quote
        require((event is not None)==expected, 'Observed a missing, repeated or terminal quote')
        if event is not None:
            n=len(checked)+1; rgb=frame(ticks,index,news); reward=(0 if unpriced else equity-anchor) if arm['learning'] else 0
            stimulus='reward' if reward>.01 else 'aversive' if reward<-.01 else 'none'
            require(abs(event['equity_reward_usd']-reward)<1e-8 and event['stimulus']==stimulus and
                event['stimulus_ms']==(0 if stimulus=='none' else 200), 'Feedback differs from this arm ledger')
            require(event['plasticity_enabled']==arm['learning'] and event['diagnostics']['plasticity_enabled']==arm['learning'] and not event.get('stimulation'), 'Learning or extra current differs')
            require(event['input_sha256']==array_hash(rgb) and np.array_equal(event['news_features'],news.features(t.ts)) and event['market_decision_ts']==stamp, 'Price/news image differs')
            paths=[f'boundaries/boundary-{n:02}.npz',f'boundaries/end-{n:02}.npz',f'trace/step-{n:02}.npz',f'trace/input-{n:02}.png']
            require(set(paths)<=set(hashes), 'Missing full observation artifacts')
            captured=view['frames'][n-1]
            for image in (Image.open(root/paths[3]),Image.open(io.BytesIO(base64.b64decode(captured['input_png'])))):
                require(np.array_equal(np.asarray(image.convert('RGB')),rgb), 'Saved or displayed image differs')
            boundary=event['activity_boundary']; b=read_arrays(root/paths[0])
            audit_boundary(root/paths[0],boundary,initial,{k:state[k] for k in memory},arm['activity_reset'],n,ids)
            require(boundary['before_clock']==clock, 'Boundary clock differs')
            for field in initial:require(np.array_equal(b['before__'+field],previous[field]), 'Boundary continuity differs: '+field)
            full_weight[circuit['edges']]=state['weights']
            require(boundary['all_weight_sha256']==array_hash(full_weight), 'Boundary changed graph weights')
            state['kc']=b['after__rate_kc'].copy();state['dan']=b['after__rate_dan'].copy()
            a=read_arrays(root/paths[2])
            require(np.array_equal(a['neuron_ids'],ids) and np.array_equal(a['ms'],np.arange(10,501,10)+clock['sim_ms']), 'Trace identities or time grid differ')
            credit,metrics=reconstruct(a,state,circuit,baseline,rule,arm['learning'],np.asarray(view['plastic_selection'],dtype=np.int64),eta=arm['eta'])
            for field in memory:state[field]=a[field][-1].copy()
            total=a['counts'].sum(axis=0).astype(initial['counts'].dtype)
            require(array_hash(total)==event['spike_sha256'] and int(total.sum())==event['total_spikes'], 'Full counts differ')
            left=2*int(total[lookup['10162']]);right=2*int(total[lookup['10059']]);difference=right-left
            gate=sum(int(total[lookup[k]]) for k in ('10527','555871'))
            side='HOLD' if gate==0 or abs(difference)<2 else 'BUY' if difference>0 else 'SELL'
            require(event['cell_ids']=={'left':['10162'],'right':['10059'],'gate':['10527','555871']} and
                (event['side'],event['left_hz'],event['right_hz'],event['difference_hz'],event['gate_spikes'])==
                (side,left,right,difference,gate), 'Fixed decoder differs from full counts')
            delta=a['weights'][-1]-a['initial_weights'];norm=audit_update(delta,event['diagnostics'])
            audit_update(delta,{'weight_delta_l2':event['weight_delta_l2'],'changed_edges':event['diagnostics']['changed_edges']})
            require(event['memory']['sha256']==array_hash(state['weights']), 'Recorded memory differs')
            full_weight[circuit['edges']]=state['weights']
            require(event['all_weight_sha256']==array_hash(full_weight), 'Observation changed nonplastic graph weights')
            clock={'cursor':clock['cursor']+5000,'sim_ms':clock['sim_ms']+500,'total_spikes':clock['total_spikes']+int(total.sum())}
            require(event['brain_ms']==clock['sim_ms'], 'Neural time advanced incorrectly')
            previous=read_arrays(root/paths[1])
            require(set(previous)==DYNAMIC_FIELDS and digest(root/paths[1])==event['end_state_sha256'], 'End state differs')
            for field,desired in (('counts',total),('v',a['voltage'][-1]),('rate_kc',a['kc'][-1]),('rate_dan',a['dan'][-1])):
                require(np.array_equal(previous[field],desired), 'End state differs from trace: '+field)
            selected=np.asarray([node['index'] for node in view['nodes']])
            require(np.array_equal(captured['counts'],a['counts'][:,selected]) and
                np.array_equal(captured['voltage'],a['voltage'][:,selected].round(3)), 'Displayed activity differs')
            for field,value in enrich_frame(view,captured,root/paths[2]).items():
                require(np.array_equal(captured[field],value), 'Displayed memory differs')
            expanded['frames'][n-1]['plastic_credit']=credit
            checked.append({'observation':n,'decision_ts':stamp,'side':side,'equity_reward_usd':reward,
                'stimulus':stimulus,'gate_spikes':gate,'weight_norm_audit':norm,'rule':metrics})
            if side!='HOLD':pending=(.5 if side=='BUY' else 0,stamp)
            last_quote=t.ts;anchor=equity
        unpriced=not t.available
    final=read_arrays(root/'final-dynamics.npz')
    require(set(final)==DYNAMIC_FIELDS and all(np.array_equal(final[k],previous[k]) for k in final), 'Final dynamics differ')
    final_memory=read_memory(root/'final-memory.npz')
    require(all(np.array_equal(final_memory[k],state[k]) for k in memory) and
        outcome['final_memory_sha256']==memory_signature(final_memory), 'Final memory differs')
    if view:
        require(view['report']['initial_memory_sha256']==memory_signature(memory) and
            view['report']['final_memory_sha256']==memory_signature(final_memory), 'Viewer memory signatures differ')
    fills=[row['fill'] for row in rows if row['fill']['status']=='filled']
    require(outcome['start']==r[stage+'_start'] and outcome['end']==r[stage+'_start']+r['phase_steps']*300 and
        abs(outcome['equity']-rows[-1]['equity'])<1e-8 and abs(outcome['return_pct']-100*(outcome['equity']/250-1))<1e-8 and
        outcome['fills']==len(fills) and abs(outcome['fees']-sum(float(f['fee']) for f in fills))<1e-8 and
        outcome['unavailable_marks']==sum(not row['available'] for row in rows), 'Outcome summary does not reconcile')
    require(summary['trading_metrics']==trading_metrics(rows), 'Trading metrics differ from replayed ledger')
    result={'status':'paper_rate_chunk_audited','chunk':summary['chunk'],'plan_sha256':envelope['sha256'],
        'observations':checked,'equity':outcome['equity'],'fees':outcome['fees'],'news':news_check,
        'trading_metrics':trading_metrics(rows),
        'artifact_sha256':hashes,'executed_source_sha256':summary['code_sha256'],'audit_source_sha256':digest(__file__),
        'verification':{'decision_slots':len(rows),'observations':len(checked),'native_bins':50*len(checked),
            'plastic_edges_per_bin':len(baseline),'ledger_replayed':True,'feedback_reconstructed':True,
            'all_boundaries_verified':True,'all_weights_exact':True,'rule_rtol':1e-11,'rule_atol':1e-12}}
    if expanded:
        expanded['validation_only']=any('synthetic' in tick['source'] for values in p['series'].values() for tick in values)
        expanded['credit_audit']={'schema':1,'source_sha256':summary['code_sha256'],
            'audit_source_sha256':digest(__file__),'verification':result['verification'],
            'interpretation':'Full phase ledger, online feedback, boundary continuity and every recorded learning bin audited. Drive components are algebraic, not causal attribution.'}
    return result,expanded


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('root','fly-data','out'):parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise ValueError('Preserve earlier audit output')
    result,view=audit_chunk(json.loads((args.root/'protocol.json').read_text()),
        json.loads((args.root/'summary.json').read_text()),args.root,args.fly_data)
    args.out.mkdir(parents=True);atomic_json(args.out/'audit.json',result)
    if view:atomic_json(args.out/'view.json',view)


if __name__=='__main__':main()
