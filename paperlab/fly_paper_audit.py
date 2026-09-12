"""Reconstruct paper fills, news inputs, decoder outputs and native boundaries."""
import copy
import base64
import io
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from .core import Broker,Tick,digest
from .fly import frame
from .fly_market_activity_audit import read_arrays,audit_boundary,audit_view_boundaries,DYNAMIC_FIELDS
from .fly_market_memory_audit import read_memory
from .fly_market_study import quote_at,memory_signature,decision_timeline
from .fly_paper_inputs import validate,audit_news,SealedNews
from .fly_paper_memory import array_hash
from .fly_paper_protocol import development_choice
from .multi import DEX_COSTS


def audit(envelope,summary,root):
    from .fly_paper_study import PHASES,trace_name
    # The observer has not constructed a Fly. Load only the vendored display
    # adapter for pixel reconstruction; no graph/data preparation or simulation.
    vendor=Path(__file__).resolve().parents[1]/'vendor/stonkfly'
    if not (vendor/'stonkfly/display.py').exists():raise RuntimeError('Vendored display adapter missing')
    if str(vendor) not in sys.path:sys.path.insert(0,str(vendor))
    p=validate(envelope);r=p['registration'];root=Path(root);summary=copy.deepcopy(summary)
    if summary['plan_sha256']!=envelope['sha256'] or summary['registration']!=r or summary['initial_capital']!=1000 or summary['costs']!=r['costs']:
        raise ValueError('Report differs from registered comparison')
    news_check=audit_news(envelope,root/'news.db');news=SealedNews(p['news_features'])
    if digest(root/'imported/audit.json')!=r['training_audit_sha256'] or json.loads((root/'imported/audit.json').read_text())!=p['training_audit']:
        raise ValueError('Imported training provenance differs')
    initial=read_arrays(root/'initial-dynamics.npz');identities=read_arrays(root/'neuron-ids.npz')
    if set(initial)!=DYNAMIC_FIELDS or set(identities)!={'neuron_ids'} or digest(root/'initial-dynamics.npz')!=summary['initial_dynamics_sha256'] or digest(root/'neuron-ids.npz')!=summary['neuron_ids_sha256']:
        raise ValueError('Initial native state or identity file differs')
    ids=identities['neuron_ids']
    if ids.ndim!=1 or len(np.unique(ids))!=len(ids) or initial['v'].shape!=ids.shape:raise ValueError('Invalid full neuron identity map')
    source_metadata=next(iter(p['training_audit']['pools'].values()))['source_metadata']
    if array_hash(ids)!=source_metadata['graph_ids_sha256']:raise ValueError('Neural identities differ from trained graph')
    indices={key:int(np.flatnonzero(ids==int(key))[0]) for key in ('10162','10059','10527','555871')}
    pristine=read_memory(root/'pristine-memory.npz')
    if digest(root/'pristine-memory.npz')!=summary['pristine_memory_file_sha256'] or memory_signature(pristine)!=summary['pristine_memory_sha256']:
        raise ValueError('Pristine memory identity differs')
    results=json.loads((root/'results.json').read_text());selection=json.loads((root/'selection.json').read_text())
    if set(results)!=set(r['cohort']):raise ValueError('Result cohort differs')
    verified={};weight_hashes={};diagnostics={}
    for i,key in enumerate(r['cohort']):
        imported_path=root/f'imported/pool{i}-memory.npz';imported=read_memory(imported_path)
        if digest(imported_path)!=r['source_memories'][key]['memory_file_sha256'] or memory_signature(imported)!=r['source_memories'][key]['memory_sha256']:
            raise ValueError('Imported learned memory differs')
        if set(results[key])!=set(r['arms']):raise ValueError('Missing result arm')
        verified[key]={};diagnostics[key]={};ticks=[Tick(**t) for t in p['series'][key]]
        for name,arm in r['arms'].items():
            if set(results[key][name])!=set(PHASES):raise ValueError('Missing result phase')
            state=imported if arm['memory']=='paper_trained' else pristine;memory_hash=memory_signature(state)
            verified[key][name]={};diagnostics[key][name]={}
            for stage in PHASES:
                outcome=results[key][name][stage];rows=outcome['rows'];folder=root/'boundaries'/f'pool{i}-{name}-{stage}'
                trace=root/trace_name(i,name,stage);saved=read_memory(trace/'initial-memory.npz')
                if any(not np.array_equal(state[k],saved[k]) for k in state) or outcome['initial_memory_sha256']!=memory_hash or outcome['final_memory_sha256']!=memory_hash:
                    raise ValueError('Inference failed to preserve declared memory')
                if len(rows)!=4:raise ValueError('Incomplete decision timeline')
                broker=Broker(DEX_COSTS);pending=None;last_quote=0;events=[];checks=[]
                clock={'cursor':0,'sim_ms':0.,'total_spikes':0};observed=[row for row in rows if row['event'] is not None]
                final_counts=read_arrays(folder/'final-counts.npz')
                if set(final_counts)!={'counts'}:raise ValueError('Missing final full counts')
                for slot,row in enumerate(rows):
                    stamp=r[stage+'_start']+slot*300;index,t=quote_at(ticks,stamp);event=row['event']
                    if row['decision_ts']!=stamp or row['terminal']!=(slot==3) or row['quote_ts']!=t.ts or row['available']!=t.available:
                        raise ValueError('Quote, eligibility or timeline differs from sealed prices')
                    fill=broker.execute(*pending,t) if pending else {'status':'hold'};pending=None
                    if row['fill']!=fill or row['broker']!=broker.state() or abs(row['equity']-broker.equity(t))>1e-8:
                        raise ValueError('Paper fills, positions or equity do not replay')
                    expected=slot<3 and t.available and t.ts>last_quote
                    if (event is not None)!=expected:raise ValueError('Observed an unavailable, repeated or terminal quote')
                    if event is None:continue
                    last_quote=t.ts;events.append(event);n=len(events)
                    rgb=frame(ticks,index,news)
                    if event['input_sha256']!=array_hash(rgb) or not np.array_equal(event['news_features'],news.features(t.ts)):
                        raise ValueError('Event price/news image differs from sealed inputs')
                    if event['stimulus']!='none' or event['stimulus_ms']!=0 or event['plasticity_enabled'] or event['weight_delta_l2']!=0 or event['equity_reward_usd']!=0:
                        raise ValueError('Frozen inference enabled learning or reinforcement')
                    if 'recipient_current' in arm:
                        current_path=trace/f'current-{n:02}.npz';applied=read_arrays(current_path)
                        targets=[int(np.flatnonzero(ids==int(x))[0]) for x in arm['recipient_ids']]
                        expected_current={'target_ids':arm['recipient_ids'],'current':arm['recipient_current'],
                                          'duration_ms':500,'artifact_sha256':digest(current_path)}
                        if event.get('stimulation')!=expected_current or set(applied)!={'durations_ms','currents','target_indices'}:
                            raise ValueError('Registered recipient stimulation differs')
                        if not np.array_equal(applied['target_indices'],targets) or not np.array_equal(applied['durations_ms'],np.full(50,10)) or not np.array_equal(applied['currents'],np.full((50,2),arm['recipient_current'])):
                            raise ValueError('Native recipient targets, current or timing differs')
                    elif event.get('stimulation'):
                        raise ValueError('Unregistered stimulation in the paper comparison')
                    if event['market_decision_ts']!=stamp:raise ValueError('Neural event decision time differs')
                    boundary=event['activity_boundary']
                    if boundary['before_clock']!=clock or event['brain_ms']!=clock['sim_ms']+500:
                        raise ValueError('Phase imported or altered undeclared activity time')
                    check=audit_boundary(folder/f'boundary-{n:02}.npz',boundary,initial,state,arm['activity_reset'],n,ids)
                    if n<len(observed):
                        with np.load(folder/f'boundary-{n+1:02}.npz',allow_pickle=False) as next_boundary:counts=next_boundary['before__counts'].copy()
                    else:counts=final_counts['counts']
                    if counts.shape!=ids.shape or counts.dtype!=initial['counts'].dtype or np.any(counts<0) or array_hash(counts)!=event['spike_sha256'] or int(counts.sum())!=event['total_spikes']:
                        raise ValueError('Full neuron counts differ from recorded event')
                    left=2*int(counts[indices['10162']]);right=2*int(counts[indices['10059']]);gate=sum(int(counts[indices[k]]) for k in ('10527','555871'));difference=right-left
                    side='HOLD' if gate==0 or abs(difference)<2 else 'BUY' if difference>0 else 'SELL'
                    if any(event[k]!=v for k,v in {'left_hz':left,'right_hz':right,'difference_hz':difference,'gate_spikes':gate,'side':side}.items()):
                        raise ValueError('Fixed decoder differs from full neuron counts')
                    if side!='HOLD':pending=(.5 if side=='BUY' else 0,stamp)
                    clock={'cursor':clock['cursor']+5000,'sim_ms':event['brain_ms'],'total_spikes':clock['total_spikes']+event['total_spikes']}
                    shared=weight_hashes.setdefault((key,arm['memory']),boundary['all_weight_sha256'])
                    if shared!=boundary['all_weight_sha256']:raise ValueError('Whole graph weights differ across frozen conditions')
                    checks.append(check)
                if not events and np.any(final_counts['counts']):raise ValueError('Unobserved phase has nonzero neuron counts')
                fills=[row['fill'] for row in rows if row['fill']['status']=='filled']
                if outcome['start']!=r[stage+'_start'] or outcome['end']!=r[stage+'_start']+900 or abs(outcome['return_pct']-100*(outcome['equity']/250-1))>1e-8:
                    raise ValueError('Phase timestamps or return differ')
                if abs(outcome['equity']-rows[-1]['equity'])>1e-8 or outcome['fills']!=len(fills) or abs(outcome['fees']-sum(float(f['fee']) for f in fills))>1e-8:
                    raise ValueError('Phase summary does not reconcile with its ledger')
                if events:
                    view=json.loads((trace/'view.json').read_text());vreport=view['report']
                    if vreport['events']!=events or [f['event'] for f in view['frames']]!=events or vreport['market_timeline']!=decision_timeline(rows):raise ValueError('Viewer events or market timeline differs')
                    if vreport['config']['news']!='sealed_timestamped' or vreport['config']['activity_reset']!=arm['activity_reset'] or vreport['evaluation_phase']!=stage or vreport['memory_origin']!=arm['memory']:
                        raise ValueError('Viewer inference provenance differs')
                    if 'recipient_current' in arm and any(vreport['config'].get(k)!=arm[k] for k in ('recipient_current','recipient_ids')):
                        raise ValueError('Viewer recipient intervention differs')
                    expected_exposure=r['source_memories'][key] if arm['memory']=='paper_trained' else None
                    if vreport.get('training_exposure')!=expected_exposure:raise ValueError('Viewer training exposure differs from pinned checkpoint')
                    if vreport['native_build']!=summary['native_build'] or vreport['plan_sha256']!=envelope['sha256'] or vreport['initial_memory_sha256']!=memory_hash or vreport['final_memory_sha256']!=memory_hash or vreport['config']['learning'] is not False:
                        raise ValueError('Viewer model or memory identity differs')
                    for captured,event in zip(view['frames'],events):
                        rgb=np.asarray(Image.open(io.BytesIO(base64.b64decode(captured['input_png']))))
                        if array_hash(rgb)!=event['input_sha256']:raise ValueError('Displayed image differs from the sealed input')
                    audit_view_boundaries(view,folder,ids)
                elif (trace/'view.json').exists():raise ValueError('Unobserved phase has a fabricated viewer')
                verified[key][name][stage]={'observations':len(events),'boundaries':checks,'full_count_decoders_verified':len(events),'paper_ledger_replayed':True,
                    **({'recipient_current_observations_verified':len(events)} if 'recipient_current' in arm else {})}
                diagnostics[key][name][stage]={**{k:v for k,v in outcome.items() if k!='rows'},'decisions':[{**{k:v for k,v in row.items() if k!='event'},'neural':row['event']} for row in rows]}
    for name in r['arms']:
        for stage in PHASES:
            total=500+sum(results[key][name][stage]['equity'] for key in r['cohort'])
            if abs(summary['total_equity'][name][stage]-total)>1e-8:raise ValueError('Aggregate equity differs')
    equities={name:summary['total_equity'][name]['development'] for name in r['arms']}
    expected={'selected':development_choice(equities),'development_equity':equities,'cash_equity':1000,'plan_sha256':envelope['sha256'],'test_simulated_before_selection':False}
    if selection!=expected or summary['selection']!=expected:raise ValueError('Selection differs from registered development gate')
    summary['phase_diagnostics']=diagnostics
    summary['verification']={'news_reconstructed':True,'imported_memory_verified':True,'all_paper_ledgers_replayed':True,'all_full_count_decoders_verified':True,'activity_boundaries_audited':True,'selection_matches_development_gate':True}
    return summary,{'plan_sha256':envelope['sha256'],'news':news_check,'pools':verified}
