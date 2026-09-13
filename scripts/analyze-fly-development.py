"""Read audited development arrays only; no model construction or propagation."""
import json
from pathlib import Path
import numpy as np
from paperlab.core import atomic_json,digest
from paperlab.fly_market_study import signature
from paperlab.fly_online_protocol import ARMS,chunk_order,chunk_name
from paperlab.fly_online_study import select_development
from paperlab.fly_credit_divergence import compare_arrays
original=Path('runs/online-cloud-11');parent=Path('runs/online-recovery-02');current=Path('runs/online-completion-01')
envelope=json.loads((original/'plan.json').read_text());summaries={};proofs={};roots={}
for i,c in enumerate(chunk_order()[:8]):
 name=chunk_name(*c);base=original if i<2 else parent if i<6 else current;root=base/'chunks'/name
 projection=Path('runs/online-first-projection-11' if i==0 else 'runs/online-second-projection-11') if i<2 else base/'projections'/name
 s=json.loads((root/'summary.json').read_text());p=json.loads((projection/'report.json').read_text());a=json.loads((projection/'audit.json').read_text());r=json.loads((base/'receipts'/(name+'-completed.json')).read_text())
 assert p['status']=='online_view_projection_audited' and p['validation_only'] is False
 assert digest(root/'summary.json')==p['summary_sha256']==r['summary_sha256']
 assert digest(projection/'audit.json')==p['native_audit_sha256'] and digest(projection/'view.json')==p['audited_view_sha256']
 assert a['status']=='paper_online_chunk_audited' and all(a['verification'][k] for k in ('ledger_replayed','feedback_reconstructed','all_boundaries_verified','all_weights_exact'))
 assert p['projection']['all_topology_verified'] and p['projection']['all_plotted_series_verified']
 summaries[name]=s;roots[name]=root
 proofs[name]={'summary_sha256':r['summary_sha256'],'audit_sha256':p['native_audit_sha256'],'projection_sha256':digest(projection/'report.json'),'receipt_sha256':digest(base/'receipts'/(name+'-completed.json')),'call_id':r['call_id'],'observations':a['verification']['observations'],'bins':a['verification']['native_bins'],'equity':s['outcome']['equity'],'fees':s['outcome']['fees'],'fills':s['outcome']['fills']}
selection=select_development(envelope,summaries);assert selection==json.loads((current/'selection.json').read_text())
unique_slots={(s['pool_index'],row['decision_ts']) for s in summaries.values() for row in s['outcome']['rows'] if row['event'] is not None}
result={'status':'all_eight_development_conditions_audited','plan_sha256':envelope['sha256'],'conditions':proofs,'selection':selection,'selection_receipt':json.loads((current/'selection-receipt.json').read_text()),'observations':sum(p['observations'] for p in proofs.values()),'unique_asset_time_slots':len(unique_slots),'native_bins':sum(p['bins'] for p in proofs.values()),'held_out_results':False,'policy_promoted':False,'source_sha256':digest(__file__),'interpretation':'All eight development recordings and their independent full-bin audits reconcile. The 188 neural observations repeat 47 eligible asset/time slots across four conditions, not 188 independent market observations. Reset has the highest development equity but remains below cash plus hosting, so no candidate is selected. Held-out results remain separate.'}
atomic_json('reports/fly-online-development-result-11.json',result)
pairs=[]
for pool in range(2):
 carry=chunk_name('development',pool,'trained_online_carry');reset=chunk_name('development',pool,'trained_online_reset_rates')
 obs=[]
 for i in (1,2):
  paths=[roots[n]/'trace'/f'step-{i:02}.npz' for n in (carry,reset)]
  for n,path in zip((carry,reset),paths):assert digest(path)==summaries[n]['artifact_sha256'][f'trace/step-{i:02}.npz']
  events={label:[r['event'] for r in summaries[n]['outcome']['rows'] if r['event']][i-1] for label,n in [('carry',carry),('reset',reset)]}
  assert all(events['carry'][k]==events['reset'][k] for k in ('input_sha256','market_decision_ts','stimulus','equity_reward_usd'))
  with np.load(paths[0],allow_pickle=False) as left,np.load(paths[1],allow_pickle=False) as right:
   compared=compare_arrays(left,right)
   if i==1:assert set(left.files)==set(right.files) and all(left[f].dtype==right[f].dtype and np.array_equal(left[f],right[f]) for f in left.files)
  for label in ('carry','reset'):
   assert all(compared['decoder'][label][k]==events[label][k] for k in ('side','difference_hz','gate_spikes'))
  obs.append({'observation':i,'input_sha256':events['carry']['input_sha256'],'market_decision_ts':events['carry']['market_decision_ts'],'stimulus':events['carry']['stimulus'],'equity_reward_usd':events['carry']['equity_reward_usd'],'trace_sha256':{k:digest(p) for k,p in zip(('carry','reset'),paths)},**compared})
 pairs.append({'pool':pool,'carry':carry,'reset':reset,'first_observation_all_arrays_identical':True,'observations':obs})
d={'status':'audited_development_first_divergence','development_report_sha256':digest('reports/fly-online-development-result-11.json'),'pairs':pairs,'source_sha256':digest(__file__),'new_neural_observations':0,'cloud_submissions':0,'interpretation':'Both first observations are identical across every recorded array. The second observations share image, time, feedback and pulse; differences follow the rate-reset boundary intervention. Earliest unequal 10 ms bins are not exact event times or proof of a mediating causal connection.'}
atomic_json('reports/fly-online-development-divergence-11.json',d)
for p in pairs:print(json.dumps({'pool':p['pool'],'second_observation':{k:v['first'] for k,v in p['observations'][1]['fields'].items()},'decoder':{k:{f:v[f] for f in ('side','difference_hz','gate_spikes')} for k,v in p['observations'][1]['decoder'].items()}}))
