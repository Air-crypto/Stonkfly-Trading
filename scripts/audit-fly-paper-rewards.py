"""Offline reconciliation of the pinned original paper-training reward histories."""
if not __debug__:
    raise RuntimeError('Run without -O: reconciliation assertions must remain enabled')
import json,sqlite3
from pathlib import Path
import numpy as np
from paperlab.core import Broker, digest, atomic_json
from paperlab.multi import DEX_COSTS,pool_tick
from paperlab.universe import Pool
from paperlab.fly_paper_memory import exposure
import argparse
parser=argparse.ArgumentParser(description='Reconcile the original two $250 paper sleeves and decompose delivered rewards. No model or cloud execution.')
parser.add_argument('--capture',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
if args.out.exists():raise ValueError('Refuse to overwrite a reward audit')
root=args.capture;capture=json.loads((root/'capture.json').read_text())
assert digest(root/'paper.db')==capture['ledger_sha256']
db=sqlite3.connect(f'file:{(root/"paper.db").resolve()}?mode=ro',uri=True)
assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
assert len(capture['cohort'])==2 and len(set(capture['cohort']))==2
ledger=list(db.execute("SELECT slot,payload FROM ledger WHERE name='fly' ORDER BY slot"))
report={'capture_sha256':digest(root/'capture.json'),'ledger_sha256':capture['ledger_sha256'],'source_sha256':digest(__file__),'pools':{}}
for key in capture['cohort']:
 pin=capture['pools'][key]
 assert exposure(db,key,pin['sleeve'],pin['assigned'])==pin['events']
 before=Broker(DEX_COSTS);anchor=250.;unpriced=False;pending=None;rows=[]
 for slot,raw in ledger:
  if not pin['events'][0]['slot']<=slot<=pin['events'][-1]['slot']:continue
  e=next(x for x in json.loads(raw)['sleeves'] if x['sleeve']==pin['sleeve']);assert e['pool']==key
  learning=e['detail'].get('learning_diagnostics')
  if learning:
   p=Pool(**e['observation']);t=pool_tick(p,p.observed)
   pre_equity=before.equity(t)
   actual=before.execute(pending['target'],pending['ts'],t) if pending else {'status':'hold'}
   assert actual==e['fill'],(slot,'fill')
   assert before.state()==e['broker'],(slot,'broker')
   equity=before.equity(t);assert abs(equity-e['equity'])<1e-8
   reward=learning['equity_reward_usd'];expected=0 if unpriced else equity-anchor
   assert abs(reward-expected)<1e-8,(slot,'reward')
   stimulus='reward' if reward>.01 else 'aversive' if reward<-.01 else 'none'
   assert e['detail']['stimulus']==stimulus
   rows.append({'slot':slot,'reward':reward,'stimulus':stimulus,'gap_reset':unpriced,
                'prior_inventory_revaluation':pre_equity-anchor,
                'execution_mark_effect':equity-pre_equity,'fill':actual,
                'current_side':e['detail']['side']})
   anchor=equity
  else:
   assert before.state()==e['broker'],(slot,'unexpected unavailable trade')
  unpriced=e['unpriced_inventory'];pending=e['decision']
 assert len(rows)==pin['observations']
 active=[x for x in rows if not x['gap_reset']]
 report['pools'][key]={'observations':len(rows),'gap_resets':sum(x['gap_reset'] for x in rows),
    'reward_sum':sum(x['reward'] for x in rows),
    'prior_inventory_revaluation_sum':sum(x['prior_inventory_revaluation'] for x in active),
    'execution_mark_effect_sum':sum(x['execution_mark_effect'] for x in active),
    'negative_rewards':sum(x['reward']<-.01 for x in rows),
    'execution_crossed_negative_deadband':sum(x['reward']<-.01 and x['prior_inventory_revaluation']>=-.01 for x in active),
    'reward_magnitude_quantiles':np.quantile([abs(x['reward']) for x in rows],[0,.25,.5,.75,1]).tolist(),
    'rows':rows}
 assert abs(report['pools'][key]['reward_sum']-sum(x['prior_inventory_revaluation']+x['execution_mark_effect'] for x in active))<1e-7
report['verification']={'paper_fills_and_states_replayed':True,'delivered_rewards_reconstructed':True,'gap_resets_excluded_from_decomposition':True,'neural_observations':0}
report['interpretation']='Recorded reward decomposition for the original cash-starting sleeves. Prior inventory revaluation plus execution mark effect equals delivered reward outside gap resets. Execution mark effect includes the modeled liquidation reserve, not only charged fees. These reward sums omit reset intervals and are not total account returns or an isolated causal effect of the learning rule.'
report['code_sha256']={name:digest(Path('paperlab')/name) for name in ('core.py','multi.py','fly.py','fly_paper_memory.py')}
db.close()
atomic_json(args.out,report)
for key,p in report['pools'].items():print(key,json.dumps({k:v for k,v in p.items() if k!='rows'},indent=2))
