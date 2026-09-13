"""Synthetic price/news/ledger and recorder wiring tests; no native construction."""
import copy
from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

from paperlab.core import Broker,Tick,atomic_json,digest
from paperlab.fly_market_study import learned_state,quote_at,signature
from paperlab.fly_rate_inputs import REFERENCE_FILES,audit_news,news_stamps,seal,validate,verify_reference
from paperlab.fly_rate_metrics import replay_ledger,trading_metrics
from paperlab.fly_rate_price_audit import audit_prices
from paperlab.fly_rate_protocol import (ARMS,CANDIDATE,COHORT_POLICY,HYPOTHESIS,INFERENCE,
    SELECTION,EXECUTION,chunk_name,chunk_order)
from paperlab.fly_paper_protocol import MEMORY_FIELDS,NEWS
from paperlab.multi import DEX_COSTS
from paperlab.news import News
from paperlab.universe import Pool,Store


def documents(root):
    reference=root/'reference';reference.mkdir()
    with zipfile.ZipFile('reports/fly-rate-checkpoint-12.zip') as z:z.extractall(reference)
    shutil.copyfile('reports/fly-online-result-11.json',reference/'parent-report.json')
    shutil.copyfile('reports/fly-learning-scale-audit-01.json',reference/'mechanism-audit.json')
    a=json.loads((reference/'audit.json').read_text());cap=json.loads((reference/'capture.json').read_text())
    parent=json.loads((reference/'parent-report.json').read_text())
    start=(int(a['captured_at']//300)+2)*300
    r={'schema':4,'kind':'paper_checkpoint_comparison','study':'12','arms':copy.deepcopy(ARMS),
       'phase_steps':24,'decision_seconds':300,'costs':asdict(DEX_COSTS),
       'hypothesis':HYPOTHESIS,'inference_protocol':INFERENCE,'selection_rule':SELECTION,
       'news_protocol':NEWS,'cohort_policy':COHORT_POLICY,'python_hash_seed':'0','execution_protocol':EXECUTION,
       'recorded_at':start-1,'development_start':start,'test_start':start+7200,'end':start+14400,
       'training_cutoff':max((p['last_slot']+1)*300 for p in a['pools'].values()),
       'checkpoint_capture_at':a['captured_at'],'parent_end':parent['registration']['end'],
       'cohort':cap['cohort'],'source_memories':{k:{f:v[f] for f in MEMORY_FIELDS} for k,v in a['pools'].items()}}
    r.update({field:digest(reference/name) for name,field in REFERENCE_FILES.items()})
    return r,a,reference


@pytest.fixture
def sealed(tmp_path):
    r,a,reference=documents(tmp_path);start=r['development_start'];store=Store(tmp_path/'universe.db')
    for minute in range(-300,246):
        ts=start+minute*60
        for i,key in enumerate(r['cohort']):
            if i==1 and minute>=175:continue
            if i==0 and 18<=minute<=27:continue
            p=Pool(key,'solana',key.split(':')[1],f'token{i}',f'FIXTURE{i}',
                1+minute/1000 if ts<=r['end'] else 99999,100000,0 if minute==40 else 10000,
                20,20,start-86400,ts,source='synthetic')
            store.add([p]);store.add([replace(p,observed=ts+1,price=123456)])
    store.db.close();news=News(tmp_path/'news.db')
    for label,published,seen,encoded in [('early',start-60,start-60,start-60),
        ('dev',start-60,start+6000,start+6000),('test',start-60,start-60,start+13200),
        ('future',r['end']+1,start-60,start-60)]:
        news.add('https://example.invalid/'+label,'growth rally '+label,'fixture',published,seen,encoded=encoded)
    news.db.close();atomic_json(tmp_path/'registration.json',r)
    env=seal(tmp_path/'universe.db',tmp_path/'news.db',tmp_path/'registration.json',
             reference/'audit.json',reference,tmp_path/'plan.json')
    return env,tmp_path,reference


def test_seal_keeps_missing_pools_and_all_price_news_terminal_slots(sealed):
    env,root,ref=sealed;validate(env);news=audit_news(env,root/'news.db')
    prices=audit_prices(env,root/'universe.db');r=env['plan']['registration']
    assert news['vectors_reconstructed']>40 and verify_reference(r,env['plan']['training_audit'],ref)
    for p in prices['pools'].values():
        for stage in ('development','test'):
            assert len(p['coverage'][stage])==25
            assert p['coverage'][stage][-1]['terminal'] and not p['coverage'][stage][-1]['neural_observation_expected']
    assert all(not x['available'] for x in prices['pools'][r['cohort'][1]]['coverage']['test'][-10:])
    assert all(t['bid']<10 for series in env['plan']['series'].values() for t in series)
    with pytest.raises(ValueError,match='overwrite'):
        seal(root/'universe.db',root/'news.db',root/'registration.json',ref/'audit.json',ref,root/'plan.json')


@pytest.mark.parametrize('kind',['price','news','future','cohort','reference'])
def test_rehashed_bad_inputs_cannot_pass_independent_checks(sealed,kind):
    env,root,_=sealed;bad=copy.deepcopy(env);p=bad['plan'];key=p['registration']['cohort'][0]
    if kind=='price':p['series'][key][-1]['bid']+=.1
    if kind=='news':next(iter(p['news_features'].values()))[0]+=.1
    if kind=='future':p['series'][key][-1]['ts']=p['registration']['end']+1
    if kind=='cohort':p['series'].pop(key)
    if kind=='reference':p['reference_sha256']['selection.json']='f'*64
    bad['sha256']=signature(p)
    with pytest.raises(ValueError):
        validate(bad);audit_news(bad,root/'news.db');audit_prices(bad,root/'universe.db')


@pytest.mark.parametrize('name',list(REFERENCE_FILES)+['pool0-memory.npz'])
def test_changed_parent_or_checkpoint_cannot_be_sealed(tmp_path,name):
    r,a,ref=documents(tmp_path);(ref/name).write_bytes(b'changed')
    with pytest.raises(ValueError):verify_reference(r,a,ref)


def test_news_requires_seed_before_process_start_even_if_env_is_changed_late():
    code="import os;os.environ['PYTHONHASHSEED']='0';from paperlab.fly_rate_inputs import require_seed;require_seed()"
    child=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,
                         env={**os.environ,'PYTHONHASHSEED':'1'})
    assert child.returncode!=0 and 'fresh process' in child.stderr


def ledger(env,pool,arm,*,stage='development',hold_after_buy=True):
    r=env['plan']['registration'];ticks=[Tick(**t) for t in env['plan']['series'][r['cohort'][pool]]]
    broker=Broker(DEX_COSTS);pending=None;anchor=250.;last=0;unpriced=False;rows=[];n=0
    for slot in range(25):
        stamp=r[stage+'_start']+slot*300;_,t=quote_at(ticks,stamp)
        fill=broker.execute(*pending,t) if pending else {'status':'hold'};pending=None;equity=broker.equity(t);event=None
        if slot<24 and t.available and t.ts>last:
            side='BUY' if n==0 else 'HOLD' if hold_after_buy else 'SELL' if n==1 else 'HOLD'
            reward=(0 if unpriced else equity-anchor) if ARMS[arm]['learning'] else 0
            event={'synthetic_account_fixture':True,'side':side,'market_decision_ts':stamp,
                   'equity_reward_usd':reward,'plasticity_enabled':ARMS[arm]['learning'],
                   'stimulus':'reward' if reward>.01 else 'aversive' if reward<-.01 else 'none'}
            if side!='HOLD':pending=(.5 if side=='BUY' else 0,stamp)
            n+=1;last=t.ts;anchor=equity
        rows.append({'decision_ts':stamp,'quote_ts':t.ts,'available':t.available,'terminal':slot==24,
                     'fill':fill,'equity':equity,'broker':broker.state(),'event':event})
        unpriced=not t.available
    metrics=trading_metrics(rows)
    return {'rows':rows,'equity':rows[-1]['equity'],'fills':metrics['fills'],'fees':metrics['fees_usd'],
            'unavailable_marks':metrics['unavailable_marks']}


def development(env):
    r=env['plan']['registration'];chunks={}
    for stage,pool,arm in chunk_order()[:8]:
        name=chunk_name(stage,pool,arm);outcome=ledger(env,pool,arm,hold_after_buy=arm==CANDIDATE)
        chunks[name]={'status':'paper_rate_chunk_completed','chunk':name,'stage':stage,'pool_index':pool,
            'pool':r['cohort'][pool],'arm':arm,'registration':r,'plan_sha256':env['sha256'],
            'outcome':outcome,'trading_metrics':trading_metrics(outcome['rows'])}
    return chunks


def test_selection_replays_all_accounts_before_using_equity(sealed):
    from paperlab.fly_rate_study import select_development
    env,_,_=sealed;chunks=development(env);selected=select_development(env,chunks)
    assert selected['ledgers_replayed_before_selection'] and selected['selected']==CANDIDATE
    assert len(selected['development_chunk_sha256'])==8
    metrics=next(iter(chunks.values()))['trading_metrics']
    assert metrics['fills']==2 and metrics['buy_notional_usd']==pytest.approx(25)
    assert metrics['turnover_usd']>49 and metrics['fees_usd']>.6
    with pytest.raises(ValueError,match='eight development'):select_development(env,{})


@pytest.mark.parametrize('kind',['equity','fill','reward','late_row','metrics','pool','terminal','nan_outcome','nan_reward'])
def test_selection_rejects_rehashed_forged_ledger_details(sealed,kind):
    from paperlab.fly_rate_study import select_development
    env,_,_=sealed;chunks=development(env)
    changed=chunks[chunk_name('development',0,CANDIDATE)];rows=changed['outcome']['rows']
    if kind=='equity':changed['outcome']['equity']+=100
    if kind=='fill':rows[1]['fill']['fee']='0'
    if kind=='reward':rows[1]['event']['equity_reward_usd']+=1
    if kind=='late_row':rows[-2]['quote_ts']+=1
    if kind=='metrics':changed['trading_metrics']['turnover_usd']=0
    if kind=='pool':changed['pool']='solana:other'
    if kind=='terminal':rows[-1]['terminal']=False
    if kind=='nan_outcome':changed['outcome']['equity']=float('nan')
    if kind=='nan_reward':rows[1]['event']['equity_reward_usd']=float('nan')
    with pytest.raises(ValueError):select_development(env,chunks)


def test_unpriced_inventory_drawdown_is_explicit_and_feedback_does_not_bridge_gap(sealed):
    env,_,_=sealed;r=env['plan']['registration'];outcome=ledger(env,1,CANDIDATE,stage='test')
    checked=replay_ledger(outcome,env['plan']['series'][r['cohort'][1]],r['test_start'],ARMS[CANDIDATE])
    assert checked['metrics']['unavailable_marks']>=10
    assert 'not an observed market crash' in checked['metrics']['marking_note']
    assert checked['metrics']['maximum_marked_drawdown_pct']>0


def test_missing_development_rejected_before_native_construction(sealed,monkeypatch):
    from paperlab.fly_rate_study import run_chunk
    env,root,ref=sealed
    monkeypatch.setattr('paperlab.fly_rate_study.TraceLab',lambda *a:pytest.fail('native constructor forbidden'))
    with pytest.raises(ValueError,match='eight development'):
        run_chunk(env,ref,root/'news.db','unused',root/'bad',stage='test',pool_index=0,arm=CANDIDATE)
    assert not (root/'bad').exists()


@pytest.mark.parametrize('arm',list(ARMS))
def test_recorder_passes_declared_eta_through_real_phase_with_no_observations(sealed,monkeypatch,arm):
    from paperlab.fly_rate_study import run_chunk
    env,root,ref=sealed;env=copy.deepcopy(env);p=env['plan'];key=p['registration']['cohort'][0]
    for tick in p['series'][key]:
        if tick['ts']>=p['registration']['development_start']:tick['available']=False
    env['sha256']=signature(p)
    @dataclass
    class Settings: learning:bool=False
    class Brain:
        def __init__(self):
            self.eta=.001;self.ids=np.array([1,2,3]);self.n=3;self.post=np.array([1,2,0])
            self.weight=np.array([1.,2.,3.],dtype=np.float32);self.baseline_plastic=self.weight[[0,2]].copy()
            self.memory_u=np.zeros(2);self.memory_w=np.zeros(2);self.initial={'v':np.zeros(3)};self.v=np.zeros(3)
            self.circuit={'edges':np.array([0,2]),'pre':np.array([0,1]),'dan':np.array([2]),'gain':np.array([[.02,.02]])}
            self.build={'model':'synthetic non-native recorder fixture'}
        def reset(self):self.v[:]=0
    brain=Brain();lab=SimpleNamespace(brain=brain,fly=SimpleNamespace(controller=SimpleNamespace(s=Settings())))
    monkeypatch.setattr('paperlab.fly_rate_study.TraceLab',lambda *a:lab)
    def imported(b,p,r):
        assert b.eta==.001
        return {key:learned_state(b) for key in p['registration']['cohort']}
    monkeypatch.setattr('paperlab.fly_rate_study.imported_memory',imported)
    result=run_chunk(env,ref,root/'news.db','unused',root/arm,stage='development',pool_index=0,arm=arm)
    assert brain.eta==ARMS[arm]['eta'] and brain.weights_frozen is not ARMS[arm]['learning']
    assert result['graph']['neurons']==3 and result['status']=='paper_rate_chunk_completed'
    assert len(result['outcome']['rows'])==25 and all(r['event'] is None for r in result['outcome']['rows'])
    assert result['outcome']['equity']==250 and result['trading_metrics']['turnover_usd']==0
    assert not (root/arm/'trace/view.json').exists()
