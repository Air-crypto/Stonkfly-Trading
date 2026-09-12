"""Long-window input integrity, cost-aware selection and full native recording."""
import copy
from dataclasses import asdict, replace
import json
import os
from pathlib import Path

import numpy as np
import pytest

from paperlab.core import atomic_json, digest
from paperlab.fly_market_study import signature
from paperlab.fly_online_protocol import (ARMS, HOSTING_ALLOCATION, INFERENCE, PHASE_STEPS,
    RATIONALE, SELECTION, chunk_name, chunk_order, development_choice)
from paperlab.fly_paper_inputs import audit_news, news_stamps, seal, validate
from paperlab.fly_paper_price_audit import audit_prices
from paperlab.fly_paper_protocol import validate_registration
from paperlab.news import News
from paperlab.universe import Pool, Store


def documents():
    r=json.loads(Path('reports/fly-market-study-10-preregistration.json').read_text())
    a=json.loads(Path('reports/fly-paper-memory-audit-01.json').read_text())
    parent=json.loads(Path('reports/fly-market-study-10-plan.json').read_text())
    start=parent['plan']['registration']['end']+300
    r.update(schema=3,study='11',arms=copy.deepcopy(ARMS),inference_protocol=INFERENCE,
        selection_rule=SELECTION,rationale=RATIONALE,phase_steps=PHASE_STEPS,
        recorded_at=start-1,development_start=start,test_start=start+7200,end=start+14400,
        parent_plan_sha256=parent['sha256'],parent_report_sha256=digest('reports/fly-market-study-10.json'),
        mechanism_report_sha256=digest('reports/fly-credit-reset-study-01.json'),
        mechanism_audit_sha256=digest('reports/fly-credit-reset-audit-01.json'))
    return r,a


def test_protocol_is_two_independent_two_hour_phases_and_sixteen_complete_chunks():
    r,a=documents();validate_registration(r,a)
    assert r['test_start']-r['development_start']==r['end']-r['test_start']==7200
    chunks=chunk_order();assert len(chunks)==len(set(chunks))==16
    assert [c[0] for c in chunks]==['development']*8+['test']*8
    assert HOSTING_ALLOCATION==pytest.approx(1/9)
    for stage,pool,arm in chunks:assert chunk_name(stage,pool,arm)
    for wrong in (('test',True,'trained_frozen'),('test',2,'trained_frozen'),('training',0,'trained_frozen')):
        with pytest.raises(ValueError):chunk_name(*wrong)


@pytest.mark.parametrize('bad',['short','overlap','reset','freeze','selection','rationale','mechanism','late'])
def test_online_protocol_cannot_silently_shorten_or_change_intervention(bad):
    r,a=documents()
    if bad=='short':r['phase_steps']=3
    if bad=='overlap':r['test_start']-=300
    if bad=='reset':r['arms']['trained_online_reset_rates']['activity_reset']='full'
    if bad=='freeze':r['arms']['trained_online_reset_rates']['learning']=False
    if bad=='selection':r['selection_rule']='choose the best test arm'
    if bad=='rationale':r['rationale']='try many rates'
    if bad=='mechanism':r['mechanism_audit_sha256']='missing'
    if bad=='late':r['recorded_at']=r['development_start']
    with pytest.raises(ValueError):validate_registration(r,a)


def test_selection_requires_coverage_profit_after_hosting_and_every_control():
    e={arm:999. for arm in ARMS};n={arm:[24,18] for arm in ARMS};candidate='trained_online_reset_rates'
    for profit in (0,HOSTING_ALLOCATION,HOSTING_ALLOCATION+5e-10):
        e[candidate]=1000+profit;assert development_choice(e,n) is None
    e[candidate]=1001;assert development_choice(e,n)==candidate
    e['trained_online_carry']=1001;assert development_choice(e,n) is None
    e['trained_online_carry']=1000
    for arm in n:n[arm]=[24,17]
    assert development_choice(e,n) is None
    n['trained_frozen']=[24,24]
    with pytest.raises(ValueError,match='coverage'):development_choice(e,n)
    e[candidate]=float('nan')
    with pytest.raises(ValueError):development_choice(e,n)


@pytest.fixture
def sealed(tmp_path):
    r,a=documents();start=r['development_start'];store=Store(tmp_path/'universe.db')
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
        'reports/fly-paper-memory-audit-01.json','reports/fly-market-study-10-plan.json',tmp_path/'plan.json')
    return env,tmp_path


def test_long_seal_reconstructs_prices_news_and_all_terminal_slots(sealed):
    env,root=sealed;validate(env);audit_news(env,root/'news.db');result=audit_prices(env,root/'universe.db')
    r=env['plan']['registration'];assert len(news_stamps(r,env['plan']['series']))>40
    assert str(float(r['development_start']+6000)) in env['plan']['news_features']
    assert str(float(r['development_start']+13200)) in env['plan']['news_features']
    for pool in result['pools'].values():
        for phase in ('development','test'):
            assert len(pool['coverage'][phase])==25
            assert pool['coverage'][phase][-1]['terminal']
            assert not pool['coverage'][phase][-1]['neural_observation_expected']
    disappeared=result['pools'][r['cohort'][1]]['coverage']['test']
    assert all(not row['available'] for row in disappeared[-10:])
    assert all(t['bid']<10 for series in env['plan']['series'].values() for t in series)
    bad=copy.deepcopy(env)
    bad['plan']['news_features'].pop(str(float(r['development_start']+13200)))
    bad['sha256']=signature(bad['plan'])
    with pytest.raises(ValueError,match='timestamps'):validate(bad)
    bad=copy.deepcopy(env);key=r['cohort'][0]
    for side in ('bid','ask'):bad['plan']['series'][key][-1][side]+=.1
    bad['sha256']=signature(bad['plan'])
    with pytest.raises(ValueError,match='history'):audit_prices(bad,root/'universe.db')


def test_native_end_recording_guards_and_selection_precede_loading_graph(sealed,tmp_path,monkeypatch):
    from paperlab.fly_online_study import run_chunk
    from paperlab.fly_market_study import phase
    monkeypatch.setattr('paperlab.fly_online_study.TraceLab',lambda *a:pytest.fail('must validate before graph loading'))
    env,root=sealed
    with pytest.raises(ValueError,match='eight development'):
        run_chunk(env,'memory',root/'news.db','graph',tmp_path/'bad',stage='test',pool_index=0,arm='trained_frozen')
    assert not (tmp_path/'bad').exists()
    with pytest.raises(ValueError,match='full traces'):
        phase(None,[],0,24,{},True,0,record_end_state=True)
    with pytest.raises(ValueError,match='only carry'):
        phase(None,[],0,24,{'activity_reset':'full'},True,0,activity_output=tmp_path/'bad')


def test_twenty_fourth_boundary_preserves_memory_and_other_activity(tmp_path):
    from test_fly_market_activity import Brain
    from paperlab.fly_market_activity import apply_boundary
    from paperlab.fly_market_activity_audit import audit_boundary
    from paperlab.fly_market_study import learned_state
    b=Brain()
    for field,size in (('rate_kc',2),('rate_dan',1)):
        b.initial[field]=np.zeros(size);setattr(b,field,np.full(size,7.))
    memory=learned_state(b);initial={k:v for k,v in b.initial.items() if k not in ('memory_u','memory_w')}
    path=tmp_path/'boundary.npz';metadata=apply_boundary(b,'reset_rates',24,path)
    result=audit_boundary(path,metadata,initial,memory,'reset_rates',24)
    assert result['changed_fields']==['rate_dan','rate_kc']
    assert b.sim_ms==500 and np.all(b.v==2)
    for value in (0,25,True,2.0):
        with pytest.raises(ValueError):apply_boundary(b,'reset_rates',value,path)


def test_selection_needs_all_development_chunks_and_rejects_wrong_phase(sealed):
    from paperlab.fly_online_study import select_development
    env,_=sealed;r=env['plan']['registration'];chunks={}
    for stage,pool,arm in chunk_order()[:8]:
        name=chunk_name(stage,pool,arm)
        chunks[name]={'status':'paper_online_chunk_completed','chunk':name,'plan_sha256':env['sha256'],
            'registration':r,'stage':stage,'pool_index':pool,'arm':arm,
            'outcome':{'equity':251 if arm=='trained_online_reset_rates' else 249,
                       'rows':[{'event':{'synthetic_selection_fixture':True}}]*24+[{'event':None}]}}
    selection=select_development(env,chunks)
    assert selection['selected']=='trained_online_reset_rates'
    assert selection['development_equity']['trained_online_reset_rates']==1002
    assert len(selection['development_chunk_sha256'])==8
    bad=copy.deepcopy(chunks);bad.pop(next(iter(bad)))
    with pytest.raises(ValueError,match='eight development'):select_development(env,bad)
    bad=copy.deepcopy(chunks);next(iter(bad.values()))['stage']='test'
    with pytest.raises(ValueError,match='identity'):select_development(env,bad)


@pytest.mark.skipif(not os.environ.get('FLY_TRACE_DATA'),reason='requires the full prepared native graph')
def test_native_long_online_chunk_reconstructs_feedback_boundaries_and_every_bin(tmp_path):
    from test_fly_paper_study import native_fixture
    from paperlab.fly_paper_inputs import SnapshotNews
    from paperlab.fly_online_study import run_chunk
    from paperlab.fly_online_audit import audit_chunk
    env,package,news=native_fixture(tmp_path,study='10')
    r=env['plan']['registration'];template,_=documents()
    r.update({k:v for k,v in template.items() if k not in ('source_memories','training_audit_sha256')})
    start=r['development_start'];series={}
    from paperlab.core import Tick
    for i,key in enumerate(r['cohort']):
        ticks=[]
        for minute in range(-200,241):
            ts=start+minute*60;price=1.2+.15*np.sin(minute/10)
            ticks.append(Tick(ts,price*.995,price*1.005,1000,key,'synthetic',ts,
                              not (i==0 and 35<=minute<=45)))
        series[key]=[asdict(t) for t in ticks]
    env['plan'].update(series=series,snapshot_end=r['end'])
    reader=SnapshotNews(news)
    try:env['plan']['news_features']={str(float(t)):reader.features(t).tolist() for t in news_stamps(r,series)}
    finally:reader.db.close()
    env['sha256']=signature(env['plan']);validate(env)
    root=tmp_path/'online';data=Path(os.environ['FLY_TRACE_DATA'])
    result=run_chunk(env,package,news,data,root,stage='development',pool_index=0,arm='trained_online_reset_rates')
    checked,view=audit_chunk(env,result,root,data)
    assert checked['verification']['decision_slots']==25
    assert checked['verification']['observations']==21
    assert checked['verification']['native_bins']==1050
    events=[row for row in result['outcome']['rows'] if row['event'] is not None]
    assert events[0]['event']['stimulus']=='none'
    assert next(row for row in events if row['decision_ts']==start+50*60)['event']['equity_reward_usd']==0
    assert any(row['event']['equity_reward_usd']!=0 for row in events)
    assert any(row['event']['diagnostics']['changed_edges']>0 for row in events)
    assert view['frames'][-1]['plastic_credit']['learning']
    atomic_json(tmp_path/'audit.json',checked);atomic_json(tmp_path/'view.json',view)
    for bad in ('feedback','fill','equity','terminal','brain_clock'):
        changed=copy.deepcopy(result);row=next(row for row in changed['outcome']['rows'][1:] if row['event'])
        if bad=='feedback':row['event']['equity_reward_usd']+=10
        if bad=='fill':row['fill']['status']='forged'
        if bad=='equity':row['equity']+=10
        if bad=='terminal':changed['outcome']['rows'][-1]['terminal']=False
        if bad=='brain_clock':row['event']['brain_ms']+=500
        with pytest.raises(ValueError):audit_chunk(env,changed,root,data)
    late=copy.deepcopy(result);late['outcome']['rows'].pop(-2)
    with pytest.raises(ValueError,match='timeline'):audit_chunk(env,late,root,data)
    # Rehash a corrupted late native update: the independent rule must catch it.
    file=root/'trace/step-21.npz'
    with np.load(file) as a:arrays={k:a[k].copy() for k in a.files}
    original=file.read_bytes();arrays['u'][-1,0]+=.1;np.savez_compressed(file,**arrays)
    forged=copy.deepcopy(result);forged['artifact_sha256']['trace/step-21.npz']=digest(file)
    try:
        with pytest.raises(ValueError,match='rule state'):audit_chunk(env,forged,root,data)
    finally:file.write_bytes(original)
