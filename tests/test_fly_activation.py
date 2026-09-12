import copy
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pytest

from paperlab.core import atomic_json,digest
from paperlab.fly_paper_protocol import development_choice,validate_registration


def documents():
    return json.loads(Path('reports/fly-market-study-10-preregistration.json').read_text()),json.loads(Path('reports/fly-paper-memory-audit-01.json').read_text())


def test_activation_registration_pins_prior_mechanism_and_future_conditions():
    r,a=documents();validate_registration(r,a)
    assert r['mechanism_report_sha256']==digest('reports/fly-paper-stimulation-study-01.json')
    assert r['mechanism_audit_sha256']==digest('reports/fly-paper-stimulation-audit-01.json')
    assert r['parent_report_sha256']==digest('reports/fly-market-study-09.json')
    assert r['parent_plan_sha256']==json.loads(Path('reports/fly-market-study-09-plan.json').read_text())['sha256']
    assert r['recorded_at']<r['development_start']<r['test_start']<r['end']
    assert {a['recipient_current'] for a in r['arms'].values()}=={0,10}
    for value in (5,20,float('nan')):
        changed=copy.deepcopy(r);changed['arms']['trained_stimulated']['recipient_current']=value
        with pytest.raises(ValueError):validate_registration(changed,a)


def test_activation_selection_demands_advantage_over_both_controls_and_cash():
    e={'pristine_frozen':1001,'pristine_stimulated':1003,'trained_frozen':1002,'trained_stimulated':1003}
    assert development_choice(e) is None
    e['trained_stimulated']=1004;assert development_choice(e)=='trained_stimulated'
    e['trained_frozen']=1004;assert development_choice(e)=='trained_frozen'
    e={k:999 for k in e};assert development_choice(e) is None


def schedule_fixture(tmp_path):
    from paperlab.fly_paper_schedule import execute_due
    r,a=documents();state=tmp_path/'state';spec=tmp_path/'spec';memory=tmp_path/'memory';discovery=tmp_path/'discovery'
    for d in (state,spec,memory,discovery):d.mkdir()
    for name in ('fly-market-study-10-preregistration.json','fly-market-study-09.json',
                 'fly-paper-stimulation-study-01.json','fly-paper-stimulation-audit-01.json'):
        shutil.copyfile(Path('reports')/name,spec/name)
    shutil.copyfile('reports/fly-paper-memory-audit-01.json',memory/'audit.json')
    events=[]
    kwargs=dict(study='10',call_id='fc-arm-fixture',input_id='in-arm-fixture',commit=lambda:events.append('commit'),
                run=lambda *a:pytest.fail('No model before the future endpoint'),settle_budget=lambda:pytest.fail('No extra compute reservation'))
    return execute_due,(state,discovery,spec,memory),kwargs,r,events


def test_schedule_witnesses_registration_before_data_window_and_rejects_changes(tmp_path):
    due,args,kw,r,events=schedule_fixture(tmp_path)
    assert due(*args,**kw,now=r['recorded_at']+1) is None
    root=args[0]/'registered-paper-10';armed=json.loads((root/'armed.json').read_text())
    assert armed['armed_at']<r['development_start'] and len(events)==1
    assert due(*args,**kw,now=r['development_start']+1) is None
    assert len(events)==1
    path=args[2]/'fly-market-study-10-preregistration.json';changed=json.loads(path.read_text());changed['hypothesis']+=' modified after arming';atomic_json(path,changed)
    with pytest.raises(ValueError,match='registration differs'):due(*args,**kw,now=r['end'])


def test_late_first_registration_is_closed_without_model_compute(tmp_path):
    due,args,kw,r,events=schedule_fixture(tmp_path)
    assert due(*args,**kw,now=r['development_start']) is None
    root=args[0]/'registered-paper-10';assert not (root/'armed.json').exists()
    assert json.loads((root/'cloud-call.json').read_text())['status']=='registration_missed'
    assert due(*args,**kw,now=r['end']) is None and len(events)==1


@pytest.mark.skipif(not os.environ.get('FLY_TRACE_DATA'),reason='requires prepared full native graph')
def test_native_future_activation_records_current_and_replays_costed_ledger(tmp_path):
    from test_fly_paper_study import native_fixture
    from paperlab.fly_paper_study import run
    from paperlab.fly_paper_audit import audit
    env,memory,news=native_fixture(tmp_path,study='10');root=tmp_path/'run'
    summary=run(env,memory,news,Path(os.environ['FLY_TRACE_DATA']),root)
    report,verified=audit(env,summary,root)
    assert all(report['verification'].values())
    count=sum(v['recipient_current_observations_verified'] for arms in verified['pools'].values() for phases in arms.values() for v in phases.values())
    assert count==32 and len(list(root.glob('pool*/current-*.npz')))==32
    # Corrupt the actual stimulus and update its checksum; the auditor must still
    # reject it against registered native target/current/time semantics.
    file=root/'pool0-trained_stimulated/current-01.npz'
    with np.load(file,allow_pickle=False) as saved:arrays={k:saved[k].copy() for k in saved.files}
    arrays['currents'][17,0]=5;np.savez_compressed(file,**arrays)
    raw=json.loads((root/'results.json').read_text());key=env['plan']['registration']['cohort'][0]
    raw[key]['trained_stimulated']['test']['rows'][0]['event']['stimulation']['artifact_sha256']=digest(file)
    atomic_json(root/'results.json',raw)
    with pytest.raises(ValueError,match='Native recipient targets, current or timing'):
        audit(env,summary,root)


def test_activation_sealer_accepts_pinned_paper_parent_and_timestamped_news(tmp_path):
    from test_fly_paper_study import make_news
    from paperlab.universe import Pool,Store
    from paperlab.fly_paper_inputs import seal,audit_news
    r,_=documents();start=r['development_start'];archive=tmp_path/'prices.db';store=Store(archive)
    for i in range(240):
        ts=start-120*60+i*60
        store.add([Pool(key,key.split(':')[0],key.split(':')[1],f'fixture{k}',f'TEST{k}',
                       1,100000,10000,20,20,ts-3600,ts,source='synthetic') for k,key in enumerate(r['cohort'])])
    store.db.close();news=tmp_path/'news.db';make_news(news,start)
    env=seal(archive,news,'reports/fly-market-study-10-preregistration.json','reports/fly-paper-memory-audit-01.json',
             'reports/fly-market-study-09-plan.json',tmp_path/'plan.json')
    audit_news(env,news)
    assert env['plan']['registration']==r
    assert all(t['ts']<=r['end'] for ticks in env['plan']['series'].values() for t in ticks)


def test_armed_activation_claims_once_at_endpoint(monkeypatch,tmp_path):
    import sqlite3
    due,args,kw,r,events=schedule_fixture(tmp_path)
    assert due(*args,**kw,now=r['recorded_at']+1) is None
    state,discovery,_,_=args;root=state/'registered-paper-10'
    (state/'meme-pools-v1').mkdir();(state/'meme-pools-v1/news.db').write_bytes(b'synthetic fixture; sealer mocked')
    with sqlite3.connect(discovery/'universe-snapshot.db') as db:
        db.execute('CREATE TABLE observations(payload TEXT)');db.execute('INSERT INTO observations VALUES(?)',(json.dumps({'observed':r['end']}),))
    def sealed(*args):
        assert json.loads((root/'cloud-call.json').read_text())['status']=='claimed'
        assert args[-2].name=='fly-market-study-09-plan.json'
        env={'plan':{'fixture':True},'sha256':'a'*64};atomic_json(args[-1],env);return env
    monkeypatch.setattr('paperlab.fly_paper_schedule.seal',sealed)
    calls=[];settled=[]
    def run(*args):
        assert json.loads((root/'cloud-call.json').read_text())['status']=='pending'
        calls.append(args);return {'status':'paper_checkpoint_study_completed'}
    kw.update(run=run,settle_budget=lambda:(settled.append(True) or {}))
    assert due(*args,**kw,now=r['end'])['status']=='paper_checkpoint_study_completed'
    assert due(*args,**kw,now=r['end']+300) is None
    assert len(calls)==len(settled)==1
    assert json.loads((root/'cloud-call.json').read_text())['run_id'].startswith('assay-paper-10-')
