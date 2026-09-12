"""Dispatch durability and actual cloud-entrypoint budget/cycle ordering."""
import copy
import json
from pathlib import Path
import shutil
import sqlite3
import time

import pytest

from paperlab.core import atomic_json, digest
from paperlab.fly_market_study import signature
from paperlab.fly_online_protocol import chunk_name, chunk_order
from paperlab.fly_online_schedule import DIRECTORY, REGISTRATION, EXECUTION, execute_due
from paperlab.fly_online_study import SOURCE_FILES
from test_fly_online import documents


@pytest.fixture
def setup(tmp_path,monkeypatch):
    from types import SimpleNamespace
    r,a=documents();r['execution_protocol']=EXECUTION
    spec=tmp_path/'spec';spec.mkdir();memory=tmp_path/'memory';memory.mkdir()
    state=tmp_path/'state';discovery=tmp_path/'discovery';discovery.mkdir();root=state/DIRECTORY
    for name in ('fly-market-study-10-plan.json','fly-market-study-10.json','fly-credit-reset-study-01.json','fly-credit-reset-audit-01.json'):
        shutil.copyfile(Path('reports')/name,spec/name)
    for i,key in enumerate(r['cohort']):
        file=memory/f'pool{i}-memory.npz';file.write_bytes(b'pinned test fixture')
        a['pools'][key]['memory_file_sha256']=digest(file);r['source_memories'][key]['memory_file_sha256']=digest(file)
    atomic_json(memory/'audit.json',a);r['training_audit_sha256']=digest(memory/'audit.json')
    atomic_json(spec/REGISTRATION,r)
    (state/'meme-pools-v1').mkdir(parents=True);(state/'meme-pools-v1/news.db').write_bytes(b'news fixture')
    with sqlite3.connect(discovery/'universe-snapshot.db') as db:
        db.execute('CREATE TABLE observations(payload TEXT)');db.execute('INSERT INTO observations VALUES (?)',(json.dumps({'observed':r['end']}),))
    sources={f'paperlab/{f}':'a'*64 for f in SOURCE_FILES};sources['cloud.py']='b'*64
    monkeypatch.setattr('paperlab.fly_online_schedule.source_hashes',lambda *a:copy.deepcopy(sources))
    events=[];calls=[];env={'plan':{'registration':r},'sha256':'c'*64}
    # Actual prospective registration/memory checks stay active. Input sealing
    # and propagation are mocked here; their real implementation has native tests.
    def seal(*args):
        assert json.loads((root/'sealing.json').read_text())['status']=='claimed'
        assert 'commit' in events
        atomic_json(args[-1],env);return env
    original_validate=__import__('paperlab.fly_online_schedule',fromlist=['validate']).validate
    monkeypatch.setattr('paperlab.fly_online_schedule.seal',seal)
    monkeypatch.setattr('paperlab.fly_online_schedule.validate',lambda x:x['plan'] if x==env else original_validate(x))
    monkeypatch.setattr('paperlab.fly_online_study.validate',lambda x:x['plan'])
    def run(envelope,mem,news,data,out,**kw):
        name=chunk_name(kw['stage'],kw['pool_index'],kw['arm'])
        receipt=json.loads((root/'chunks'/name/'receipt.json').read_text())
        assert receipt['status']=='claimed' and receipt['call_id']==kw_current['call_id']
        assert events[-1]=='commit'
        assert 0<kw['seconds']<=480
        if kw['stage']=='test':
            assert len(kw['development'])==8
            assert (root/'selection-receipt.json').exists()
        else:assert kw['development'] is None
        calls.append(name);events.append('run:'+name);out.mkdir(parents=True)
        result={'status':'paper_online_chunk_completed','chunk':name,'stage':kw['stage'],
            'pool_index':kw['pool_index'],'arm':kw['arm'],'pool':r['cohort'][kw['pool_index']],
            'registration':r,'plan_sha256':env['sha256'],'code_sha256':{f:sources['paperlab/'+f] for f in SOURCE_FILES},
            'outcome':{'equity':251 if kw['arm']=='trained_online_reset_rates' else 249,
                'rows':[{'event':{'synthetic_dispatch_fixture':True}}]*24+[{'event':None}]}}
        from paperlab.fly_online_study import select_development
        result['selection']=select_development(envelope,kw['development']) if kw['stage']=='test' else None
        atomic_json(out/'summary.json',result);return result
    kw_current={'call_id':'fc-armed','input_id':'in-armed','commit':lambda:events.append('commit'),
        'run':run,'deadline':time.monotonic()+570,'now':r['development_start']-.5}
    return SimpleNamespace(args=(state,discovery,spec,memory),kw=kw_current,r=r,root=root,
        events=events,calls=calls,sources=sources,env=env)


def arm(s):
    result=execute_due(*s.args,**s.kw)
    assert result['status']=='paper_online_collecting' and not s.calls
    s.kw['now']=s.r['end']+1;s.kw['call_id']='fc-next';s.kw['input_id']='in-next'
    return result


def test_dispatch_waits_then_runs_one_complete_chunk_per_call_and_selects_before_test(setup):
    s=setup;arm(s)
    for i,expected in enumerate(chunk_order()):
        s.kw.update(call_id=f'fc-{i}',input_id=f'in-{i}',deadline=time.monotonic()+570)
        result=execute_due(*s.args,**s.kw)
        assert result['status']=='paper_online_chunk_completed'
        assert len(s.calls)==i+1 and s.calls[-1]==chunk_name(*expected)
    result=execute_due(*s.args,**s.kw)
    assert result['status']=='paper_online_study_completed' and len(s.calls)==16
    summary=json.loads((s.root/'summary.json').read_text())
    assert summary['selection']['selected']=='trained_online_reset_rates' and summary['audited'] is False
    assert summary['total_equity']['trained_online_reset_rates']=={'development':1002,'test':1002}
    # Historical completion survives later app source changes without re-execution.
    s.sources['cloud.py']='f'*64
    assert execute_due(*s.args,**s.kw)['status']=='paper_online_study_completed'
    assert len(s.calls)==16


@pytest.mark.parametrize('failure',[RuntimeError,KeyboardInterrupt])
def test_failed_or_uncertain_chunk_never_restarts_or_advances(setup,failure):
    s=setup;arm(s);attempted=[]
    def fail(*a,**kw):attempted.append(1);raise failure('controlled interruption')
    s.kw['run']=fail
    with pytest.raises(failure):execute_due(*s.args,**s.kw)
    s.kw.update(call_id='fc-later',input_id='in-later')
    result=execute_due(*s.args,**s.kw)
    assert result['status']=='paper_online_chunk_unresolved' and len(attempted)==1
    assert result['receipt']['status']==('failed' if failure is RuntimeError else 'claimed')


def test_sealing_interruption_is_not_retried(setup,monkeypatch):
    s=setup;arm(s);calls=[]
    def fail(*a):calls.append(1);raise KeyboardInterrupt()
    monkeypatch.setattr('paperlab.fly_online_schedule.seal',fail)
    with pytest.raises(KeyboardInterrupt):execute_due(*s.args,**s.kw)
    assert execute_due(*s.args,**s.kw)['status']=='paper_online_seal_unresolved'
    assert len(calls)==1 and not s.calls


@pytest.mark.parametrize('reason',['late','source','registration','memory','evidence','short_time','slow_cycle','no_snapshot','no_news'])
def test_registration_and_resource_guards_prevent_compute(setup,reason):
    s=setup
    if reason=='late':
        s.kw['now']=s.r['development_start']
        assert execute_due(*s.args,**s.kw)['status']=='paper_online_registration_missed'
        s.kw['now']=s.r['end']
        assert execute_due(*s.args,**s.kw)['status']=='paper_online_registration_missed'
        assert not s.calls;return
    arm(s)
    if reason=='source':s.sources['cloud.py']='e'*64
    if reason=='registration':
        changed=copy.deepcopy(s.r);changed['recorded_at']-=1;atomic_json(s.args[2]/REGISTRATION,changed)
    if reason=='memory':(s.args[3]/'pool0-memory.npz').write_bytes(b'different')
    if reason=='evidence':(s.args[2]/'fly-credit-reset-study-01.json').write_text('{}')
    if reason in ('source','registration','memory','evidence'):
        with pytest.raises(ValueError):execute_due(*s.args,**s.kw)
    else:
        if reason=='short_time':s.kw['deadline']=time.monotonic()+100
        if reason=='slow_cycle':s.kw['allow_compute']=False
        if reason=='no_snapshot':
            with sqlite3.connect(s.args[1]/'universe-snapshot.db') as db:db.execute('DELETE FROM observations')
        if reason=='no_news':(s.args[0]/'meme-pools-v1/news.db').unlink()
        assert execute_due(*s.args,**s.kw)['status'].startswith('paper_online_waiting')
    assert not s.calls and not (s.root/'chunks').exists()


def test_selection_receipt_must_be_durable_before_test_even_after_interruption(setup):
    s=setup;arm(s)
    for _ in range(8):execute_due(*s.args,**s.kw)
    from paperlab.fly_online_study import select_development
    completed={name:json.loads((s.root/'chunks'/name/'artifacts/summary.json').read_text()) for name in s.calls}
    atomic_json(s.root/'selection.json',select_development(s.env,completed))
    with pytest.raises(ValueError,match='Uncertain selection'):execute_due(*s.args,**s.kw)
    assert len(s.calls)==8


def test_completed_summary_cannot_change_on_resume(setup):
    s=setup;arm(s);execute_due(*s.args,**s.kw)
    p=s.root/'chunks'/s.calls[0]/'artifacts/summary.json'
    obj=json.loads(p.read_text());obj['outcome']['equity']=99999;atomic_json(p,obj)
    with pytest.raises(ValueError,match='Committed chunk summary'):execute_due(*s.args,**s.kw)
    assert len(s.calls)==1


@pytest.mark.parametrize('failure',[False,True])
def test_cloud_worker_commits_normal_cycle_before_chunk_and_settles_only_once(monkeypatch,failure):
    import cloud
    events=[];writes={}
    class Volume:
        def reload(self):events.append('reload')
        def commit(self):events.append('commit')
    monkeypatch.setattr(cloud,'volume',Volume());monkeypatch.setattr(cloud,'discovery_volume',Volume())
    for name in ('PAPERLAB_FLY','PAPERLAB_UNIVERSE','PAPERLAB_PAPER_STUDY','PAPERLAB_ONLINE_STUDY'):monkeypatch.setenv(name,'1')
    monkeypatch.setattr(cloud.modal,'current_function_call_id',lambda:'fc-normal')
    monkeypatch.setattr(cloud.modal,'current_input_id',lambda:'in-normal')
    def reserve(*a,**kw):
        assert kw['limit_override']==25 and kw['memory_gib']==8
        events.append('reserve');return {'started':time.time()}
    monkeypatch.setattr('paperlab.budget.reserve',reserve)
    def settle(*a):
        events.append('settle');return {'estimated_compute_usd':.01,'monthly_reserved_usd':2}
    monkeypatch.setattr('paperlab.budget.settle',settle)
    monkeypatch.setattr('paperlab.core.atomic_json',lambda p,v:writes.update({str(p):copy.deepcopy(v)}))
    monkeypatch.setattr('paperlab.fly_market_schedule.execute_due',lambda *a,**kw:None)
    monkeypatch.setattr('paperlab.fly_paper_schedule.execute_due',lambda *a,**kw:None)
    monkeypatch.setattr(Path,'exists',lambda self:True)
    def cycle(*a,**kw):events.append('normal');return {'status':'paper_research','normal_sentinel':42}
    monkeypatch.setattr('paperlab.multi.cycle',cycle)
    def due(*a,**kw):
        assert events[-2:]==['normal','commit']
        assert writes['/state/latest.json']['normal_sentinel']==42
        assert 'settle' not in events and kw['allow_compute']
        assert 500<kw['deadline']-time.monotonic()<=570
        events.append('study')
        if failure:raise RuntimeError('controlled research failure')
        return {'status':'paper_online_collecting'}
    monkeypatch.setattr('paperlab.fly_online_schedule.execute_due',due)
    if failure:
        with pytest.raises(RuntimeError,match='controlled research'):cloud._worker()
        assert writes['/state/latest.json']['normal_sentinel']==42
        assert events.count('reserve')==1 and 'settle' not in events and events[-1]=='commit'
        return
    result=cloud._worker()
    assert result['normal_sentinel']==42 and result['study_11']['status']=='paper_online_collecting'
    assert events.count('reserve')==events.count('normal')==events.count('study')==events.count('settle')==1
    assert events.index('normal')<events.index('study')<events.index('settle')


def test_source_manifest_includes_native_code_and_deployed_worker(tmp_path):
    from paperlab.fly_online_schedule import source_hashes
    shutil.copyfile('cloud.py',tmp_path/'cloud-source.py')
    hashes=source_hashes(tmp_path)
    assert hashes['cloud.py']==digest('cloud.py')
    assert hashes['paperlab/fly_online_schedule.py']==digest('paperlab/fly_online_schedule.py')
    assert hashes['vendor/stonkfly/stonkfly/neural/kernel.cpp']==digest('vendor/stonkfly/stonkfly/neural/kernel.cpp')
    assert hashes['vendor/stonkfly/stonkfly/neural/visual.py']==digest('vendor/stonkfly/stonkfly/neural/visual.py')
    assert all(hashes['paperlab/'+name]==digest(Path('paperlab')/name) for name in SOURCE_FILES)
