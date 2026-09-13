"""Study 12 dispatcher with actual references and synthetic market ledgers."""
import copy
import json
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import pytest

from paperlab.core import atomic_json,digest
from paperlab.fly_rate_protocol import ARMS,CANDIDATE,chunk_name,chunk_order
from paperlab.fly_rate_schedule import DIRECTORY,REGISTRATION,execute_due
from paperlab.fly_rate_study import SOURCE_FILES,select_development
from paperlab.fly_rate_metrics import trading_metrics
from test_fly_rate_pipeline import sealed,ledger


@pytest.fixture
def setup(sealed,monkeypatch):
    env,base,reference=sealed;r=env['plan']['registration']
    state=base/'state';(state/'meme-pools-v1').mkdir(parents=True)
    shutil.copyfile(base/'news.db',state/'meme-pools-v1/news.db')
    discovery=base/'discovery';discovery.mkdir();shutil.copyfile(base/'universe.db',discovery/'universe-snapshot.db')
    specs=base/'specs';specs.mkdir();atomic_json(specs/REGISTRATION,r)
    root=state/DIRECTORY;events=[];calls=[]
    sources={'paperlab/'+name:'a'*64 for name in SOURCE_FILES};sources['rate_market_cloud.py']='b'*64
    monkeypatch.setattr('paperlab.fly_rate_schedule.source_hashes',lambda:copy.deepcopy(sources))
    def sealing(*args):
        assert json.loads((root/'sealing.json').read_text())['status']=='claimed'
        assert events[-1]=='commit';atomic_json(args[-1],env);return env
    def run(envelope,memory,news,data,out,**kwargs):
        stage,pool,arm=kwargs['stage'],kwargs['pool_index'],kwargs['arm'];name=chunk_name(stage,pool,arm)
        receipt=json.loads((root/'chunks'/name/'receipt.json').read_text())
        assert receipt['status']=='claimed' and receipt['call_id']==kw['call_id'] and events[-1]=='commit'
        assert 1<=kwargs['seconds']<=480
        selected=select_development(env,kwargs['development']) if stage=='test' else None
        if stage=='test':assert (root/'selection-receipt.json').exists()
        outcome=ledger(env,pool,arm,stage=stage,hold_after_buy=arm==CANDIDATE)
        result={'status':'paper_rate_chunk_completed','chunk':name,'stage':stage,'pool_index':pool,
                'pool':r['cohort'][pool],'arm':arm,'plan_sha256':env['sha256'],'registration':r,
                'outcome':outcome,'trading_metrics':trading_metrics(outcome['rows']),'selection':selected,
                'code_sha256':{f:sources['paperlab/'+f] for f in SOURCE_FILES}}
        out.mkdir(parents=True);atomic_json(out/'summary.json',result);calls.append(name);return result
    kw={'call_id':'fc-arm','input_id':'in-arm','commit':lambda:events.append('commit'),
        'run':run,'deadline':time.monotonic()+570,'now':r['development_start']-1,'seal_inputs':sealing}
    return SimpleNamespace(args=(state,discovery,specs,reference),kw=kw,root=root,env=env,r=r,
                           events=events,calls=calls,sources=sources)


def arm(s):
    assert execute_due(*s.args,**s.kw)['status']=='paper_rate_collecting'
    s.kw.update(now=s.r['end']+1,call_id='fc-capture-0',input_id='in-capture-0')


def test_all_sixteen_chunks_run_once_with_ledger_selection_before_test(setup):
    s=setup;arm(s)
    for i,chunk in enumerate(chunk_order()):
        s.kw.update(call_id=f'fc-{i}',input_id=f'in-{i}',deadline=time.monotonic()+570)
        result=execute_due(*s.args,**s.kw)
        assert result['status']=='paper_rate_chunk_completed' and s.calls[-1]==chunk_name(*chunk)
        assert len(s.calls)==i+1
    result=execute_due(*s.args,**s.kw)
    assert result['status']=='paper_rate_study_completed'
    summary=json.loads((s.root/'summary.json').read_text())
    assert summary['selection']['selected']==CANDIDATE
    assert summary['selection']['ledgers_replayed_before_selection'] and not summary['audited']
    assert not summary['test_coverage_sufficient']
    assert summary['trading_metrics'][CANDIDATE]['test'][1]['unavailable_marks']>=10


@pytest.mark.parametrize('failure',[RuntimeError,KeyboardInterrupt])
def test_failed_or_uncertain_capture_is_not_retried(setup,failure):
    s=setup;arm(s);attempts=[]
    def fail(*a,**kw):attempts.append(1);raise failure('synthetic interruption')
    s.kw['run']=fail
    with pytest.raises(failure):execute_due(*s.args,**s.kw)
    s.kw.update(call_id='fc-next',input_id='in-next')
    result=execute_due(*s.args,**s.kw)
    assert result['status']=='paper_rate_chunk_unresolved' and result['receipt']['status']=='failed'
    assert attempts==[1]


def test_interrupted_seal_never_repeats(setup):
    s=setup;arm(s);attempts=[]
    def fail(*a):attempts.append(1);raise KeyboardInterrupt()
    s.kw['seal_inputs']=fail
    with pytest.raises(KeyboardInterrupt):execute_due(*s.args,**s.kw)
    assert execute_due(*s.args,**s.kw)['status']=='paper_rate_seal_unresolved'
    assert attempts==[1] and not s.calls


@pytest.mark.parametrize('bad',['late','source','registration','memory','evidence','time','disabled','snapshot','news'])
def test_arming_and_resource_guards_precede_capture(setup,bad):
    s=setup
    if bad=='late':
        s.kw['now']=s.r['development_start']
        assert execute_due(*s.args,**s.kw)['status']=='paper_rate_registration_missed'
        s.kw['now']=s.r['end']+1
        assert execute_due(*s.args,**s.kw)['status']=='paper_rate_registration_missed'
        assert not s.calls;return
    arm(s)
    if bad=='source':s.sources['rate_market_cloud.py']='e'*64
    if bad=='registration':
        r=copy.deepcopy(s.r);r['recorded_at']-=1;atomic_json(s.args[2]/REGISTRATION,r)
    if bad=='memory':(s.args[3]/'pool0-memory.npz').write_bytes(b'changed')
    if bad=='evidence':(s.args[3]/'mechanism-audit.json').write_bytes(b'{}')
    if bad in ('source','registration','memory','evidence'):
        with pytest.raises(ValueError):execute_due(*s.args,**s.kw)
    else:
        if bad=='time':s.kw['deadline']=time.monotonic()+100
        if bad=='disabled':s.kw['allow_compute']=False
        if bad=='snapshot':(s.args[1]/'universe-snapshot.db').unlink()
        if bad=='news':(s.args[0]/'meme-pools-v1/news.db').unlink()
        assert execute_due(*s.args,**s.kw)['status'].startswith('paper_rate_waiting')
    assert not s.calls


def test_missing_selection_receipt_blocks_test_capture(setup):
    s=setup;arm(s)
    for i in range(8):
        s.kw.update(call_id=f'fc-{i}',input_id=f'in-{i}');execute_due(*s.args,**s.kw)
    development={name:json.loads((s.root/'chunks'/name/'artifacts/summary.json').read_text()) for name in s.calls}
    atomic_json(s.root/'selection.json',select_development(s.env,development))
    s.kw.update(call_id='fc-test',input_id='in-test')
    with pytest.raises(ValueError,match='Uncertain selection'):execute_due(*s.args,**s.kw)
    assert len(s.calls)==8


def test_two_chunks_cannot_claim_the_same_call(setup):
    s=setup;arm(s)
    execute_due(*s.args,**s.kw)
    with pytest.raises(ValueError,match='same owning'):execute_due(*s.args,**s.kw)
    assert len(s.calls)==1


@pytest.mark.parametrize('corruption',['before_development','after_test','nan'])
def test_selection_receipt_must_follow_development_and_precede_test(setup,corruption):
    s=setup;arm(s)
    for i in range(9):
        s.kw.update(call_id=f'fc-{i}',input_id=f'in-{i}');execute_due(*s.args,**s.kw)
    path=s.root/'selection-receipt.json';receipt=json.loads(path.read_text())
    receipt['selected_at']=0 if corruption=='before_development' else time.time()+100 if corruption=='after_test' else float('nan')
    # Bypass the production writer deliberately: it already rejects NaN.
    path.write_text(json.dumps(receipt));s.kw.update(call_id='fc-next',input_id='in-next')
    with pytest.raises(ValueError,match='Selection receipt'):execute_due(*s.args,**s.kw)
    assert len(s.calls)==9


def test_source_manifest_covers_recorder_auditor_runtime_native_and_cloud():
    from paperlab.fly_rate_schedule import source_hashes
    sources=source_hashes()
    for name in ('rate_market_cloud.py','cloud.py','paperlab/fly_rate_runtime.py',
                 'paperlab/fly_rate_inputs.py','paperlab/fly_rate_metrics.py','paperlab/fly_learning_scale_credit.py',
                 'vendor/stonkfly/stonkfly/neural/kernel.cpp'):
        assert sources[name]==digest(name)
