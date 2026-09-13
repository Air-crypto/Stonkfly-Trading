"""Recovery order, retained evidence, crash durability and shared budget guards."""
import copy
import json
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from paperlab import fly_online_recovery as recovery
from paperlab.core import atomic_json, digest
from paperlab.fly_market_study import signature
from paperlab.fly_online_protocol import chunk_name, chunk_order
from paperlab.fly_online_study import SOURCE_FILES


@pytest.fixture
def case(tmp_path, monkeypatch):
    state=tmp_path/'state'; original=state/recovery.ORIGINAL; root=state/recovery.DIRECTORY
    evidence=tmp_path/'evidence'; evidence.mkdir()
    p=json.loads(Path('reports/fly-online-recovery-protocol-11.json').read_text())
    atomic_json(evidence/'protocol.json',p)
    env={'sha256':p['plan_sha256'],'plan':{'registration':{'study':'11','phase_steps':24}}}
    sources={'paperlab/'+name:'a'*64 for name in SOURCE_FILES}
    monkeypatch.setattr(recovery,'verify_reference',lambda *a:(env,sources,{'vectors_reconstructed':49}))
    monkeypatch.setattr('paperlab.fly_online_study.validate',lambda e:e['plan'])
    def result(condition):
        stage,pool,arm=condition
        return {'status':'paper_online_chunk_completed','chunk':chunk_name(*condition),
            'stage':stage,'pool_index':pool,'arm':arm,'registration':env['plan']['registration'],
            'plan_sha256':env['sha256'],'code_sha256':{f:sources['paperlab/'+f] for f in SOURCE_FILES},
            'outcome':{'equity':251 if arm=='trained_online_reset_rates' else 249,
                'rows':[{'event':{'synthetic_dispatch_fixture':True}} for _ in range(24)]+[{'event':None}]}}
    for i,condition in enumerate(chunk_order()[:2]):
        name=chunk_name(*condition);folder=original/'chunks'/name
        atomic_json(folder/'artifacts/summary.json',result(condition))
        atomic_json(folder/'receipt.json',{'status':'completed','chunk':name,'call_id':f'original-{i}',
            'input_id':f'input-{i}','plan_sha256':env['sha256'],'source_sha256':signature(sources),
            'remote_path':str(folder/'artifacts'),'summary_sha256':digest(folder/'artifacts/summary.json'),
            'completed_at':p['created_at']-1})
    calls=[];commits=[]
    def run(env,memory,news,data,out,**kw):
        condition=kw['stage'],kw['pool_index'],kw['arm'];name=chunk_name(*condition)
        receipt=json.loads((out.parent/'receipt.json').read_text())
        assert receipt['status']=='claimed' and commits
        assert json.loads((state/'budget.json').read_text())['months']
        if kw['stage']=='test':
            assert len(kw['development'])==8
            assert (root/'selection-receipt.json').exists()
        else:assert kw['development'] is None
        calls.append(name);r=result(condition);atomic_json(out/'summary.json',r);return r
    def invoke(index=0,runner=run):
        return recovery.execute(state,tmp_path/'spec',evidence,Path.cwd(),call_id=f'recovery-{index}',
            input_id=f'recovery-input-{index}',commit=lambda:commits.append(True),run=runner)
    return SimpleNamespace(state=state,original=original,root=root,p=p,env=env,sources=sources,
        invoke=invoke,run=run,calls=calls,commits=commits)


def test_exactly_fourteen_new_chunks_preserve_controls_and_select_before_test(case):
    c=case;before={str(p):digest(p) for p in c.original.rglob('*') if p.is_file()}
    for i,condition in enumerate(chunk_order()[2:]):
        r=c.invoke(i)
        assert r['status']=='recovery_chunk_completed' and r['completed_chunks']==i+3
        assert c.calls[-1]==chunk_name(*condition)
    final=c.invoke(15)
    assert final['status']=='paper_online_recovery_captured' and final['audited'] is False
    assert final['aggregate']['selection']['selected']=='trained_online_reset_rates'
    assert len(c.calls)==14 and final['original_attempt']=='failed'
    assert before=={str(p):digest(p) for p in c.original.rglob('*') if p.is_file()}
    assert c.invoke(16)['status']=='paper_online_recovery_captured' and len(c.calls)==14


@pytest.mark.parametrize('failure',[RuntimeError,KeyboardInterrupt])
def test_failed_or_interrupted_claim_never_retried(case,failure):
    c=case;attempts=[]
    def fail(*a,**k):attempts.append(1);raise failure('interrupted fixture')
    with pytest.raises(failure):c.invoke(runner=fail)
    r=c.invoke(1)
    assert r['status']=='recovery_chunk_unresolved' and len(attempts)==1
    assert r['receipt']['status']==('failed' if failure is RuntimeError else 'claimed')
    assert r['receipt']['reserved_usd']>0


def test_budget_guard_prevents_claim_and_compute(case):
    from datetime import datetime,timezone
    c=case;month=datetime.now(timezone.utc).strftime('%Y-%m')
    atomic_json(c.state/'budget.json',{'first_month':month,'months':{month:18.75}})
    assert c.invoke()['status']=='budget_stopped'
    assert not c.calls and not (c.root/'chunks').exists()


@pytest.mark.parametrize('change',['summary','orphan','source','owner'])
def test_corrupt_or_unclaimed_evidence_cannot_advance(case,change):
    c=case;c.invoke();folder=c.root/'chunks'/chunk_name(*chunk_order()[2])
    if change=='summary':(folder/'artifacts/summary.json').write_text('{}')
    elif change=='orphan':(c.root/'chunks'/chunk_name(*chunk_order()[3])).mkdir()
    else:
        r=json.loads((folder/'receipt.json').read_text())
        r['source_sha256' if change=='source' else 'call_id']='changed' if change=='source' else 'original-0'
        atomic_json(folder/'receipt.json',r)
    with pytest.raises(ValueError):c.invoke(1)
    assert len(c.calls)==1


def test_partial_selection_commit_does_not_start_test(case):
    c=case
    for i in range(6):c.invoke(i)
    completed,_,_,_=recovery.retained_chunks(c.p,c.env,c.sources,c.original,c.root)
    atomic_json(c.root/'selection.json',recovery.select_development(c.env,completed))
    with pytest.raises(FileNotFoundError):c.invoke(6)
    assert len(c.calls)==6


def test_validation_fails_before_new_model_or_budget(case,monkeypatch):
    def reject(*a):raise ValueError('Sealed news differs')
    monkeypatch.setattr(recovery,'verify_reference',reject)
    with pytest.raises(ValueError,match='Sealed news'):case.invoke()
    assert not case.calls and not (case.state/'budget.json').exists()


@pytest.mark.parametrize('field,value',[('seed','1'),('policy','retry failures'),('controls',{})])
def test_fixed_amendment_rejects_changes(field,value):
    p=json.loads(Path('reports/fly-online-recovery-protocol-11.json').read_text())
    recovery.validate_protocol(p);p[field]=value
    with pytest.raises(ValueError):recovery.validate_protocol(p)


def test_entrypoint_imports_using_only_explicitly_packaged_modules(tmp_path):
    import ast
    import os
    import subprocess
    import sys
    tree=ast.parse(Path('recovery_cloud.py').read_text())
    # Reconstruct only the explicitly packaged Python modules, not the checkout.
    for node in ast.walk(tree):
        if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Attribute) or node.func.attr!='add_local_file':continue
        local,remote=node.args[:2]
        if not isinstance(remote,ast.Constant) or not isinstance(remote.value,str):continue
        target=Path(remote.value)
        if target.parent!=Path('/opt/paperlab') or target.suffix!='.py':continue
        assert isinstance(local,ast.BinOp) and isinstance(local.right,ast.Constant)
        (tmp_path/target.name).write_bytes(Path(local.right.value).read_bytes())
    result=subprocess.run([sys.executable,'-I','-c',
        'import sys; sys.path.insert(0,sys.argv[1]); import recovery_cloud; print("import passed")',str(tmp_path)],
        cwd=tmp_path,env={**os.environ,'PAPERLAB_FLY':'1','PAPERLAB_UNIVERSE':'1',
            'PAPERLAB_PAPER_STUDY':'1','PAPERLAB_ONLINE_STUDY':'1'},capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    assert 'import passed' in result.stdout
