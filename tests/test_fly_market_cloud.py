import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from paperlab.fly_market_cloud import build_report,cloud_run,validate_registration


def fixture():
    envelope=json.loads(Path('reports/fly-market-study-05-plan.json').read_text())
    registration=json.loads(Path('reports/fly-market-study-05-preregistration.json').read_text())
    report=json.loads(Path('reports/fly-market-study-05.json').read_text())
    raw={pool:{arm:{phase:{**{k:v for k,v in p.items() if k!='decisions'},'rows':[
        {**{k:v for k,v in d.items() if k!='neural'},'event':d['neural']} for d in p['decisions']]} for phase,p in phases.items()}
        for arm,phases in arms.items()} for pool,arms in report['phase_diagnostics'].items()}
    return envelope,registration,report,raw


def test_market_materialization_reconciles_published_ledger():
    envelope,registration,report,raw=fixture();validate_registration(envelope,registration)
    result=build_report(envelope,report,raw,report['selection'])
    assert result['phase_diagnostics']==report['phase_diagnostics']
    assert result['verification']['test_fills_per_arm']=={'pristine_frozen':2,'trained_frozen':1,'online_original':2}


@pytest.mark.parametrize('mutation',['time','terminal','fee','selection','aggregate'])
def test_market_observer_rejects_inconsistent_results(mutation):
    envelope,_,report,raw=fixture();selection=copy.deepcopy(report['selection']);pool=next(iter(raw));p=raw[pool]['trained_frozen']['test']
    if mutation=='time':p['rows'][0]['decision_ts']+=1
    if mutation=='terminal':p['rows'][-1]['equity']+=1
    if mutation=='fee':p['fees']+=1
    if mutation=='selection':selection['selected']='trained_frozen'
    if mutation=='aggregate':report['total_equity']['trained_frozen']['test']+=1
    with pytest.raises(ValueError):build_report(envelope,report,raw,selection)


def test_market_registration_rejects_changed_or_late_protocol():
    envelope,registration,_,_=fixture();registration['recorded_at']=registration['end']
    with pytest.raises(ValueError):validate_registration(envelope,registration)
    envelope,registration,_,_=fixture();registration['arms']={}
    with pytest.raises(ValueError):validate_registration(envelope,registration)


def test_market_observer_reuses_saved_call(tmp_path,monkeypatch):
    import modal
    calls=[]
    class Remote:
        object_id='fc-market-test'
        def get(self,timeout=0):raise TimeoutError('running')
    remote=Remote();monkeypatch.setattr(modal.Function,'from_name',lambda *a,**k:SimpleNamespace(spawn=lambda **kw:(calls.append(kw) or remote)))
    monkeypatch.setattr(modal.FunctionCall,'from_id',lambda *a:remote)
    envelope,registration,_,_=fixture();cloud_run(envelope,registration,tmp_path);cloud_run(envelope,registration,tmp_path)
    assert len(calls)==1
    registration['recorded_at']+=1
    with pytest.raises(ValueError,match='different request'):cloud_run(envelope,registration,tmp_path)
    assert len(calls)==1


def test_uncertain_market_submission_never_duplicates(tmp_path,monkeypatch):
    import modal
    calls=[]
    def spawn(**kw):calls.append(kw);raise ConnectionError('response lost')
    monkeypatch.setattr(modal.Function,'from_name',lambda *a,**k:SimpleNamespace(spawn=spawn))
    envelope,registration,_,_=fixture()
    with pytest.raises(ConnectionError):cloud_run(envelope,registration,tmp_path)
    with pytest.raises(RuntimeError,match='Uncertain'):cloud_run(envelope,registration,tmp_path)
    assert len(calls)==1
