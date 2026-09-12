import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

from paperlab.core import atomic_json
from paperlab.fly_market_study import signature
from paperlab.fly_market_schedule import execute_due, DIRECTORY, REGISTRATION


def setup_study(tmp_path, monkeypatch, last=None):
    import paperlab.fly_market_schedule as schedule
    registration=json.loads((Path('reports')/REGISTRATION).read_text())
    p=json.loads(Path('reports/fly-market-study-07-plan.json').read_text())['plan']
    p.update(registration, snapshot_end=registration['end']);p.pop('end');p.pop('restoration_timing')
    envelope={'plan':p,'sha256':signature(p)}
    discovery=tmp_path/'discovery';discovery.mkdir()
    archive=discovery/'universe-snapshot.db'
    with sqlite3.connect(archive) as db:
        db.execute('CREATE TABLE observations(payload TEXT)')
        db.execute('INSERT INTO observations VALUES(?)',(json.dumps({'observed':registration['end'] if last is None else last}),))
    state=tmp_path/'state';root=state/DIRECTORY;committed=[];calls=[];settled=[]
    def commit():
        path=root/'cloud-call.json'
        committed.append(json.loads(path.read_text()) if path.exists() else None)
    def seal(archive,parent,registration,out):
        assert committed[-1]['status']=='claimed'
        atomic_json(out,envelope);return envelope
    monkeypatch.setattr(schedule,'seal_registered',seal)
    def run(request,output,data):
        assert committed[-1]['status']=='pending'
        assert json.loads((root/'plan.json').read_text())==request['market_plan']
        calls.append(request)
        return {'status':'market_study_completed','run_id':request['run_id'],
                'remote_path':str(Path(output)/request['run_id'])}
    kwargs=dict(call_id='fc-scheduled',input_id='in-scheduled',commit=commit,run=run,
                settle_budget=lambda:(settled.append(True) or {'estimated_compute_usd':.01}),now=registration['end'])
    args=(state,discovery,Path('reports'))
    return SimpleNamespace(args=args,kwargs=kwargs,registration=registration,envelope=envelope,
                           root=root,committed=committed,calls=calls,settled=settled)


@pytest.mark.parametrize('reason',['clock','snapshot','missing'])
def test_due_study_requires_closed_window_and_actual_snapshot(tmp_path,monkeypatch,reason):
    s=setup_study(tmp_path,monkeypatch,last=1 if reason=='snapshot' else None)
    if reason=='clock':s.kwargs['now']-=1
    if reason=='missing':(s.args[1]/'universe-snapshot.db').unlink()
    assert execute_due(*s.args,**s.kwargs) is None
    assert not s.calls and not s.committed and not s.root.exists()


def test_cloud_study_claims_before_execution_and_runs_only_once(tmp_path,monkeypatch):
    s=setup_study(tmp_path,monkeypatch)
    result=execute_due(*s.args,**s.kwargs)
    assert result['status']=='market_study_completed'
    assert [c['status'] for c in s.committed]==['claimed','pending','completed']
    assert s.committed[-1]['call_id']=='fc-scheduled' and s.committed[-1]['input_id']=='in-scheduled'
    assert len(s.calls)==len(s.settled)==1
    assert execute_due(*s.args,**s.kwargs) is None
    assert len(s.calls)==1


@pytest.mark.parametrize('failure',['seal','run','hard_termination'])
def test_failed_or_uncertain_study_is_never_automatically_retried(tmp_path,monkeypatch,failure):
    s=setup_study(tmp_path,monkeypatch)
    def fail(*args,**kwargs):
        if failure=='hard_termination':raise KeyboardInterrupt()
        raise RuntimeError('controlled failure')
    if failure=='seal':monkeypatch.setattr('paperlab.fly_market_schedule.seal_registered',fail)
    else:s.kwargs['run']=fail
    with pytest.raises(KeyboardInterrupt if failure=='hard_termination' else RuntimeError):
        execute_due(*s.args,**s.kwargs)
    receipt=json.loads((s.root/'cloud-call.json').read_text())
    assert receipt['status']==('pending' if failure=='hard_termination' else 'failed')
    assert not s.settled
    assert execute_due(*s.args,**s.kwargs) is None


def test_scheduled_observer_attaches_without_any_spawn_even_on_timeout(tmp_path,monkeypatch):
    import modal
    from paperlab.fly_market_cloud import observe_scheduled
    s=setup_study(tmp_path,monkeypatch);execute_due(*s.args,**s.kwargs)
    class Volume:
        def read_file(self,path):return [(s.args[0]/path.lstrip('/')).read_bytes()]
    observed=[]
    class Call:
        def get(self,timeout):raise TimeoutError('still running')
    def from_id(call_id):observed.append(call_id);return Call()
    monkeypatch.setattr(modal.Volume,'from_name',lambda *a,**k:Volume())
    monkeypatch.setattr(modal.FunctionCall,'from_id',from_id)
    monkeypatch.setattr(modal.Function,'from_name',lambda *a,**k:pytest.fail('Observer attempted a worker submission'))
    out=tmp_path/'observer'
    observe_scheduled(s.registration,out);observe_scheduled(s.registration,out)
    assert observed==['fc-scheduled','fc-scheduled']
    receipt=json.loads((out/'cloud-call.json').read_text());receipt['call_id']='fc-unrelated';atomic_json(out/'cloud-call.json',receipt)
    with pytest.raises(ValueError,match='another call'):observe_scheduled(s.registration,out)
    remote=json.loads((s.root/'cloud-call.json').read_text());remote['status']='failed';atomic_json(s.root/'cloud-call.json',remote)
    with pytest.raises(RuntimeError,match='uncertain'):observe_scheduled(s.registration,tmp_path/'never-submitted')


def test_attach_mode_refuses_to_spawn_without_local_receipt(tmp_path,monkeypatch):
    import modal
    from paperlab.fly_market_cloud import cloud_run
    s=setup_study(tmp_path,monkeypatch)
    monkeypatch.setattr(modal.Function,'from_name',lambda *a,**k:pytest.fail('Forbidden submission'))
    with pytest.raises(RuntimeError,match='cannot submit'):
        cloud_run(s.envelope,s.registration,tmp_path/'missing',allow_submit=False)


@pytest.mark.parametrize('budget_available',[True,False])
def test_scheduled_study_uses_existing_reservation_and_replaces_one_baseline_cycle(monkeypatch,budget_available):
    import cloud
    events=[]
    class Volume:
        def reload(self):events.append('reload')
        def commit(self):events.append('commit')
    monkeypatch.setattr(cloud,'volume',Volume());monkeypatch.setattr(cloud,'discovery_volume',Volume())
    monkeypatch.setenv('PAPERLAB_FLY','1');monkeypatch.setenv('PAPERLAB_UNIVERSE','1')
    def reserve(*args,**kwargs):
        assert kwargs['limit_override']==25 and kwargs['memory_gib']==8
        events.append('reserve');return {'started':0} if budget_available else None
    def due(*args,**kwargs):
        assert events==['reload','reserve','commit','reload']
        events.append('study');return {'status':'market_study_completed'}
    monkeypatch.setattr('paperlab.budget.reserve',reserve)
    monkeypatch.setattr('paperlab.fly_market_schedule.execute_due',due)
    monkeypatch.setattr('paperlab.multi.cycle',lambda *a,**k:pytest.fail('Study must replace this baseline cycle'))
    result=cloud._worker()
    assert result['status']==('market_study_completed' if budget_available else 'budget_stopped')
    assert events.count('study')==int(budget_available)
