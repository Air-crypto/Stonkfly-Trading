import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

from paperlab.core import atomic_json
from paperlab.fly_market_study import signature
from paperlab.fly_paper_inputs import news_stamps
from paperlab.fly_paper_schedule import execute_due,DIRECTORY


def fixture(tmp_path,monkeypatch):
    r=json.loads(Path('reports/fly-market-study-09-preregistration.json').read_text())
    a=json.loads(Path('reports/fly-paper-memory-audit-01.json').read_text())
    series=json.loads(Path('reports/fly-market-study-08-plan.json').read_text())['plan']['series']
    p={'registration':r,'training_audit':a,'series':series,'snapshot_end':r['end'],'snapshot_sha256':'a'*64,'news_snapshot_sha256':'b'*64,
       'news_features':{str(float(t)):[0]*50 for t in news_stamps(r,series)}}
    envelope={'plan':p,'sha256':signature(p)};state=tmp_path/'state';discovery=tmp_path/'discovery';discovery.mkdir()
    (state/'meme-pools-v1').mkdir(parents=True);(state/'meme-pools-v1/news.db').write_bytes(b'fixture; sealer mocked')
    with sqlite3.connect(discovery/'universe-snapshot.db') as db:
        db.execute('CREATE TABLE observations(payload TEXT)');db.execute('INSERT INTO observations VALUES(?)',(json.dumps({'observed':r['end']}),))
    committed=[];calls=[];settled=[];root=state/DIRECTORY
    def commit():committed.append(json.loads((root/'cloud-call.json').read_text()))
    def seal(*args):
        assert committed[-1]['status']=='claimed';atomic_json(args[-1],envelope);return envelope
    monkeypatch.setattr('paperlab.fly_paper_schedule.seal',seal)
    def run(*args):
        assert committed[-1]['status']=='pending';calls.append(args);return {'status':'paper_checkpoint_study_completed'}
    return SimpleNamespace(args=(state,discovery,Path('reports'),tmp_path/'memory'),
        kwargs={'call_id':'fc-paper-test','input_id':'in-paper-test','commit':commit,'run':run,'settle_budget':lambda:(settled.append(True) or {'estimated_compute_usd':.01}),'now':r['end']},
        registration=r,root=root,committed=committed,calls=calls,settled=settled)


@pytest.mark.parametrize('reason',['clock','snapshot','news'])
def test_paper_schedule_requires_closed_prices_and_news_archive(tmp_path,monkeypatch,reason):
    s=fixture(tmp_path,monkeypatch)
    if reason=='clock':s.kwargs['now']-=1
    if reason=='snapshot':
        with sqlite3.connect(s.args[1]/'universe-snapshot.db') as db:db.execute('DELETE FROM observations')
    if reason=='news':(s.args[0]/'meme-pools-v1/news.db').unlink()
    assert execute_due(*s.args,**s.kwargs) is None and not s.calls and not s.committed


def test_paper_schedule_claims_once_before_running_and_settles_once(tmp_path,monkeypatch):
    s=fixture(tmp_path,monkeypatch);result=execute_due(*s.args,**s.kwargs)
    assert result['status']=='paper_checkpoint_study_completed'
    assert [r['status'] for r in s.committed]==['claimed','pending','completed']
    assert execute_due(*s.args,**s.kwargs) is None
    assert len(s.calls)==len(s.settled)==1


@pytest.mark.parametrize('error',[RuntimeError,KeyboardInterrupt])
def test_failed_and_hard_stopped_paper_comparison_cannot_retry(tmp_path,monkeypatch,error):
    s=fixture(tmp_path,monkeypatch)
    def fail(*args):raise error('controlled interruption')
    s.kwargs['run']=fail
    with pytest.raises(error):execute_due(*s.args,**s.kwargs)
    assert not s.settled and execute_due(*s.args,**s.kwargs) is None


def test_paper_observer_reuses_saved_call_without_a_submission_path(tmp_path,monkeypatch):
    import modal
    from paperlab.fly_paper_cloud import observe
    s=fixture(tmp_path,monkeypatch);execute_due(*s.args,**s.kwargs)
    class Volume:
        def read_file(self,path):return [(s.args[0]/path.lstrip('/')).read_bytes()]
    observed=[]
    class Call:
        def get(self,timeout):raise TimeoutError('running')
    def from_id(identity):observed.append(identity);return Call()
    monkeypatch.setattr(modal.Volume,'from_name',lambda *a,**k:Volume())
    monkeypatch.setattr(modal.FunctionCall,'from_id',from_id)
    monkeypatch.setattr(modal.Function,'from_name',lambda *a,**k:pytest.fail('Observer must never submit compute'))
    observe(s.registration,tmp_path/'out');observe(s.registration,tmp_path/'out')
    assert observed==['fc-paper-test','fc-paper-test']
    receipt=json.loads((s.root/'cloud-call.json').read_text());receipt['status']='failed';atomic_json(s.root/'cloud-call.json',receipt)
    with pytest.raises(RuntimeError,match='uncertain'):observe(s.registration,tmp_path/'out')


def test_worker_paper_study_uses_one_existing_budget_reservation(monkeypatch):
    import cloud
    events=[]
    class Volume:
        def reload(self):pass
        def commit(self):events.append('commit')
    monkeypatch.setattr(cloud,'volume',Volume());monkeypatch.setattr(cloud,'discovery_volume',Volume())
    for name in ('PAPERLAB_FLY','PAPERLAB_UNIVERSE','PAPERLAB_PAPER_STUDY'):monkeypatch.setenv(name,'1')
    def reserve(*args,**kwargs):
        assert kwargs['limit_override']==25 and kwargs['memory_gib']==8
        events.append('reserve');return {'started':0}
    monkeypatch.setattr('paperlab.budget.reserve',reserve)
    monkeypatch.setattr('paperlab.fly_market_schedule.execute_due',lambda *a,**k:None)
    monkeypatch.setattr('paperlab.fly_paper_schedule.execute_due',lambda *a,**k:{'status':'paper_checkpoint_study_completed'})
    monkeypatch.setattr('paperlab.multi.cycle',lambda *a,**k:pytest.fail('No baseline cycle alongside this study'))
    assert cloud._worker()['status']=='paper_checkpoint_study_completed'
    assert events==['reserve','commit']
