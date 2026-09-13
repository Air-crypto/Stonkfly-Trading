"""Checkpoint eligibility, durable capture and cloud budget ordering; no brain runs."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import sqlite3
import time

import pytest

from paperlab.core import atomic_json, digest
from paperlab.fly_rate_checkpoint import DIRECTORY, POLICY, execute, select_cohort, source_hashes
from paperlab.universe import Pool, Store


@pytest.fixture
def snapshot(tmp_path):
    now = int(time.time()//300)*300+1
    state = tmp_path/'state'; paper = state/'meme-pools-v1'; paper.mkdir(parents=True)
    discovery = tmp_path/'discovery'; discovery.mkdir()
    store = Store(discovery/'universe-snapshot.db')
    pools = [Pool(f'solana:p{i}', 'solana', f'p{i}', f't{i}', f'FIXTURE{i}',
                  1., 100000., 10000.+i*1000, 20, 20, now-86400, now, source='synthetic')
             for i in range(4)]
    pools[3] = replace(pools[3], network='base', key='base:p3')
    for minute in range(-300, 1):
        store.add([replace(p, observed=now+minute*60) for p in pools])
    store.db.close()
    last_slot = int(now//300)-1; assigned = (last_slot-23)*300
    lanes = []
    for i,p in enumerate(pools):
        checkpoint = paper/f'fly-{i}-{last_slot}.npz'; checkpoint.write_bytes(b'checkpoint fixture '+bytes([i]))
        lanes.append({'pool': p.key, 'checkpoint': str(checkpoint), 'assigned': assigned,
                      'broker': {'cash': str(10000 if i == 0 else 1)}})
    with sqlite3.connect(paper/'paper.db') as db:
        db.executescript('CREATE TABLE state(name TEXT PRIMARY KEY,payload TEXT);'
                        'CREATE TABLE ledger(slot INTEGER,name TEXT,payload TEXT);')
        db.execute('INSERT INTO state VALUES (?,?)', ('fly', json.dumps(lanes)))
        for n in range(24):
            sleeves = [{'sleeve': i, 'pool': p.key, 'detail': {'brain_ms': (n+1)*500,
                'memory': {'sha256': 'a'*64},
                'learning_diagnostics': {'equity_reward_usd': 1 if n%2 else -1,
                                         'learning_enabled': True}}} for i,p in enumerate(pools)]
            db.execute('INSERT INTO ledger VALUES (?,?,?)', (last_slot-23+n, 'fly', json.dumps({'sleeves': sleeves})))
    return state, discovery, now, pools, lanes


def select(s):
    state,discovery,now,*_ = s
    return select_cohort(state/'meme-pools-v1/paper.db', discovery/'universe-snapshot.db', now)


def test_selection_uses_coverage_volume_and_contract_identity_not_returns(snapshot):
    s=snapshot; result=select(s)
    assert result['cohort'] == ['solana:p2', 'solana:p1']
    assert result['candidates'][3]['reasons'] == ['not_solana']
    assert all(c['eligible_slots'] == 24 for c in result['candidates'])
    with sqlite3.connect(s[0]/'meme-pools-v1/paper.db') as db:
        lanes=copy.deepcopy(s[4]);lanes[2]['broker']['cash']='-9999999';lanes[0]['broker']['cash']='99999999'
        db.execute("UPDATE state SET payload=? WHERE name='fly'",(json.dumps(lanes),))
    assert select(s)['cohort'] == result['cohort']


@pytest.mark.parametrize('reason',['stale','sparse','untrained','version','missing','recent_context'])
def test_unqualified_checkpoints_are_not_promoted(snapshot,reason):
    s=snapshot; state,discovery,now,pools,lanes=s
    archive=discovery/'universe-snapshot.db'
    if reason in ('stale','sparse','missing','recent_context'):
        with sqlite3.connect(archive) as db:
            if reason=='missing':db.execute("DELETE FROM pools WHERE key='solana:p2'")
            if reason in ('sparse','recent_context'):
                if reason=='sparse':db.execute("DELETE FROM observations WHERE key='solana:p2' AND slot>?",(int((now-3600)//60),))
                else:db.execute("DELETE FROM observations WHERE key='solana:p2' AND slot<?",(int((now-120)//60),))
            if reason=='stale':
                raw=json.loads(db.execute("SELECT payload FROM pools WHERE key='solana:p2'").fetchone()[0]);raw['observed']=now-301
                db.execute("UPDATE pools SET payload=?,last_seen=? WHERE key='solana:p2'",(json.dumps(raw),now-301))
    else:
        with sqlite3.connect(state/'meme-pools-v1/paper.db') as db:
            if reason=='untrained':
                for slot,raw in db.execute('SELECT slot,payload FROM ledger').fetchall():
                    value=json.loads(raw);value['sleeves'][2]['detail']['learning_diagnostics']['equity_reward_usd']=0
                    db.execute('UPDATE ledger SET payload=? WHERE slot=?',(json.dumps(value),slot))
            else:
                lanes[2]['checkpoint']=str(state/'bad.npz')
                db.execute("UPDATE state SET payload=? WHERE name='fly'",(json.dumps(lanes),))
    result=select(s);assert result['cohort']==['solana:p1','solana:p0']
    assert result['candidates'][2]['reasons']


def test_contract_duplicate_cannot_fill_both_cohort_slots(snapshot):
    state,discovery,now,*_=snapshot
    with sqlite3.connect(discovery/'universe-snapshot.db') as db:
        raw=json.loads(db.execute("SELECT payload FROM pools WHERE key='solana:p1'").fetchone()[0]);raw['token']='t2'
        db.execute("UPDATE pools SET payload=? WHERE key='solana:p1'",(json.dumps(raw),))
    assert select(snapshot)['cohort']==['solana:p2','solana:p0']


def test_insufficient_qualified_pools_fails_without_selection(snapshot):
    with sqlite3.connect(snapshot[1]/'universe-snapshot.db') as db:
        db.execute("DELETE FROM pools WHERE key IN ('solana:p1','solana:p2')")
    with pytest.raises(ValueError,match='Fewer than two'):select(snapshot)


def mocked_export(capture,output,brain):
    output.mkdir();(output/'pool0-memory.npz').write_bytes(b'memory0')
    (output/'pool1-memory.npz').write_bytes(b'memory1')
    value={'synthetic_export_fixture':True,'capture_sha256':digest(capture/'capture.json')}
    atomic_json(output/'audit.json',value);return value


def test_capture_commits_claim_before_copy_and_export_then_rejects_repeat(snapshot,monkeypatch):
    state,discovery,now,*_=snapshot;events=[]
    sources={'test':'a'*64};monkeypatch.setattr('paperlab.fly_rate_checkpoint.source_hashes',lambda:sources)
    monkeypatch.setattr('paperlab.fly_rate_checkpoint.export',mocked_export)
    def reference(data):
        claim=json.loads((state/DIRECTORY/'claim.json').read_text())
        assert claim['status']=='claimed' and claim['call_id']=='fc-capture'
        assert json.loads((state/'budget.json').read_text())['months']
        assert events[-1]=='commit';events.append('reference');return object()
    request={'policy':POLICY,'source_sha256':sources}
    result=execute(request,state,discovery,call_id='fc-capture',input_id='in-capture',
                   commit=lambda:events.append('commit'),build_reference=reference,clock=lambda:now+2)
    assert result['neural_observations']==result['paper_orders']==0
    assert result['budget']['reserved_usd']==pytest.approx(.1608936)
    assert result['budget']['monthly_limit_usd']==25 and result['budget']['price_multiplier']==3
    assert result['cohort']==['solana:p2','solana:p1']
    root=state/DIRECTORY
    for path,sha in result['artifact_sha256'].items():assert digest(root/path)==sha
    claim=json.loads((root/'claim.json').read_text())
    assert claim['status']=='completed' and claim['result_sha256']==digest(root/'result.json')
    with pytest.raises(ValueError,match='already claimed'):
        execute(request,state,discovery,call_id='fc-other',input_id='in-other',commit=lambda:None)
    assert events.count('reference')==1


@pytest.mark.parametrize('failure',[RuntimeError,KeyboardInterrupt])
def test_failed_capture_is_terminal_and_keeps_full_reservation(snapshot,monkeypatch,failure):
    state,discovery,now,*_=snapshot;sources={'test':'b'*64}
    monkeypatch.setattr('paperlab.fly_rate_checkpoint.source_hashes',lambda:sources)
    def fail(*a):raise failure('synthetic export failure')
    monkeypatch.setattr('paperlab.fly_rate_checkpoint.export',fail)
    request={'policy':POLICY,'source_sha256':sources}
    with pytest.raises(failure):
        execute(request,state,discovery,call_id='fc-failure',input_id='in-failure',commit=lambda:None,
                build_reference=lambda *a:object(),clock=lambda:now+2)
    claim=json.loads((state/DIRECTORY/'claim.json').read_text())
    assert claim['status']=='failed'
    assert sum(json.loads((state/'budget.json').read_text())['months'].values())==pytest.approx(.1608936)
    with pytest.raises(ValueError,match='already claimed'):
        execute(request,state,discovery,call_id='fc-next',input_id='in-next',commit=lambda:None)


def test_bad_sources_rejected_before_reservation(snapshot):
    state,discovery,*_=snapshot
    with pytest.raises(ValueError,match='sources differ'):
        execute({'policy':POLICY,'source_sha256':{}},state,discovery,call_id='fc-bad',input_id='in-bad',commit=lambda:None)
    assert not (state/'budget.json').exists() and not (state/DIRECTORY).exists()


def test_source_manifest_covers_native_exporter_graph_and_cloud_lease():
    hashes=source_hashes()
    for name in ('cloud.py','rate_checkpoint_cloud.py','paperlab/fly_rate_checkpoint.py',
                 'paperlab/fly_paper_memory.py','paperlab/universe.py','paperlab/budget.py',
                 'vendor/stonkfly/stonkfly/neural/brain.py','vendor/stonkfly/stonkfly/neural/kernel.cpp'):
        assert hashes[name]==digest(name)


def test_cloud_entrypoint_reloads_both_volumes_inside_shared_lease(monkeypatch):
    monkeypatch.setenv('PAPERLAB_FLY','1');monkeypatch.setenv('PAPERLAB_UNIVERSE','1')
    import sys
    sys.modules.pop('cloud',None)
    import rate_checkpoint_cloud as cloud
    events=[]
    class Volume:
        def __init__(self,name):self.name=name
        def reload(self):events.append('reload-'+self.name)
        def commit(self):events.append('commit')
    monkeypatch.setattr(cloud,'volume',Volume('state'))
    monkeypatch.setattr(cloud,'discovery_volume',Volume('discovery'))
    monkeypatch.setattr(cloud.modal,'current_function_call_id',lambda:'fc-cloud')
    monkeypatch.setattr(cloud.modal,'current_input_id',lambda:'in-cloud')
    def execute(*args,**kw):
        assert events==['lease','reload-state','reload-discovery']
        assert args==({'test':True},'/state','/discovery')
        assert kw['call_id']=='fc-cloud' and kw['input_id']=='in-cloud'
        kw['commit']();return {'status':'synthetic'}
    monkeypatch.setattr('paperlab.fly_rate_checkpoint.execute',execute)
    def exclusive(name,fn,*args):
        assert name=='worker';events.append('lease');return fn(*args)
    monkeypatch.setattr(cloud,'exclusive',exclusive)
    assert cloud.worker.local({'test':True})=={'status':'synthetic'}
    assert events[-1]=='commit'


def test_capture_waits_only_for_frozen_checkpoint_slot_boundary(snapshot,monkeypatch):
    state,discovery,now,*_=snapshot;selected=select(snapshot);sources={'test':'d'*64}
    monkeypatch.setattr('paperlab.fly_rate_checkpoint.source_hashes',lambda:sources)
    monkeypatch.setattr('paperlab.fly_rate_checkpoint.select_cohort',lambda *args:selected)
    monkeypatch.setattr('paperlab.fly_rate_checkpoint.export',mocked_export)
    current=[now-2];waits=[]
    def sleep(seconds):
        assert 0<seconds<=10;waits.append(seconds);current[0]+=seconds
    result=execute({'policy':POLICY,'source_sha256':sources},state,discovery,
        call_id='fc-wait',input_id='in-wait',commit=lambda:None,build_reference=lambda *a:object(),
        clock=lambda:current[0],sleep=sleep)
    capture=json.loads((state/DIRECTORY/'capture/capture.json').read_text())
    assert waits and capture['captured_at']>now-1 and result['neural_observations']==0


def test_exhausted_shared_budget_prevents_capture(snapshot,monkeypatch):
    state,discovery,*_=snapshot;sources={'test':'e'*64}
    monkeypatch.setattr('paperlab.fly_rate_checkpoint.source_hashes',lambda:sources)
    monkeypatch.setattr('paperlab.fly_rate_checkpoint.reserve',lambda *a,**k:None)
    result=execute({'policy':POLICY,'source_sha256':sources},state,discovery,
        call_id='fc-budget',input_id='in-budget',commit=lambda:pytest.fail('no commit needed'))
    assert result=={'status':'budget_stopped'} and not (state/DIRECTORY).exists()


def test_observer_uses_same_handle_after_timeout(tmp_path,monkeypatch):
    import modal
    from paperlab.fly_rate_checkpoint_cloud import observe
    atomic_json(tmp_path/'request.json',{})
    atomic_json(tmp_path/'cloud-call.json',{'call_id':'fc-existing'})
    seen=[]
    class Call:
        def get(self,timeout):
            assert timeout==50;raise modal.exception.TimeoutError()
    def from_id(value):seen.append(value);return Call()
    monkeypatch.setattr(modal.FunctionCall,'from_id',from_id)
    for _ in range(2):assert observe(tmp_path)['status']=='pending'
    assert seen==['fc-existing','fc-existing']
    assert not (tmp_path/'submission.json').exists()


def test_completed_observer_download_uses_state_namespace_and_never_resubmits(tmp_path,monkeypatch):
    import modal
    import paperlab.fly_rate_checkpoint_cloud as client
    from paperlab.fly_market_study import signature
    request={'policy':POLICY};receipt={'call_id':'fc-saved'}
    result={'status':'rate_checkpoint_exported','call_id':'fc-saved','input_id':'in-saved',
            'request_sha256':signature(request),'remote_path':'/state/'+DIRECTORY,
            'neural_observations':0,'paper_orders':0,'native_reference_constructions':1,
            'artifact_sha256':{name:'a'*64 for name in client.ARTIFACTS},
            'budget':{'monthly_limit_usd':25,'nonpreemptible':True,'price_multiplier':3,'reserved_usd':.1608936}}
    atomic_json(tmp_path/'request.json',request);atomic_json(tmp_path/'cloud-call.json',receipt)
    atomic_json(tmp_path/'cloud-result.json',result)
    monkeypatch.setattr(modal.FunctionCall,'from_id',lambda *a:pytest.fail('saved result needs no RPC'))
    monkeypatch.setattr(modal.Volume,'from_name',lambda *a,**kw:object())
    async def download(volume,remote,destination,manifest,**kw):
        assert remote=='/state/'+DIRECTORY and destination==tmp_path/'artifacts'
        assert manifest==result['artifact_sha256'];return {'status':'synthetic_download'}
    monkeypatch.setattr(client,'transfer_files',download)
    monkeypatch.setattr(client,'verify_download',lambda *a:{'status':'synthetic_verified'})
    assert client.observe(tmp_path)=={'status':'synthetic_verified'}
    assert json.loads((tmp_path/'download.json').read_text())['status']=='synthetic_download'


@pytest.mark.parametrize('field',['call_id','request_sha256','remote_path','manifest','budget','orders'])
def test_observer_rejects_wrong_terminal_ownership_manifest_or_budget(field):
    from paperlab.fly_market_study import signature
    from paperlab.fly_rate_checkpoint_cloud import ARTIFACTS,terminal_result
    request={'policy':POLICY};receipt={'call_id':'fc-a'}
    result={'status':'rate_checkpoint_exported','call_id':'fc-a','input_id':'in-a',
            'request_sha256':signature(request),'remote_path':'/state/'+DIRECTORY,
            'neural_observations':0,'paper_orders':0,'native_reference_constructions':1,
            'artifact_sha256':{name:'a'*64 for name in ARTIFACTS},
            'budget':{'monthly_limit_usd':25,'nonpreemptible':True,'price_multiplier':3,'reserved_usd':.1608936}}
    terminal_result(result,request,receipt)
    if field=='manifest':result['artifact_sha256'].pop('export/audit.json')
    elif field=='budget':result['budget']['monthly_limit_usd']=100
    elif field=='orders':result['paper_orders']=1
    else:result[field]='wrong'
    with pytest.raises(ValueError):terminal_result(result,request,receipt)
