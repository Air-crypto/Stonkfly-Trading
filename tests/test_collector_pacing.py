"""Request-boundary tests: cooldown survives restart and assignments are not starved."""
import asyncio
from dataclasses import replace
from email.utils import formatdate
from types import SimpleNamespace

import pytest
import requests

import paperlab.universe as universe
from test_universe import pool


def clocked_api(monkeypatch, responses):
    clock=SimpleNamespace(now=1700000100.)
    clock.time=lambda:clock.now
    clock.monotonic=lambda:clock.now
    monkeypatch.setattr(universe,"time",clock)
    sleeps=[]
    async def sleep(seconds):
        sleeps.append(seconds)
        clock.now+=seconds
    monkeypatch.setattr(universe.asyncio,"sleep",sleep)
    attempted=[]
    def get(*args,**kwargs):
        attempted.append(clock.now)
        response=responses.pop(0)
        assert kwargs["headers"]["Accept"].endswith("version=20230203")
        return response
    monkeypatch.setattr(universe.requests,"get",get)
    return clock,attempted,sleeps


def response(status=200, retry=None):
    r=requests.Response();r.status_code=status;r._content=b'{"data":[]}'
    if retry is not None:r.headers["Retry-After"]=retry
    return r


def test_pacing_persists_across_restarted_collectors(tmp_path,monkeypatch):
    clock,attempted,_=clocked_api(monkeypatch,[response() for _ in range(9)])
    path=tmp_path/"rate.json"
    for _ in range(9):
        api=universe.PublicAPI(path)
        asyncio.run(api.get("networks/new_pools",deadline=clock.now+100))
    assert all(b-a>=7.5 for a,b in zip(attempted,attempted[1:]))
    assert attempted[-1]-attempted[0]>=60


@pytest.mark.parametrize("retry",["180","date","invalid"])
def test_429_retry_after_and_exponential_backoff_survive_restart(tmp_path,monkeypatch,retry):
    header=formatdate(1700000100+180,usegmt=True) if retry=="date" else retry
    clock,attempted,_=clocked_api(monkeypatch,[response(429,header),response(429),response()])
    path=tmp_path/"rate.json"
    first=universe.PublicAPI(path)
    with pytest.raises(requests.HTTPError):asyncio.run(first.get("x",deadline=clock.now+1000))
    expected=60 if retry=="invalid" else 180
    assert first.state["next_request_at"]==clock.now+expected
    restarted=universe.PublicAPI(path)
    with pytest.raises(requests.HTTPError):asyncio.run(restarted.get("x",deadline=clock.now+1000))
    assert attempted[1]-attempted[0]==expected
    assert restarted.state["consecutive_429"]==2
    assert restarted.state["next_request_at"]==clock.now+120
    before=len(attempted)
    with pytest.raises(universe.CollectionDeadline):
        asyncio.run(universe.PublicAPI(path).get("x",deadline=clock.now+100))
    assert len(attempted)==before  # Do not spend the window waiting for an impossible request.
    recovered=universe.PublicAPI(path)
    asyncio.run(recovered.get("x",deadline=clock.now+200))
    assert recovered.state["consecutive_429"]==0
    assert attempted[-1]-attempted[-2]==120


def test_assigned_batches_precede_discovery_even_if_many_other_pools_share_chain():
    watch=[pool(i) for i in range(100)]
    watch.append(replace(pool(101),key="zchain:pool101",network="zchain"))
    priority={watch[99].key,watch[100].key}
    requests=universe.collection_requests(watch,priority,0)[:universe.REQUESTS_PER_ROUND]
    assert requests[0][0].endswith('/pool99')
    assert requests[1][0].endswith('/pool101')
    assert requests[2][0]=='networks/new_pools'
    assert len(requests)<=8


def test_background_chains_rotate_with_smaller_budget_and_priority_retained():
    watch=[replace(pool(i),key=f"chain{i}:pool{i}",network=f"chain{i}") for i in range(20)]
    priority={watch[-1].key};seen=set()
    for minute in range(20):
        paths=[p for p,_ in universe.collection_requests(watch,priority,minute)[:8]]
        assert paths[0].startswith('networks/chain19/pools/multi/')
        seen.update(p.split('/')[1] for p in paths if '/pools/multi/' in p)
    assert seen=={p.network for p in watch}


def test_priority_health_distinguishes_fresh_rejection_missing_and_stale(tmp_path):
    store=universe.Store(tmp_path/'universe.db');p=pool();bad=replace(pool(1),volume5=0)
    store.add([p,bad])
    fresh=store.priority_health([p.key,bad.key,'solana:absent'],p.observed+30)
    assert {r['pool']:r['rejection'] for r in fresh}=={
        p.key:None,bad.key:'insufficient_two_sided_activity','solana:absent':'missing_pool'}
    assert store.priority_health([p.key],p.observed+181)[0]['rejection']=='stale_observation'
    store.db.close()


def test_rate_limit_ends_round_and_recovery_starts_with_assigned_pool(tmp_path,monkeypatch):
    clock,_,_=clocked_api(monkeypatch,[])
    store=universe.Store(tmp_path/'universe.db');p=pool();store.add([p]);store.db.close()
    attempted=[]
    class API:
        def __init__(self,path):
            self.calls=0;self.state={'next_request_at':0,'consecutive_429':0}
        async def get(self,path,deadline,**params):
            attempted.append(path);self.calls+=1
            if len(attempted)==1:
                self.state.update(next_request_at=clock.now+60,consecutive_429=1)
                response(429).raise_for_status()
            clock.now=deadline-10
            return {'data':[]}
    async def stream(*args):
        await asyncio.Future()
    monkeypatch.setattr(universe,'PublicAPI',API)
    monkeypatch.setattr(universe,'launch_stream',stream)
    result=asyncio.run(universe.collect_window(tmp_path,seconds=130,priorities=lambda:[p.key]))
    assert len(attempted)==2
    assert attempted[0]==attempted[1]==f'networks/solana/pools/multi/{p.address}'
    assert result['window_requests_attempted']==2
    assert result['window_requests_completed']==1
    assert result['window_exhausted']
