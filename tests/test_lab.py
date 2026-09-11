from dataclasses import asdict
import json
from pathlib import Path
import sqlite3

import numpy as np
import pytest
import torch

from paperlab.compact import Policy, load_policy, train
from paperlab.core import Broker, Costs, Environment, Tick, features, load_ticks
from paperlab.data import synthetic
from paperlab.news import News
from paperlab.runtime import cycle


def test_news_respects_publication_first_seen_and_encoding():
    news = News()
    news.add("a", "ETF approved", "test", 100, 200, encoded=300)
    assert not news.features(199).any()
    assert not news.features(299).any()
    assert news.features(300)[48] > 0
    news.add("b", "future publication", "test", 500, 200, encoded=300)
    assert len(news.rows(400)) == 1
    assert len(news.rows(500)) == 2


def test_news_revisions_are_causal_and_duplicates_do_not_refresh():
    news = News()
    news.add("a", "ETF approved", "test", 100, 100)
    first = news.features(150).copy()
    news.add("a", "ETF approved", "test", 100, 300)
    news.add("a", "ETF fraud", "test", 100, 300)
    assert np.array_equal(first, news.features(150))
    assert news.features(350)[1] > news.features(350)[0]
    assert np.isclose(news.features(350)[48], np.log(2))


def test_news_expires_and_can_be_ablated():
    news = News()
    news.add("a", "ETF approved", "test", 100, 100)
    assert not news.features(100000).any()
    news.enabled = False
    assert not news.features(100).any()


def test_execution_requires_later_tick_and_expires():
    broker = Broker()
    assert broker.execute(.5, 100, Tick(100, 100, 100))["reason"] == "not_after_decision"
    assert broker.execute(.5, 100, Tick(5000, 100, 100))["reason"] == "expired_decision"
    assert broker.qty == 0


def test_roundtrip_costs_and_inventory():
    broker = Broker(Costs(max_order=1000))
    a = broker.execute(.5, 100, Tick(200, 99, 101))
    assert a["reason"] == "spread"
    a = broker.execute(.5, 100, Tick(200, 100, 100))
    assert a["status"] == "filled"
    assert broker.equity(Tick(200, 100, 100)) < 1000
    b = broker.execute(0, 200, Tick(300, 100, 100))
    assert b["side"] == "SELL" and broker.qty == 0
    assert float(broker.cash) < 1000 and broker.fees > 0


def test_guard_rejects_risk_without_substitute_trade():
    broker = Broker(Costs(loss_stop=.1, max_order=1000))
    broker.execute(.5, 100, Tick(200, 100, 100))
    qty = broker.qty
    result = broker.execute(.5, 200, Tick(300, 1, 1))
    assert result["reason"] == "loss_stop" and broker.qty == qty
    assert broker.execute(0, 300, Tick(400, 1, 1))["side"] == "SELL"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -.1, 1.1])
def test_invalid_target(value):
    with pytest.raises(ValueError):
        Broker().execute(value, 100, Tick(200, 100, 100))


def test_state_roundtrip():
    a = Broker()
    a.execute(.5, 100, Tick(200, 100, 100))
    b = Broker(state=a.state())
    assert a.state() == b.state()


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "data.jsonl"
    synthetic(path, count=450)
    return load_ticks(path)


def test_observation_cannot_see_future(dataset):
    news = News()
    before = features(dataset, 70, Broker(), news)
    altered = list(dataset)
    altered[71] = Tick(dataset[71].ts, 1, 1)
    assert np.array_equal(before, features(altered, 70, Broker(), news))
    assert before.shape == (64,)


def test_environment_uses_next_observation(dataset):
    env = Environment(dataset, News(), Costs(), 63, 70)
    _, _, _, info = env.step(2)
    assert info["fill"]["fill_ts"] == dataset[64].ts
    assert info["fill"]["decision_ts"] == dataset[63].ts


def test_ppo_updates_weights_and_seals_model_before_test(tmp_path, dataset):
    torch.manual_seed(7)
    before = Policy().actor.weight.detach().clone()
    result = train(dataset, News(), tmp_path / "run", steps=256)
    policy, meta = load_policy(tmp_path / "run/policy.pt")
    assert result["parameters"] == 247780
    assert not torch.equal(before, policy.actor.weight)
    assert meta["train_until"] < dataset[int(.7 * len(dataset))].ts
    assert result["splits"]["test"]["cash"]["return_pct"] == 0
    assert (tmp_path / "run/test-ppo-ledger.json").exists()


def test_runtime_restart_duplicate_and_rollback(tmp_path, monkeypatch):
    for i in range(64):
        tick = Tick(1700000000 + i * 900, 100, 100, source="forward_rest_book")
        cycle(tmp_path, tick=tick, ingest=False)
    duplicate = cycle(tmp_path, tick=tick, ingest=False)
    assert duplicate["status"] == "duplicate_or_old_slot"
    db = sqlite3.connect(tmp_path / "paper.db")
    assert db.execute("SELECT count(*) FROM ticks").fetchone()[0] == 64
    assert db.execute("SELECT count(*) FROM ledger").fetchone()[0] == 1
    assert json.loads(db.execute("SELECT payload FROM state WHERE name='compact'").fetchone()[0])["broker"]["cash"] == "1000"
    db.close()
    # A corrupt checkpoint raises; the quote and accounting transaction is rolled back.
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"not a model")
    with pytest.raises(Exception):
        cycle(tmp_path, tick=Tick(tick.ts + 900, 100, 100, source="forward_rest_book"), ingest=False, model=bad)
    db = sqlite3.connect(tmp_path / "paper.db")
    assert db.execute("SELECT count(*) FROM ticks").fetchone()[0] == 64
    db.close()


def test_runtime_rejects_config_drift(tmp_path):
    cycle(tmp_path, tick=Tick(1700000000, 100, 100, source="forward_rest_book"), ingest=False)
    with pytest.raises(ValueError, match="configuration changed"):
        cycle(tmp_path, product="ETH-USD", tick=Tick(1700000900, 100, 100, product="ETH-USD", source="forward_rest_book"), ingest=False)


def test_news_parse_failure_is_visible(monkeypatch):
    class Response:
        def raise_for_status(self):
            pass
        def iter_content(self, size):
            yield b'<!DOCTYPE x [<!ENTITY x "EXPAND">]><rss>&x;</rss>'
    monkeypatch.setattr("paperlab.news.requests.get", lambda *a, **k: Response())
    report = News().ingest(["https://example.com/feed"])
    assert report[0]["status"] == "error"


def test_runtime_executes_persisted_decision_only_on_next_slot(tmp_path, monkeypatch):
    for i in range(63):
        cycle(tmp_path, tick=Tick(1700000000 + i * 900, 100, 100, source="forward_rest_book"), ingest=False)
    model = Policy()
    with torch.no_grad():
        model.actor.weight.zero_()
        model.actor.bias.copy_(torch.tensor([-1., -1., 1.]))
    path = tmp_path / "model.pt"
    torch.save({"schema": "64-market14-news50-v1", "model": model.state_dict(), "costs": asdict(Costs()), "product": "BTC-USD", "source": "synthetic"}, path)
    now = 1700000000 + 63 * 900
    monkeypatch.setattr("paperlab.runtime.time.time", lambda: now + 2)
    first = cycle(tmp_path, tick=Tick(now, 100, 100, source="forward_rest_book"), ingest=False, model=path)
    assert first["policies"]["compact"]["fill"]["status"] == "hold"
    now += 900
    second = cycle(tmp_path, tick=Tick(now, 100, 100, source="forward_rest_book"), ingest=False, model=path)
    fill = second["policies"]["compact"]["fill"]
    assert fill["status"] == "filled" and fill["fill_ts"] > fill["decision_ts"]


def test_budget_reserves_before_work_and_releases_unused(tmp_path):
    from paperlab.budget import reserve, settle
    from datetime import datetime, timezone
    stamp = datetime(2026, 9, 11, tzinfo=timezone.utc).timestamp()
    path = tmp_path / "budget.json"
    reservation = reserve(path, True, now=stamp)
    assert reservation["limit"] == 100
    before = json.loads(path.read_text())["months"]["2026-09"]
    settle(path, reservation, 2)
    assert json.loads(path.read_text())["months"]["2026-09"] < before
    stamp = datetime(2026, 10, 1, tzinfo=timezone.utc).timestamp()
    assert reserve(path, True, now=stamp)["limit"] == 40


def test_budget_stops_new_work(tmp_path):
    from paperlab.budget import reserve
    from datetime import datetime, timezone
    path = tmp_path / "budget.json"
    path.write_text(json.dumps({"first_month": "2026-09", "months": {"2026-09": 75}}))
    stamp = datetime(2026, 9, 11, tzinfo=timezone.utc).timestamp()
    assert reserve(path, True, now=stamp) is None
