"""Exercise the read-only handoff, timeout recovery and failed evidence gates."""
import json
import fcntl
from types import SimpleNamespace

import pytest

from paperlab import fly_online_watch as watcher


@pytest.fixture
def clock(monkeypatch):
    value = SimpleNamespace(now=0., sleeps=[])
    def sleep(seconds):
        assert 0 <= seconds <= 60
        value.sleeps.append(seconds)
        value.now += seconds
    monkeypatch.setattr(watcher.time, 'monotonic', lambda: value.now)
    monkeypatch.setattr(watcher.time, 'time', lambda: 1_800_000_000 + value.now)
    monkeypatch.setattr(watcher.time, 'sleep', sleep)
    return value


@pytest.fixture
def remote(monkeypatch, clock):
    import modal
    state = SimpleNamespace(owner={'call_id': 'fc-current'}, lookups=[], gets=[], replies=[])
    class Call:
        def __init__(self, identity):
            self.identity = identity
        def get(self, timeout):
            state.gets.append((self.identity, timeout))
            reply = state.replies.pop(0) if state.replies else TimeoutError()
            if isinstance(reply, TimeoutError):
                clock.now += timeout
            if isinstance(reply, Exception):
                raise reply
            return reply
    class Writers:
        def get(self, key):
            state.lookups.append(key)
            return state.owner
    monkeypatch.setattr(modal.Dict, 'from_name', lambda *a, **kw: Writers())
    monkeypatch.setattr(modal.FunctionCall, 'from_id', Call)
    monkeypatch.setattr(modal.Function, 'from_name',
                        lambda *a, **kw: pytest.fail('Watcher cannot submit a job'))
    return state


def progress(status, count=0, **kwargs):
    return {'status': status, 'completed_chunks': count, 'total_chunks': 16, **kwargs}


def test_waits_for_all_capture_before_one_full_audit(tmp_path, monkeypatch, clock, remote):
    observations = [progress('test_collection'), progress('evaluation', 15),
                    progress('captured_pending_audit', 16)]
    calls = []
    def observe(registration, output, **kwargs):
        calls.append(kwargs)
        if kwargs.get('audit'):
            return progress('paper_online_study_audited', 16, audited_chunks=16)
        return observations.pop(0)
    monkeypatch.setattr(watcher, 'observe', observe)
    result = watcher.watch({}, tmp_path, 'prepared-graph', poll_seconds=10, timeout=60)
    assert result['status'] == 'completed'
    assert calls == [{}, {}, {}, {'audit': True, 'data': 'prepared-graph'}]
    assert remote.lookups == ['collector', 'worker']
    assert json.loads((tmp_path/'watch.json').read_text()) == result


def test_observation_timeout_then_resume_uses_same_pending_call(tmp_path, monkeypatch, clock, remote):
    monkeypatch.setattr(watcher, 'observe',
                        lambda *a, **kw: progress('chunk_call_pending', 4, call_id='fc-pinned'))
    for _ in range(2):
        result = watcher.watch({}, tmp_path, 'graph', poll_seconds=10, timeout=20)
        assert result['status'] == 'observation_timeout'
        assert result['progress']['status'] == 'chunk_call_pending'
    assert remote.gets == [('fc-pinned', 10)] * 4
    assert not remote.lookups


def test_sealing_uses_its_receipt_instead_of_current_owner(remote, clock):
    result = watcher.wait_existing(progress('sealing_call_pending',
                                  receipt={'call_id': 'fc-seal'}), 12)
    assert result == {'state': 'pending', 'call_id': 'fc-seal'}
    assert remote.gets == [('fc-seal', 12)] and not remote.lookups


def test_between_calls_wait_does_not_create_work(remote, clock):
    remote.owner = None
    result = watcher.wait_existing(progress('evaluation'), 8)
    assert result == {'state': 'between_calls', 'call_id': None}
    assert clock.now == 8 and not remote.gets


def test_completed_owner_does_not_spin(tmp_path, monkeypatch, clock, remote):
    remote.replies = [{}, {}]
    monkeypatch.setattr(watcher, 'observe', lambda *a, **kw: progress('evaluation'))
    result = watcher.watch({}, tmp_path, 'graph', poll_seconds=10, timeout=20)
    assert result['status'] == 'observation_timeout'
    assert len(remote.gets) == 2 and clock.sleeps == [10, 10]


@pytest.mark.parametrize('status', ['registration_missed', 'chunk_receipt_unresolved',
                                    'sealing_receipt_unresolved', 'unknown'])
def test_unresolved_or_unrecognized_state_stops(tmp_path, monkeypatch, clock, remote, status):
    monkeypatch.setattr(watcher, 'observe', lambda *a, **kw: progress(status))
    with pytest.raises(ValueError, match='requires inspection'):
        watcher.watch({}, tmp_path, 'graph')
    saved = json.loads((tmp_path/'watch.json').read_text())
    assert saved['status'] == 'observation_failed' and saved['progress']['status'] == status
    assert not remote.gets


def test_remote_failure_is_not_retried_or_replaced(tmp_path, monkeypatch, clock, remote):
    remote.replies = [RuntimeError('worker failed')]
    monkeypatch.setattr(watcher, 'observe',
                        lambda *a, **kw: progress('chunk_call_pending', call_id='fc-failed'))
    with pytest.raises(RuntimeError, match='worker failed'):
        watcher.watch({}, tmp_path, 'graph')
    assert remote.gets == [('fc-failed', 45)] and not remote.lookups
    assert json.loads((tmp_path/'watch.json').read_text())['error']['message'] == 'worker failed'


def test_incomplete_capture_cannot_enter_audit(tmp_path, monkeypatch, clock):
    def observe(*args, **kwargs):
        assert not kwargs.get('audit')
        return progress('captured_pending_audit', 15)
    monkeypatch.setattr(watcher, 'observe', observe)
    with pytest.raises(ValueError, match='sixteen'):
        watcher.watch({}, tmp_path, 'graph')


@pytest.mark.parametrize('failure', [ValueError('corrupt raw array'),
                                    progress('paper_online_study_audited', 16, audited_chunks=15)])
def test_failed_or_incomplete_audit_never_publishes_completion(tmp_path, monkeypatch, clock, failure):
    def observe(*args, **kwargs):
        if not kwargs.get('audit'):
            return progress('captured_pending_audit', 16)
        if isinstance(failure, Exception):
            raise failure
        return failure
    monkeypatch.setattr(watcher, 'observe', observe)
    with pytest.raises(ValueError):
        watcher.watch({}, tmp_path, 'graph')
    assert json.loads((tmp_path/'watch.json').read_text())['status'] == 'observation_failed'


def test_local_interruption_is_not_a_cloud_failure(tmp_path, monkeypatch, clock, remote):
    monkeypatch.setattr(watcher, 'observe', lambda *a, **kw: progress('test_collection'))
    def interrupt(*args):
        raise KeyboardInterrupt()
    monkeypatch.setattr(watcher, 'wait_existing', interrupt)
    with pytest.raises(KeyboardInterrupt):
        watcher.watch({}, tmp_path, 'graph')
    saved = json.loads((tmp_path/'watch.json').read_text())
    assert saved['status'] == 'observation_interrupted'
    assert saved['progress']['status'] == 'test_collection' and not remote.gets


def test_second_watcher_cannot_race_downloads(tmp_path, monkeypatch):
    monkeypatch.setattr(watcher, 'observe', lambda *a, **kw: pytest.fail('No concurrent observer'))
    with (tmp_path/'watch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match='already owns'):
            watcher.watch({}, tmp_path, 'graph')


@pytest.mark.parametrize('poll,timeout', [(0, 10), (61, 10), (float('nan'), 10),
                                        (45, 0), (45, float('inf'))])
def test_invalid_waits_rejected_before_network(tmp_path, monkeypatch, poll, timeout):
    monkeypatch.setattr(watcher, 'observe', lambda *a, **kw: pytest.fail('No network access'))
    with pytest.raises(ValueError, match='finite'):
        watcher.watch({}, tmp_path, 'graph', poll_seconds=poll, timeout=timeout)
