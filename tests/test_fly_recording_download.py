"""Read deadlines preserve existing files and never become compute retries."""
import asyncio
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from paperlab.fly_recording_download import transfer


@pytest.fixture
def recording(tmp_path):
    name = 'development-pool1-trained_online_carry'
    data = {'trace/step-01.npz': b'complete first recording', 'trace/step-02.npz': b'complete second recording'}
    manifest = {k: hashlib.sha256(v).hexdigest() for k, v in data.items()}
    receipt = {'status': 'completed', 'chunk': name, 'call_id': 'fixture-completed-call',
        'remote_path': '/state/registered-paper-11-completion-01/chunks/'+name+'/artifacts', 'plan_sha256': 'a'*64}
    summary = {'chunk': name, 'status': 'paper_online_chunk_completed', 'plan_sha256': 'a'*64, 'artifact_sha256': manifest}
    return tmp_path, data, receipt, summary


def volume(data, *, hang=0, corrupt=False):
    calls = []; closed = []
    async def read(path):
        relative = path.split('/artifacts/')[1]; calls.append(relative)
        try:
            if len(calls) <= hang:
                yield b'unverified partial bytes'
                await asyncio.Event().wait()
            yield b'wrong bytes' if corrupt else data[relative]
        finally:
            closed.append(relative)
    return SimpleNamespace(read_file=SimpleNamespace(aio=read), calls=calls, closed=closed)


def test_stalled_read_is_cancelled_then_replaced_by_verified_complete_bytes(recording):
    root, data, receipt, summary = recording; v = volume(data, hang=1)
    result = asyncio.run(transfer(v, receipt, summary, root, timeout=.01, files_at_once=1))
    assert result['files'] == 2 and result['cloud_submissions'] == 0 and result['new_neural_observations'] == 0
    assert len(result['retries']) == 1 and len(v.calls) == len(v.closed) == 3
    assert not list(root.rglob('*.partial'))
    for name, raw in data.items(): assert (root/name).read_bytes() == raw
    again = asyncio.run(transfer(v, receipt, summary, root, timeout=.01, files_at_once=1))
    assert again['files'] == 2 and len(v.calls) == 3  # Reuse every verified file.


def test_exhausted_timeout_never_publishes_a_partial_file(recording):
    root, data, receipt, summary = recording; v = volume(data, hang=99)
    summary['artifact_sha256'] = {k: val for k, val in list(summary['artifact_sha256'].items())[:1]}
    with pytest.raises(TimeoutError, match='exhausted'):
        asyncio.run(transfer(v, receipt, summary, root, timeout=.01, attempts=2, files_at_once=1))
    assert len(v.calls) == len(v.closed) == 2
    assert not (root/'trace/step-01.npz').exists()
    assert (root/'trace/step-01.npz.partial').exists()


@pytest.mark.parametrize('change', ['existing', 'downloaded', 'path', 'symlink', 'claimed', 'namespace'])
def test_invalid_or_conflicting_data_is_not_published(recording, change):
    root, data, receipt, summary = recording; v = volume(data, corrupt=change=='downloaded')
    target = root/'trace/step-01.npz'
    if change == 'existing':
        target.parent.mkdir(); target.write_bytes(b'preserve these bytes')
    if change == 'path': summary['artifact_sha256']['../escape'] = 'a'*64
    if change == 'symlink': (root/'trace').symlink_to(root.parent, target_is_directory=True)
    if change == 'claimed': receipt['status'] = 'claimed'
    if change == 'namespace': receipt['remote_path'] = '/state/unrelated/private'
    with pytest.raises(ValueError): asyncio.run(transfer(v, receipt, summary, root, timeout=.01, files_at_once=1))
    if change == 'existing': assert target.read_bytes() == b'preserve these bytes'
    else: assert not target.exists()
