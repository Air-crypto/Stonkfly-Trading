"""Download completed study 11 recordings with bounded read-only transfers.

Only a saved completed receipt and its matching returned call result authorize a
download. Network timeouts may repeat reads, never model calls or neural work.
"""
import argparse
import asyncio
from contextlib import aclosing
import json
from pathlib import Path
import time

from .core import atomic_json, digest
from .fly_online_protocol import chunk_name, chunk_order

NAMES = {chunk_name(*c) for c in chunk_order()}
NAMESPACES = {'registered-paper-11', 'registered-paper-11-recovery-02', 'registered-paper-11-completion-01'}


def safe_target(root, name):
    path = Path(name)
    if path.is_absolute() or not path.parts or '..' in path.parts:
        raise ValueError('Unsafe artifact path')
    target = root/path
    if any(p.is_symlink() for p in (target, *target.parents)):
        raise ValueError('Artifact destination cannot contain symlinks')
    return target


async def transfer(volume, receipt, summary, destination, *, timeout=90, attempts=3, files_at_once=4):
    """Authorize a completed study recording before any artifact reads."""
    name = receipt['chunk']; remote = receipt['remote_path']
    allowed = {f'/state/{ns}/chunks/{name}/artifacts' for ns in NAMESPACES}
    if (name not in NAMES or receipt['status'] != 'completed' or remote not in allowed
            or summary['chunk'] != name or summary['status'] != 'paper_online_chunk_completed'
            or summary['plan_sha256'] != receipt['plan_sha256']):
        raise ValueError('Expected a completed study 11 recording')
    result = await transfer_files(volume, remote, destination, summary['artifact_sha256'],
        label=name, timeout=timeout, attempts=attempts, files_at_once=files_at_once)
    return {**result, 'chunk': name, 'call_id': receipt['call_id']}


async def transfer_files(volume, remote, destination, manifest, *, label,
                         timeout=90, attempts=3, files_at_once=4):
    """Bounded reads of an already-authorized, exact artifact manifest.

    The caller must verify completion and ownership before invoking this helper.
    It has no model submission API. All outstanding reads are cancelled and
    closed before a failed transfer returns to its caller.
    """
    if not 0 < timeout <= 180 or not 1 <= attempts <= 3 or not 1 <= files_at_once <= 4:
        raise ValueError('Invalid transfer limits')
    if (not isinstance(remote, str) or not remote.startswith('/state/')
            or '..' in Path(remote).parts):
        raise ValueError('Invalid artifact remote path')
    destination = Path(destination); semaphore = asyncio.Semaphore(files_at_once)
    for file, sha in manifest.items():
        safe_target(destination, file)
        if not isinstance(sha, str) or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
            raise ValueError('Invalid artifact hash')
    retries = []; started = time.monotonic()

    async def download(file, sha):
        path = safe_target(destination, file)
        if path.exists():
            if digest(path) != sha:
                raise ValueError('Preserve differing existing artifact: ' + file)
            return path.stat().st_size
        partial = safe_target(destination, file+'.partial')
        async with semaphore:
            path.parent.mkdir(parents=True, exist_ok=True)
            for attempt in range(1, attempts+1):
                try:
                    async with asyncio.timeout(timeout):
                        with partial.open('wb') as stream:
                            async with aclosing(volume.read_file.aio(remote.removeprefix('/state')+'/'+file)) as blocks:
                                async for block in blocks:
                                    stream.write(block)
                    break
                except TimeoutError:
                    retries.append({'file': file, 'attempt': attempt, 'reason': 'read_timeout'})
                    print(json.dumps({'event': 'artifact_read_timeout', 'recording': label,
                        'file': file, 'attempt': attempt, 'retrying': attempt < attempts}), flush=True)
                    if attempt == attempts:
                        raise TimeoutError('Artifact transfer exhausted bounded read attempts: '+file) from None
            if digest(partial) != sha:
                raise ValueError('Downloaded artifact hash differs: ' + file)
            partial.replace(path)
            return path.stat().st_size

    tasks = [asyncio.create_task(download(file, sha)) for file, sha in manifest.items()]
    try:
        sizes = await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done(): task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    return {'status': 'all_artifacts_downloaded_and_hash_verified',
        'files': len(sizes), 'bytes': sum(sizes),
        'seconds': time.monotonic()-started, 'read_timeout_seconds': timeout,
        'maximum_read_attempts': attempts, 'concurrent_files': files_at_once, 'retries': retries,
        'new_neural_observations': 0, 'cloud_submissions': 0, 'source_sha256': digest(__file__)}


def download(root, name):
    import modal
    root = Path(root)
    if name not in NAMES:
        raise ValueError('Unknown study 11 condition')
    receipt = json.loads((root/'receipts'/(name+'-completed.json')).read_text())
    result = json.loads((root/'call-results'/(name+'.json')).read_text())
    returned = result.get('study_11', result)
    if returned.get('receipt') != receipt or returned.get('status') not in (
            'paper_online_chunk_completed', 'recovery_chunk_completed', 'completion_chunk_completed'):
        raise ValueError('Saved completed call result differs from its receipt')
    summary_path = root/'summaries'/(name+'.json')
    if not summary_path.exists():
        summary_path = root/'chunks'/name/'summary.json'
    if digest(summary_path) != receipt['summary_sha256']:
        raise ValueError('Saved summary differs from completed receipt')
    summary = json.loads(summary_path.read_text()); out = root/'chunks'/name
    if out.is_symlink():
        raise ValueError('Recording root cannot be a symlink')
    out.mkdir(parents=True, exist_ok=True)
    target = safe_target(out, 'summary.json')
    if target.exists() and target.read_bytes() != summary_path.read_bytes():
        raise ValueError('Preserve differing recording summary')
    target.write_bytes(summary_path.read_bytes())
    volume = modal.Volume.from_name('fly-paper-lab-state', environment_name='main')
    report = asyncio.run(transfer(volume, receipt, summary, out))
    atomic_json(root/'downloads'/(name+'.json'), report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--chunk', required=True)
    args = parser.parse_args()
    print(json.dumps(download(args.root, args.chunk), indent=2))


if __name__ == '__main__': main()
