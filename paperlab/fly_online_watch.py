"""Observe scheduled study 11 until capture finishes, then audit it once.

This local process never submits jobs or changes cloud state. Stopping it does
not stop the scheduled experiment; restarting rechecks the same saved receipts.
"""
import argparse
import fcntl
import json
import math
from pathlib import Path
import time

from .core import atomic_json
from .fly_online_cloud import observe


WAITING = frozenset({
    'waiting_for_cloud_arming', 'arming_call_pending', 'waiting_for_development',
    'development_collection', 'test_collection', 'evaluation',
    'sealing_call_pending', 'chunk_call_pending', 'chunk_worker_return_pending',
})


def wait_existing(progress, seconds):
    """Wait on the exact known call, or inspect the current scheduled owner."""
    import modal
    call_id = progress.get('call_id') or (progress.get('receipt') or {}).get('call_id')
    if call_id is None:
        key = 'collector' if progress['status'] in (
            'development_collection', 'test_collection') else 'worker'
        writers = modal.Dict.from_name('fly-paper-lab-writers', environment_name='main')
        owner = writers.get(key)
        call_id = owner['call_id'] if owner else None
    if call_id is None:
        time.sleep(seconds)
        return {'state': 'between_calls', 'call_id': None}
    try:
        modal.FunctionCall.from_id(call_id).get(timeout=seconds)
        return {'state': 'completed', 'call_id': call_id}
    except TimeoutError:
        return {'state': 'pending', 'call_id': call_id}


def watch(registration, output, data, *, poll_seconds=45, timeout=7200):
    """The timeout bounds observation; an audit already started may finish."""
    if (not math.isfinite(poll_seconds) or not 1 <= poll_seconds <= 60
            or not math.isfinite(timeout) or timeout <= 0):
        raise ValueError('Use a 1-60 second poll interval and a positive finite timeout')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # OS ownership ends on exit, including interruption; there is no age-based
    # removal. Two local observers cannot race on the same partial downloads.
    with (output / 'watch.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Another watcher already owns this output directory') from exc
        return _watch(registration, output, data, poll_seconds=poll_seconds, timeout=timeout)


def _watch(registration, output, data, *, poll_seconds, timeout):
    started = time.monotonic()
    deadline = started + timeout
    previous = None

    def save(status, progress, waiting=None):
        value = {'status': status, 'observed_at': time.time(),
                 'elapsed_seconds': time.monotonic() - started,
                 'progress': progress, 'last_wait': waiting, 'cloud_submissions': 0}
        atomic_json(output / 'watch.json', value)
        return value

    progress = None
    try:
        while True:
            progress = observe(registration, output)
            key = (progress['status'], progress.get('completed_chunks'),
                   progress.get('chunk'), progress.get('call_id'))
            if key != previous:
                print(json.dumps({'event': 'online_watch_progress', 'status': key[0],
                                  'completed_chunks': key[1], 'chunk': key[2],
                                  'call_id': key[3]}), flush=True)
                previous = key
            if progress['status'] == 'captured_pending_audit':
                if progress.get('completed_chunks') != 16 or progress.get('total_chunks') != 16:
                    raise ValueError('Full capture must contain all sixteen chunks')
                save('auditing', progress)
                result = observe(registration, output, audit=True, data=data)
                if (result['status'] != 'paper_online_study_audited'
                        or result.get('audited_chunks') != 16
                        or result.get('completed_chunks') != 16):
                    raise ValueError('Final audit did not verify all sixteen chunks')
                return save('completed', result)
            if progress['status'] not in WAITING:
                raise ValueError('Study requires inspection: ' + progress['status'])
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return save('observation_timeout', progress)
            duration = min(poll_seconds, remaining)
            before_wait = time.monotonic()
            waiting = wait_existing(progress, duration)
            save('observing', progress, waiting)
            # A completed/stale owner must not cause a rapid polling loop. A timeout
            # does not clear that owner, submit a replacement, or mark it failed.
            pause = min(duration - (time.monotonic() - before_wait),
                        deadline - time.monotonic())
            if pause > 0:
                time.sleep(pause)
    except (Exception, KeyboardInterrupt) as exc:
        value = save('observation_interrupted' if isinstance(exc, KeyboardInterrupt)
                     else 'observation_failed', progress)
        value['error'] = {'type': type(exc).__name__, 'message': str(exc)[:300]}
        atomic_json(output / 'watch.json', value)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('registration', 'out', 'fly-data'):
        parser.add_argument('--' + field, type=Path, required=True)
    parser.add_argument('--poll-seconds', type=float, default=45)
    parser.add_argument('--timeout', type=float, default=7200,
                        help='Observation time limit in seconds; does not cancel cloud work')
    args = parser.parse_args()
    result = watch(json.loads(args.registration.read_text()), args.out, args.fly_data,
                   poll_seconds=args.poll_seconds, timeout=args.timeout)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
