"""Carry completed study 11 audit evidence to a later cloud assay.

This transports the existing audits; it does not redo the raw-recording audit,
submit a job, or promote a policy. No fixture or incomplete study can be packed.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import stat
import tempfile

from .fly_online_protocol import chunk_name, chunk_order
from .fly_selective_trace import require_completed_study

SCHEMA = 'study-11-completed-evidence-v1'
FILES = ('report.json', 'plan.json', 'price-audit.json', *(
    path for chunk in map(lambda c: chunk_name(*c), chunk_order()) for path in (
        f'chunks/{chunk}/summary.json', f'audits/{chunk}.json',
        f'receipts/{chunk}-completed.json')))
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_TOTAL_BYTES = 128 * 1024 * 1024
MAX_BUNDLE_BYTES = 192 * 1024 * 1024


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def read_regular(path, limit):
    path = Path(path)
    if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode):
        raise ValueError('Evidence must be a regular file: ' + str(path))
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError('Evidence exceeds the byte limit: ' + str(path))
    return raw


def read_members(root):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('Evidence root must be an existing directory, not a symlink')
    members = {}; total = 0
    for name in FILES:
        relative = Path(name)
        if any((root / p).is_symlink() for p in relative.parents):
            raise ValueError('Evidence directory cannot be a symlink: ' + name)
        raw = read_regular(root / name, MAX_FILE_BYTES)
        total += len(raw)
        if total > MAX_TOTAL_BYTES:
            raise ValueError('Evidence exceeds the total byte limit')
        members[name] = raw
    return members


def write_members(root, members):
    # Iterate only the fixed names, never caller-supplied extraction paths.
    for name in FILES:
        path = Path(root) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(members[name])


def verify_snapshot(members):
    # Validate the copied bytes, so a changing observer directory cannot produce
    # a bundle whose files differ from the evidence that passed the gate.
    with tempfile.TemporaryDirectory(prefix='fly-study-evidence-') as temporary:
        write_members(temporary, members)
        return require_completed_study(temporary)


def result(raw, members):
    return {'status': 'completed_study_evidence_verified', 'files': len(members),
            'raw_bytes': sum(map(len, members.values())),
            'bundle_sha256': sha256(raw),
            'report_sha256': sha256(members['report.json']),
            'neural_observations': 0, 'cloud_submissions': 0}


def pack(root, output):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError('Preserve the existing evidence bundle')
    members = read_members(root)
    report_sha = verify_snapshot(members)
    if report_sha != sha256(members['report.json']):
        raise ValueError('Release gate checked a different report')
    payload = {'schema': SCHEMA, 'report_sha256': report_sha, 'files': {
        name: {'sha256': sha256(raw), 'base64': base64.b64encode(raw).decode('ascii')}
        for name, raw in members.items()}}
    raw = (json.dumps(payload, sort_keys=True, separators=(',', ':')) + '\n').encode()
    if len(raw) > MAX_BUNDLE_BYTES:
        raise ValueError('Bundle exceeds the byte limit')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        stream.write(raw)
    return result(raw, members)


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('Duplicate JSON key in evidence bundle')
        value[key] = item
    return value


def unpack(bundle, output, expected_sha256):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError('Preserve the existing evidence directory')
    raw = read_regular(bundle, MAX_BUNDLE_BYTES)
    if sha256(raw) != expected_sha256:
        raise ValueError('Bundle differs from the expected SHA-256')
    payload = json.loads(raw, object_pairs_hook=unique_object)
    if (not isinstance(payload, dict) or set(payload) != {'schema', 'report_sha256', 'files'}
            or payload['schema'] != SCHEMA or not isinstance(payload['files'], dict)
            or set(payload['files']) != set(FILES)):
        raise ValueError('Expected exactly the completed study evidence files')
    members = {}; total = 0
    for name in FILES:
        item = payload['files'][name]
        if (not isinstance(item, dict) or set(item) != {'sha256', 'base64'}
                or not isinstance(item['base64'], str)
                or len(item['base64']) > 4 * ((MAX_FILE_BYTES + 2) // 3)):
            raise ValueError('Invalid or oversized evidence member: ' + name)
        decoded = base64.b64decode(item['base64'], validate=True)
        total += len(decoded)
        if len(decoded) > MAX_FILE_BYTES or total > MAX_TOTAL_BYTES:
            raise ValueError('Decoded evidence exceeds the byte limit')
        if sha256(decoded) != item['sha256']:
            raise ValueError('Evidence member SHA-256 differs: ' + name)
        members[name] = decoded
    if verify_snapshot(members) != payload['report_sha256']:
        raise ValueError('Release gate checked a different report')
    # Exclusive directory creation preserves even an existing empty destination.
    output.mkdir(parents=True, exist_ok=False)
    write_members(output, members)
    if require_completed_study(output) != payload['report_sha256']:
        raise ValueError('Installed evidence differs from its verified snapshot')
    return result(raw, members)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    export = commands.add_parser('pack')
    export.add_argument('--study', type=Path, required=True)
    export.add_argument('--out', type=Path, required=True)
    restore = commands.add_parser('unpack')
    restore.add_argument('--bundle', type=Path, required=True)
    restore.add_argument('--sha256', required=True)
    restore.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = (pack(args.study, args.out) if args.command == 'pack' else
              unpack(args.bundle, args.out, args.sha256))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
