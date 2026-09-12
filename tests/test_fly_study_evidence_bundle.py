"""Transport tests use labeled fixtures; the production gate rejects them."""
import base64
import json
from pathlib import Path

import pytest

from paperlab import fly_study_evidence_bundle as bundle
from paperlab.core import digest
from paperlab.fly_online_figure import evidence
from paperlab.fly_selective_trace import require_completed_study
from test_fly_online import sealed
from test_fly_online_figure import figure_evidence


@pytest.fixture
def transport(figure_evidence, tmp_path, monkeypatch):
    checked = []

    def forbidden(*args, **kwargs):
        raise AssertionError('Evidence transport must not construct a neural model')

    monkeypatch.setattr('paperlab.fly_selective_trace.TraceLab', forbidden)

    def fixture_gate(root):
        # Only the release decision is mocked. All 16 chunk audits, receipts,
        # ledgers and aggregates still pass the real consistency checker.
        report, lanes, _, fixture = evidence(root)
        assert fixture and report['synthetic_figure_fixture'] and len(lanes) == 16
        checked.append(Path(root))
        return digest(Path(root) / 'report.json')

    monkeypatch.setattr(bundle, 'require_completed_study', fixture_gate)
    path = tmp_path / 'fixture-bundle.json'
    result = bundle.pack(figure_evidence, path)
    return figure_evidence, path, result, checked


def rewrite(path, payload):
    path.write_text(json.dumps(payload))
    return digest(path)


def test_transport_preserves_exact_bytes_and_checks_installed_evidence(transport, tmp_path):
    root, path, packed, checked = transport
    out = tmp_path / 'restored'
    restored = bundle.unpack(path, out, packed['bundle_sha256'])
    assert packed == restored
    assert packed['files'] == 51
    assert packed['neural_observations'] == packed['cloud_submissions'] == 0
    assert packed['raw_bytes'] == sum((root / name).stat().st_size for name in bundle.FILES)
    assert len(checked) == 3 and checked[-1] == out
    for name in bundle.FILES:
        assert (out / name).read_bytes() == (root / name).read_bytes()
    assert {p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file()} == set(bundle.FILES)


def test_production_pack_rejects_complete_synthetic_evidence(figure_evidence, tmp_path, monkeypatch):
    registration = json.loads((figure_evidence / 'report.json').read_text())['registration']
    monkeypatch.setattr('paperlab.fly_selective_trace.time.time', lambda: registration['end'] + 10000)
    out = tmp_path / 'must-not-exist.json'
    with pytest.raises(ValueError, match='actual study'):
        bundle.pack(figure_evidence, out)
    assert not out.exists()


def test_production_unpack_rejects_synthetic_bundle_despite_valid_hashes(transport, tmp_path, monkeypatch):
    _, path, packed, _ = transport
    monkeypatch.setattr(bundle, 'require_completed_study', require_completed_study)
    out = tmp_path / 'must-not-exist'
    with pytest.raises(ValueError, match='actual study'):
        bundle.unpack(path, out, packed['bundle_sha256'])
    assert not out.exists()


@pytest.mark.parametrize('change', ['bundle_hash', 'member_hash', 'missing', 'extra_path',
    'schema', 'bad_base64', 'report_hash', 'rehash_incomplete_report', 'duplicate_key'])
def test_corruption_is_rejected_before_destination_creation(transport, tmp_path, change):
    _, path, packed, _ = transport
    payload = json.loads(path.read_text()); expected = packed['bundle_sha256']
    if change == 'bundle_hash':
        expected = '0' * 64
    elif change == 'member_hash':
        payload['files']['report.json']['sha256'] = '0' * 64
    elif change == 'missing':
        del payload['files']['price-audit.json']
    elif change == 'extra_path':
        payload['files']['../../escaped.json'] = payload['files']['report.json']
    elif change == 'schema':
        payload['schema'] = 'different-study'
    elif change == 'bad_base64':
        payload['files']['report.json']['base64'] = '!'
    elif change == 'report_hash':
        payload['report_sha256'] = '0' * 64
    elif change == 'rehash_incomplete_report':
        report = json.loads(base64.b64decode(payload['files']['report.json']['base64']))
        report['audited'] = False
        raw = json.dumps(report).encode()
        payload['files']['report.json'] = {'sha256': bundle.sha256(raw),
                                          'base64': base64.b64encode(raw).decode()}
        payload['report_sha256'] = bundle.sha256(raw)
    if change == 'duplicate_key':
        raw = json.dumps(payload)
        path.write_text('{"schema":"ignored",' + raw[1:])
        expected = digest(path)
    elif change != 'bundle_hash':
        expected = rewrite(path, payload)
    out = tmp_path / 'rejected'
    with pytest.raises(ValueError):
        bundle.unpack(path, out, expected)
    assert not out.exists() and not (tmp_path.parent / 'escaped.json').exists()


@pytest.mark.parametrize('kind', ['file', 'directory', 'root'])
def test_symlinked_evidence_is_not_followed(figure_evidence, tmp_path, kind):
    root = figure_evidence
    if kind == 'root':
        root = tmp_path / 'linked'; root.symlink_to(figure_evidence, target_is_directory=True)
    else:
        original = root / ('report.json' if kind == 'file' else 'audits')
        moved = tmp_path / ('moved-' + original.name)
        original.rename(moved); original.symlink_to(moved, target_is_directory=kind == 'directory')
    with pytest.raises(ValueError, match='symlink|regular file'):
        bundle.pack(root, tmp_path / 'rejected.json')
    assert not (tmp_path / 'rejected.json').exists()


def test_missing_required_file_cannot_pack(figure_evidence, tmp_path):
    (figure_evidence / bundle.FILES[-1]).unlink()
    with pytest.raises(FileNotFoundError):
        bundle.pack(figure_evidence, tmp_path / 'rejected.json')
    assert not (tmp_path / 'rejected.json').exists()


def test_existing_destinations_and_unlisted_files_are_preserved(transport, tmp_path):
    root, path, packed, _ = transport
    (root / 'unrelated.txt').write_text('not transported')
    before = path.read_bytes()
    with pytest.raises(ValueError, match='Preserve'):
        bundle.pack(root, path)
    assert path.read_bytes() == before
    out = tmp_path / 'existing'; out.mkdir()
    with pytest.raises(ValueError, match='Preserve'):
        bundle.unpack(path, out, packed['bundle_sha256'])
    assert list(out.iterdir()) == []
    linked = tmp_path / 'dangling'; linked.symlink_to(tmp_path / 'missing')
    with pytest.raises(ValueError, match='Preserve'):
        bundle.unpack(path, linked, packed['bundle_sha256'])
    assert linked.is_symlink()


@pytest.mark.parametrize('limit', ['MAX_FILE_BYTES', 'MAX_TOTAL_BYTES', 'MAX_BUNDLE_BYTES'])
def test_limits_reject_oversized_transport(transport, tmp_path, monkeypatch, limit):
    _, path, packed, _ = transport
    monkeypatch.setattr(bundle, limit, 1)
    out = tmp_path / 'oversized'
    with pytest.raises(ValueError, match='limit|oversized'):
        bundle.unpack(path, out, packed['bundle_sha256'])
    assert not out.exists()
