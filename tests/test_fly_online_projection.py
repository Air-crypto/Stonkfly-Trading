"""Small-array projection regressions and an opt-in retained full-graph check."""
import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest

from paperlab import fly_online_projection as online
from paperlab.core import atomic_json
from paperlab.fly_view_projection import array_digest, audit_view
from test_fly_view_projection import fixture


def phase(root, observations):
    graph, view, original, arrays = fixture()
    graph.source_sha256 = {'graph.npz': 'small-test-fixture'}
    total = np.zeros(len(graph.ids), dtype=np.int64)
    frames = []
    for i in range(observations):
        a = {k: v.copy() for k, v in arrays.items()}
        a['ms'] += 500 * i
        a['counts'][0, 4] += i  # Later frames contribute distinct neuron totals.
        f = copy.deepcopy(original)
        f.update(step=i + 1, times_ms=a['ms'].tolist(), counts=a['counts'].tolist(),
            total_spikes=a['counts'].sum(axis=1).tolist(),
            groups={k: a['counts'][:, indices].sum(axis=1).tolist() for k, indices in graph.groups.items()})
        counts = a['counts'].sum(axis=0, dtype=np.int64)
        f['event'].update(brain_ms=float(a['ms'][-1]), total_spikes=int(counts.sum()),
                          spike_sha256=array_digest(counts.astype(np.int32)))
        path = root / 'trace' / f'step-{i+1:02}.npz'
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **a)
        total += counts; frames.append(f)
    view['frames'] = frames
    view['report']['events'] = copy.deepcopy([f['event'] for f in frames])
    for node, count in zip(view['nodes'], total):
        node['total_spikes'] = int(count)
    return graph, view


@pytest.mark.parametrize('observations', [1, 3, 21, 24])
def test_complete_projection_checks_every_declared_observation(tmp_path, observations):
    graph, view = phase(tmp_path, observations)
    checked = audit_view(view, tmp_path / 'trace', graph, observations=observations)
    assert checked['observations'] == observations and checked['bins'] == 50 * observations
    assert len(checked['norm_reduction_errors']) == observations


@pytest.mark.parametrize('change', ['late_curve', 'late_population', 'late_voltage',
    'late_native_time', 'missing_frame', 'missing_event', 'aggregate', 'report_event'])
def test_long_view_cannot_hide_changes_after_the_first_three_frames(tmp_path, change):
    graph, view = phase(tmp_path, 21)
    if change == 'late_curve': view['frames'][-1]['total_spikes'][10] += 1
    elif change == 'late_population': view['frames'][-1]['groups']['DAN'][10] += 1
    elif change == 'late_voltage': view['frames'][-1]['voltage'][10][0] += 1
    elif change == 'missing_frame': view['frames'].pop()
    elif change == 'missing_event': view['report']['events'].pop()
    elif change == 'aggregate': view['nodes'][4]['total_spikes'] -= 20
    elif change == 'report_event': view['report']['events'][-1]['brain_ms'] += 1
    else:
        path = tmp_path / 'trace/step-21.npz'
        with np.load(path) as saved: a = {k: saved[k] for k in saved.files}
        a['ms'] += 500
        np.savez(path, **a)
    with pytest.raises(ValueError):
        audit_view(view, tmp_path / 'trace', graph, observations=21)


@pytest.mark.parametrize('count', [0, 25, True, 3.0, None])
def test_invalid_declared_counts_are_rejected(tmp_path, count):
    graph, view = phase(tmp_path, 3)
    with pytest.raises(ValueError, match='declared observations'):
        audit_view(view, tmp_path / 'trace', graph, observations=count)


def test_default_still_requires_three_mechanism_observations(tmp_path):
    graph, view = phase(tmp_path, 21)
    with pytest.raises(ValueError, match='declared observation count'):
        audit_view(view, tmp_path / 'trace', graph)


def mock_phase_audit(root, monkeypatch, count=21):
    # Unit tests below isolate post-audit orchestration. They do not claim that
    # the tiny fixture passed a biological or financial audit.
    graph, view = phase(root, count)
    if count == 0: view = None
    envelope = {'sha256': 'fixture-plan', 'plan': {'series': {'pool': [{'source': 'synthetic'}]}}}
    atomic_json(root / 'protocol.json', envelope)
    atomic_json(root / 'summary.json', {'chunk': 'unit-test-fixture'})
    checked = {'artifact_sha256': {}, 'verification': {
        'observations': count, 'native_bins': 50 * count, 'decision_slots': 25}}
    monkeypatch.setattr(online, 'validate', lambda value: value['plan'])
    monkeypatch.setattr(online, 'audit_chunk', lambda *args: (checked, view))
    monkeypatch.setattr(online, 'load_graph', lambda data: graph)
    return view


def test_original_expanded_view_is_checked_and_labeled(tmp_path, monkeypatch):
    root = tmp_path / 'recording'; view = mock_phase_audit(root, monkeypatch)
    supplied = tmp_path / 'supplied.json'; atomic_json(supplied, view)
    out = tmp_path / 'checked'
    report = online.run(root, 'unused', out, view_path=supplied)
    assert report['validation_only'] and report['projection']['observations'] == 21
    assert report['neural_observations_computed'] == report['cloud_submissions'] == 0
    assert json.loads((out / 'view.json').read_text()) == view
    with pytest.raises(ValueError, match='Preserve'):
        online.run(root, 'unused', out)


def test_supplied_credit_panel_cannot_disagree_with_fresh_audit(tmp_path, monkeypatch):
    root = tmp_path / 'recording'; view = mock_phase_audit(root, monkeypatch)
    view = copy.deepcopy(view)
    view['frames'][-1]['plastic_credit'] = {'forged': True}
    supplied = tmp_path / 'changed.json'; atomic_json(supplied, view)
    with pytest.raises(ValueError, match='fresh independent reconstruction'):
        online.run(root, 'unused', tmp_path / 'out', view_path=supplied)
    assert not (tmp_path / 'out').exists()


def test_unobserved_phase_is_not_reported_as_a_verified_plot(tmp_path, monkeypatch):
    root = tmp_path / 'recording'; mock_phase_audit(root, monkeypatch, 0)
    def forbidden(*args): raise AssertionError('No graph needed for an absent view')
    monkeypatch.setattr(online, 'load_graph', forbidden)
    out = tmp_path / 'empty'
    report = online.run(root, 'unused', out)
    assert report['projection'] == {'observations': 0, 'bins': 0, 'view_status': 'no_neural_observations'}
    assert report['audited_view_sha256'] is None and not (out / 'view.json').exists()


def test_retained_full_phase_without_constructing_a_brain(tmp_path, monkeypatch):
    root = os.environ.get('FLY_ONLINE_RECORDINGS'); data = os.environ.get('FLY_TRACE_DATA')
    if not root or not data:
        pytest.skip('Requires retained 21-observation arrays and prepared graph; no propagation')
    from paperlab.fly import configure
    configure(data)
    from stonkfly.neural.brain import MemoryBrain
    def forbidden(*args, **kwargs): raise AssertionError('No native model may be constructed')
    monkeypatch.setattr(MemoryBrain, '__init__', forbidden)
    report = online.run(root, data, tmp_path / 'real-arrays')
    assert report['validation_only'] and report['projection']['observations'] == 21
    assert report['projection']['bins'] == 1050 and report['projection']['displayed_connections'] == 967
    assert report['projection']['all_topology_verified'] and report['projection']['all_plotted_series_verified']
    assert report['neural_observations_computed'] == report['cloud_submissions'] == 0
