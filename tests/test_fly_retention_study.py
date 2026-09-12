"""Re-audit actual cloud outcomes and reject evidence-breaking changes."""
import json
from pathlib import Path

import pytest

from paperlab.fly_retention_study import counterbalance, summarize


@pytest.fixture(params=["01", "02"])
def recorded_study(tmp_path, request):
    reports = Path(__file__).resolve().parents[1] / "reports"
    study = json.loads((reports / f"fly-retention-study-{request.param}.json").read_text())
    (tmp_path / "protocol.json").write_bytes((reports / f"fly-retention-protocol-{request.param}.json").read_bytes())
    (tmp_path / "source-hashes.json").write_text(json.dumps(study["executed_source_sha256"]))
    for arm, report in study["reports"].items():
        (tmp_path / f"{arm}-result.json").write_text(json.dumps({"status": "debug_completed", "report": report}))
    return tmp_path, study


def test_published_retention_results_reproduce(recorded_study):
    root, published = recorded_study
    assert summarize(root) == published


def test_counterbalance_rejects_duplicate_training_cue(recorded_study):
    _, published = recorded_study
    with pytest.raises(ValueError, match="both training cues"):
        counterbalance(published, published)


@pytest.mark.parametrize("corruption", ["build", "probe_update", "wrong_image", "training_memory"])
def test_reject_uncontrolled_retention_comparisons(recorded_study, corruption):
    root, _ = recorded_study
    path = root / "reward_fall-result.json"
    result = json.loads(path.read_text())
    report = result["report"]
    if corruption == "build":
        report["native_build"]["binary_sha256"] = "different-build"
    elif corruption == "probe_update":
        report["events"][4]["diagnostics"]["weight_delta_l2"] = .01
    elif corruption == "wrong_image":
        report["events"][4]["input_sha256"] = "different-image"
    else:
        report["probe_boundary"]["memory_sha256"] = "different-memory"
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError):
        summarize(root)


def test_published_counterbalance_and_gate_interaction_reproduce():
    reports = Path(__file__).resolve().parents[1] / "reports"
    first, second = [json.loads((reports / f"fly-retention-study-{n:02}.json").read_text()) for n in (1, 2)]
    result = counterbalance(first, second)
    assert result == json.loads((reports / "fly-retention-counterbalance.json").read_text())
    # Fourth probe: reward-minus-neutral gate effect is +4 after rise training,
    # -1 after fall training, and zero on both fall probes: interaction = 5.
    assert result["descriptive_interactions"]["reward"]["gate_spikes"]["training_cue_by_probe_cue_interaction"][-1] == 5


@pytest.mark.parametrize("corruption", ["source", "config", "frozen_probe"])
def test_counterbalance_rejects_confounded_batches(corruption):
    reports = Path(__file__).resolve().parents[1] / "reports"
    first, second = [json.loads((reports / f"fly-retention-study-{n:02}.json").read_text()) for n in (1, 2)]
    if corruption == "source":
        second["executed_source_sha256"]["fly.py"] = "changed-source"
    elif corruption == "config":
        second["reports"]["reward_rise"]["config"]["eta"] = .0001
    else:
        second["reports"]["frozen_rise"]["events"][4]["spike_sha256"] = "different-spikes"
    with pytest.raises(ValueError):
        counterbalance(first, second)
