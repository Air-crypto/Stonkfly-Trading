import json
from pathlib import Path

import pytest

from paperlab.fly_visual_study import summarize


@pytest.fixture
def magnitude_study(tmp_path):
    reports = Path(__file__).resolve().parents[1] / "reports"
    published = json.loads((reports / "fly-fixed-scale-study-01.json").read_text())
    (tmp_path / "protocol.json").write_bytes((reports / "fly-fixed-scale-protocol-01.json").read_bytes())
    (tmp_path / "source-hashes.json").write_text(json.dumps(published["source_sha256"]))
    for arm, report in published["reports"].items():
        (tmp_path / f"{arm}-result.json").write_text(json.dumps({"status":"debug_completed", "report":report}))
    return tmp_path, published


def test_published_magnitude_study_reproduces(magnitude_study):
    root, published = magnitude_study
    assert summarize(root) == published
    assert published["comparisons"]["fixed_returns_rise"]["different_spike_count_observations"] == 4
    assert published["comparisons"]["original_rise"]["different_spike_count_observations"] == 0


@pytest.mark.parametrize("change", ["weights", "build", "original_control"])
def test_magnitude_audit_rejects_invalid_controls(magnitude_study, change):
    root, _ = magnitude_study
    path = root / "original_rise_small-result.json"
    result = json.loads(path.read_text())
    if change == "weights":
        result["report"]["events"][0]["diagnostics"]["weight_delta_l2"] = 1
    elif change == "build":
        result["report"]["native_build"]["binary_sha256"] = "different"
    else:
        result["report"]["events"][0]["spike_sha256"] = "different"
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError): summarize(root)
