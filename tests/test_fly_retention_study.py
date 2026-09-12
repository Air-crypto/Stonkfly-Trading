"""Re-audit actual cloud outcomes and reject evidence-breaking changes."""
import json
from pathlib import Path

import pytest

from paperlab.fly_retention_study import summarize


@pytest.fixture
def recorded_study(tmp_path):
    reports = Path(__file__).resolve().parents[1] / "reports"
    study = json.loads((reports / "fly-retention-study-01.json").read_text())
    (tmp_path / "protocol.json").write_bytes((reports / "fly-retention-protocol-01.json").read_bytes())
    (tmp_path / "source-hashes.json").write_text(json.dumps(study["executed_source_sha256"]))
    for arm, report in study["reports"].items():
        (tmp_path / f"{arm}-result.json").write_text(json.dumps({"status": "debug_completed", "report": report}))
    return tmp_path, study


def test_published_retention_results_reproduce(recorded_study):
    root, published = recorded_study
    assert summarize(root) == published


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
