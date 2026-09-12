"""Audit the registered retention assay from completed cloud results, offline."""
import argparse
import hashlib
import json
from pathlib import Path


def summarize(root):
    root = Path(root)
    protocol = json.loads((root / "protocol.json").read_text())
    expected = {f"{group}_{cue}" for group in ("frozen", "neutral", "reward", "aversive")
                for cue in ("rise", "fall")}
    if set(protocol["arms"]) != expected:
        raise ValueError("Expected all eight registered retention arms")
    reports = {}
    for arm, config in protocol["arms"].items():
        result = json.loads((root / f"{arm}-result.json").read_text())
        if result["status"] != "debug_completed":
            raise ValueError(f"{arm}: incomplete result")
        report = result["report"]
        if report["config"] != config:
            raise ValueError(f"{arm}: configuration differs from registration")
        if (config["steps"], config["probe_steps"]) != (4, 4):
            raise ValueError("This analysis requires four training and four probe observations")
        training = [e for e in report["events"] if e["phase"] == "training"]
        probe = [e for e in report["events"] if e["phase"] == "probe"]
        boundary = report["probe_boundary"]
        if (len(training), len(probe)) != (config["steps"], config["probe_steps"]):
            raise ValueError(f"{arm}: missing observations")
        if (not boundary["dynamics_reset"] or boundary["probe_learning"]
                or boundary["probe_reinforcement"] != "none" or boundary["probe_extra_current"] != 0):
            raise ValueError(f"{arm}: invalid reset/freeze")
        for i, event in enumerate(probe, 1):
            if (event["phase_step"] != i or event["stimulus"] != "none"
                    or event["input_news"] != "none"
                    or event["input_preset"] != config["probe_preset"]
                    or event["diagnostics"]["plasticity_enabled"]
                    or event["diagnostics"]["weight_delta_l2"] != 0
                    or event["memory"]["sha256"] != boundary["weight_sha256"]):
                raise ValueError(f"{arm}: probe changed memory or received stimulation")
        reports[arm] = report
    builds = {r["native_build"]["binary_sha256"] for r in reports.values()}
    sources = {r["native_build"]["source_sha256"] for r in reports.values()}
    graphs = {json.dumps(r["graph"], sort_keys=True) for r in reports.values()}
    if len(builds) != 1 or len(sources) != 1 or len(graphs) != 1:
        raise ValueError("Comparison mixes native builds or graphs")
    for group in ("frozen", "neutral", "reward", "aversive"):
        rise, fall = (reports[f"{group}_{cue}"] for cue in ("rise", "fall"))
        # Compare actual array hashes. Float32 norm reductions can differ by an
        # ulp across hosts even when every retained byte is identical.
        if any(rise["probe_boundary"][key] != fall["probe_boundary"][key]
               for key in ("weight_sha256", "memory_sha256")):
            raise ValueError(f"{group}: training memory does not reproduce")
        if [e["spike_sha256"] for e in rise["events"][:4]] != [
                e["spike_sha256"] for e in fall["events"][:4]]:
            raise ValueError(f"{group}: training dynamics do not reproduce")
    training_cues = {r["config"]["preset"] for r in reports.values()}
    if len(training_cues) != 1 or not training_cues <= {"rise", "fall"}:
        raise ValueError("Expected one training cue per batch")
    frozen = reports[f"frozen_{next(iter(training_cues))}"]["events"]
    if [e["spike_sha256"] for e in frozen[:4]] != [e["spike_sha256"] for e in frozen[4:]]:
        raise ValueError("Frozen reset fails to reproduce pristine control")
    summary = {}
    for arm, report in reports.items():
        cue = report["config"]["probe_preset"]
        probe = report["events"][4:]
        controls = {group: reports[f"{group}_{cue}"]["events"][4:]
                    for group in ("frozen", "neutral")}
        for control in controls.values():
            if [e["input_sha256"] for e in probe] != [e["input_sha256"] for e in control]:
                raise ValueError(f"{arm}: control images differ")
        summary[arm] = {
            "retained_weight_l2": report["probe_boundary"]["weight_delta_from_pristine_l2"],
            "actions": [e["side"] for e in probe],
            "difference_hz": [e["difference_hz"] for e in probe],
            "gate_spikes": [e["gate_spikes"] for e in probe],
            "different_actions": {g: sum(a["side"] != b["side"] for a, b in zip(probe, c))
                                  for g, c in controls.items()},
            "different_spike_hashes": {
                g: sum(a["spike_sha256"] != b["spike_sha256"] for a, b in zip(probe, c))
                for g, c in controls.items()},
        }
    interaction = {}
    for group in ("reward", "aversive"):
        effects = {}
        for cue in ("rise", "fall"):
            effects[cue] = [a-b for a, b in zip(summary[f"{group}_{cue}"]["difference_hz"],
                                             summary[f"neutral_{cue}"]["difference_hz"])]
        interaction[group] = {
            "minus_neutral_hz": effects,
            "rise_minus_fall_effect_hz": [a-b for a, b in zip(effects["rise"], effects["fall"])],
        }
    return {
        "purpose": protocol["purpose"], "summary": summary,
        "cue_interaction_descriptive_only": interaction,
        "verification": {"same_native_build_and_graph": True, "same_probe_images": True,
                         "paired_training_reproduces": True, "frozen_reset_reproduces": True,
                         "all_probe_weights_frozen": True},
        "interpretation": protocol["interpretation"],
        "protocol_sha256": hashlib.sha256((root / "protocol.json").read_bytes()).hexdigest(),
        "executed_source_sha256": json.loads((root / "source-hashes.json").read_text()),
        "reports": reports,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
