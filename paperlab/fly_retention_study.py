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


def counterbalance(first, second):
    """Compare audited rise/fall training batches without treating actions as returns."""
    if first["executed_source_sha256"] != second["executed_source_sha256"]:
        raise ValueError("Counterbalanced batches used different execution sources")
    batches = {}
    for study in (first, second):
        cue = study["reports"]["frozen_rise"]["config"]["preset"]
        if cue in batches:
            raise ValueError("Counterbalance requires both training cues")
        batches[cue] = study
    if set(batches) != {"rise", "fall"}:
        raise ValueError("Counterbalance requires rise and fall training")
    rise, fall = batches["rise"], batches["fall"]
    result = {}
    for arm, a in rise["reports"].items():
        b = fall["reports"][arm]
        if a["native_build"] != b["native_build"] or a["graph"] != b["graph"]:
            raise ValueError("Counterbalanced native build or graph differs")
        ac, bc = (dict(r["config"]) for r in (a, b))
        ac.pop("preset"); bc.pop("preset")
        if ac != bc:
            raise ValueError(f"{arm}: another experimental factor changed")
        ap, bp = a["events"][4:], b["events"][4:]
        if [e["input_sha256"] for e in ap] != [e["input_sha256"] for e in bp]:
            raise ValueError(f"{arm}: counterbalanced probe images differ")
        if arm.startswith("frozen_") and [e["spike_sha256"] for e in ap] != [
                e["spike_sha256"] for e in bp]:
            raise ValueError(f"{arm}: counterbalanced frozen controls do not reproduce")
        result[arm] = {
            "rise_training_actions": [e["side"] for e in ap],
            "fall_training_actions": [e["side"] for e in bp],
            "different_actions": sum(x["side"] != y["side"] for x, y in zip(ap, bp)),
            "different_spike_hashes": sum(x["spike_sha256"] != y["spike_sha256"] for x, y in zip(ap, bp)),
            "rise_minus_fall_training_hz": [x["difference_hz"]-y["difference_hz"] for x, y in zip(ap, bp)],
            "rise_minus_fall_training_gate_spikes": [x["gate_spikes"]-y["gate_spikes"] for x, y in zip(ap, bp)],
        }
    interaction = {}
    for group in ("reward", "aversive"):
        interaction[group] = {}
        for metric in ("difference_hz", "gate_spikes"):
            effects = {}
            for train, study in batches.items():
                effects[train] = {}
                for probe in ("rise", "fall"):
                    treated = study["reports"][f"{group}_{probe}"]["events"][4:]
                    neutral = study["reports"][f"neutral_{probe}"]["events"][4:]
                    effects[train][probe] = [a[metric]-b[metric] for a, b in zip(treated, neutral)]
            interaction[group][metric] = {
                "reinforcement_minus_neutral_by_training_and_probe_cue": effects,
                "training_cue_by_probe_cue_interaction": [
                    (a-b)-(c-d) for a, b, c, d in zip(effects["rise"]["rise"], effects["rise"]["fall"],
                                                     effects["fall"]["rise"], effects["fall"]["fall"])],
            }
    return {"source_protocol_sha256": {cue: study["protocol_sha256"] for cue, study in batches.items()},
            "comparison": result, "descriptive_interactions": interaction,
            "verification": {"only_training_cue_changed": True, "same_probe_images": True,
                             "frozen_controls_reproduce_across_batches": True},
            "interpretation": "Deterministic synthetic mechanics. Per-observation interactions are descriptive, "
                              "not independent trials, returns, a significance test, or a policy-selection score."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--compare-root", type=Path, help="Audit and compare a second training-cue batch")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root)
    if args.compare_root:
        result = counterbalance(result, summarize(args.compare_root))
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result.get("summary", result.get("comparison")), indent=2))


if __name__ == "__main__":
    main()
