"""Offline audit of the preregistered, frozen-network magnitude experiment."""
import argparse
import hashlib
import json
from pathlib import Path


def summarize(root):
    root = Path(root)
    protocol = json.loads((root / "protocol.json").read_text())
    expected = {f"{view}_{cue}_{size}" for view in ("original", "fixed_returns")
                for cue in ("rise", "fall") for size in ("small", "large")}
    if set(protocol["arms"]) != expected:
        raise ValueError("Expected all eight magnitude controls")
    reports = {}
    for arm, config in protocol["arms"].items():
        result = json.loads((root / f"{arm}-result.json").read_text())
        if result["status"] != "debug_completed" or result["report"]["config"] != config:
            raise ValueError(f"{arm}: incomplete or altered experiment")
        report = result["report"]
        if len(report["events"]) != 4 or config["learning"] or config["probe_steps"]:
            raise ValueError("Expected four frozen observations without probes")
        if config["view"] == "fixed_returns" and report["visual_encoding"] != protocol["encoding"]:
            raise ValueError("Reported visual encoding differs from registration")
        for event in report["events"]:
            if (event["diagnostics"]["plasticity_enabled"] or event["diagnostics"]["weight_delta_l2"] != 0
                    or event["stimulus"] != "none"):
                raise ValueError(f"{arm}: weights changed or reinforcement was delivered")
        reports[arm] = report
    builds = {json.dumps(r["native_build"], sort_keys=True) for r in reports.values()}
    graphs = {json.dumps(r["graph"], sort_keys=True) for r in reports.values()}
    memories = {e["memory"]["sha256"] for r in reports.values() for e in r["events"]}
    if len(builds) != 1 or len(graphs) != 1 or len(memories) != 1:
        raise ValueError("Controls mix native builds, graphs, or weights")
    comparisons = {}
    for view in ("original", "fixed_returns"):
        for cue in ("rise", "fall"):
            pair = [reports[f"{view}_{cue}_{size}"] for size in ("small", "large")]
            ac, bc = [dict(r["config"]) for r in pair]
            if (ac.pop("amplitude"), bc.pop("amplitude")) != (.05, 1) or ac != bc:
                raise ValueError("Magnitude comparison changes another experimental factor")
            a, b = [r["events"] for r in pair]
            different_inputs = sum(x["input_sha256"] != y["input_sha256"] for x, y in zip(a, b))
            different_spikes = sum(x["spike_sha256"] != y["spike_sha256"] for x, y in zip(a, b))
            if view == "original" and (different_inputs or different_spikes):
                raise ValueError("Original collision control did not reproduce")
            comparisons[f"{view}_{cue}"] = {
                "different_input_observations": different_inputs,
                "different_spike_count_observations": different_spikes,
                "different_actions": sum(x["side"] != y["side"] for x, y in zip(a, b)),
                "small_actions": [e["side"] for e in a], "large_actions": [e["side"] for e in b],
                "small_difference_hz": [e["difference_hz"] for e in a],
                "large_difference_hz": [e["difference_hz"] for e in b],
                "small_gate_spikes": [e["gate_spikes"] for e in a],
                "large_gate_spikes": [e["gate_spikes"] for e in b],
            }
    return {"purpose": protocol["purpose"], "comparisons": comparisons,
            "verification": {"same_build_graph_and_weights": True, "all_weights_frozen": True,
                             "original_collision_controls_reproduce": True},
            "protocol_sha256": hashlib.sha256((root / "protocol.json").read_bytes()).hexdigest(),
            "source_sha256": json.loads((root / "source-hashes.json").read_text()),
            "interpretation": protocol["interpretation"] + " Spike hashes represent whole-observation counts, not precise spike times.",
            "reports": reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = summarize(args.root)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["comparisons"], indent=2))


if __name__ == "__main__":
    main()
