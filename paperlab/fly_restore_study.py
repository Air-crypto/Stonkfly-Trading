"""Bounded synthetic memory-restoration experiment; no market-policy selection."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

from .core import atomic_json
from .fly_trace import Assay, TraceLab


def arms():
    base = Assay(preset="fall", steps=4, probe_steps=4, probe_preset="fall")
    return {"frozen": replace(base, learning=False), "retained": base,
            **{mode: replace(base, probe_restore=mode) for mode in ("all", "MBON07", "MBON11")}}


def summarize(root):
    root = Path(root)
    protocol = json.loads((root / "protocol.json").read_text())
    expected = json.loads(json.dumps({name: asdict(c) for name, c in arms().items()}))
    if protocol["arms"] != expected:
        raise ValueError("Restoration protocol differs from the registered five controls")
    reports = {name: json.loads((root / name / "report.json").read_text()) for name in expected}
    if len({json.dumps(r["native_build"], sort_keys=True) for r in reports.values()}) != 1:
        raise ValueError("Native builds differ")
    if len({json.dumps(r["graph"], sort_keys=True) for r in reports.values()}) != 1:
        raise ValueError("Retained graphs differ")
    reference = reports["frozen"]["events"][4:]
    training = reports["retained"]["events"][:4]
    rows = {}
    for name, report in reports.items():
        if report["config"] != expected[name] or len(report["events"]) != 8:
            raise ValueError("Missing or mismatched arm")
        if name != "frozen" and [e["spike_sha256"] for e in report["events"][:4]] != [e["spike_sha256"] for e in training]:
            raise ValueError("Training changed before the restoration intervention")
        probe = report["events"][4:]
        boundary = report["probe_boundary"]
        if not boundary["dynamics_reset"] or any(e["phase"] != "probe" or e["stimulus"] != "none"
                or e["diagnostics"]["plasticity_enabled"] or e["diagnostics"]["weight_delta_l2"] != 0 for e in probe):
            raise ValueError("Probe was not isolated and frozen")
        if any(a["input_sha256"] != b["input_sha256"] for a,b in zip(probe, reference)):
            raise ValueError("Probe inputs differ")
        rows[name] = {"actions": [e["side"] for e in probe],
                      "different_actions_from_pristine": sum(a["side"] != b["side"] for a,b in zip(probe,reference)),
                      "different_spike_counts_from_pristine": sum(a["spike_sha256"] != b["spike_sha256"] for a,b in zip(probe,reference)),
                      "restored_edges": boundary["restoration"]["edge_count"],
                      "trained_weight_delta_l2": boundary["restoration"]["trained_weight_delta_from_pristine_l2"],
                      "probe_weight_delta_l2": boundary["weight_delta_from_pristine_l2"]}
    if rows["all"]["different_spike_counts_from_pristine"] or rows["all"]["probe_weight_delta_l2"]:
        raise ValueError("Complete restoration failed to reproduce pristine response")
    return {"protocol_sha256": hashlib.sha256((root/"protocol.json").read_bytes()).hexdigest(),
            "source_sha256": protocol["source_sha256"], "summary": rows, "reports": reports,
            "interpretation": "Single synthetic falling-price cue with a frozen probe. Restoration is a causal memory intervention; recovered pristine behavior is not evidence of better trading."}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    p.add_argument("--fly-data",type=Path)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    if args.fly_data:
        args.root.mkdir(parents=True,exist_ok=False)
        sources=[Path(__file__),Path(__file__).with_name("fly_trace.py"),Path(__file__).with_name("fly.py"),Path(__file__).with_name("fly_visual.py")]
        protocol={"registered_at":time.time(),"arms":{n:asdict(c) for n,c in arms().items()},
                  "source_sha256":{s.name:hashlib.sha256(s.read_bytes()).hexdigest() for s in sources},
                  "purpose":"Restore all or compartment-specific learned memory; compare identical frozen probes against pristine and retained controls. Report every arm without selecting by desired output."}
        atomic_json(args.root/"protocol.json",protocol)
        lab=TraceLab(args.fly_data)
        for name,config in arms().items():
            lab.run(config,args.root/name)
    result=summarize(args.root)
    atomic_json(args.out,result)
    print(json.dumps(result["summary"],indent=2))


if __name__=="__main__": main()
