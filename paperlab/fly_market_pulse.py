"""Post hoc factorial replay of recorded market inputs; no account or order logic."""
import argparse
from dataclasses import replace
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np

from .core import Tick, atomic_json, digest
from .fly import frame, UPSTREAM_COMMIT
from .fly_market_study import validate as validate_market, signature, quote_at, learned_state, restore_learned
from .fly_trace import TraceLab
from .news import News

ARMS = {f"{memory}_{learning}_{pulses}": {"memory":memory,"learning":learning=="online","pulses":pulses}
        for memory,learning,pulses in itertools.product(("pristine","trained"),("frozen","online"),("none","recorded"))}
SOURCE = "matched_market_stimulus_assay"


def validate(payload):
    if not isinstance(payload,dict) or set(payload)!={"protocol","reference_json","market_plan"}:
        raise ValueError("Expected registered protocol, reference JSON, and sealed market plan")
    protocol=payload["protocol"]; raw=payload["reference_json"]
    if not isinstance(protocol,dict) or protocol.get("arms")!=ARMS:
        raise ValueError("Expected all eight memory/update/pulse controls")
    if not isinstance(raw,str) or len(raw.encode())>2_000_000:
        raise ValueError("Reference report must be at most 2 MB")
    if hashlib.sha256(raw.encode()).hexdigest()!=protocol.get("reference_report_sha256"):
        raise ValueError("Reference report hash differs from registration")
    reference=json.loads(raw);envelope=payload["market_plan"]
    if not isinstance(envelope,dict) or set(envelope)!={"plan","sha256"}:
        raise ValueError("Expected sealed market envelope")
    plan=validate_market(envelope["plan"])
    if (signature(plan)!=envelope["sha256"] or envelope["sha256"]!=protocol.get("plan_sha256")
            or reference.get("plan_sha256")!=envelope["sha256"]):
        raise ValueError("Market plan and reference provenance differ")
    if plan["phase_steps"]!=3 or plan["arms"].get("online_original",{}).get("view")!="original":
        raise ValueError("This diagnostic requires three observations per phase and the original view")
    pool=protocol.get("pool")
    if pool not in plan["cohort"] or pool not in reference.get("phase_diagnostics",{}):
        raise ValueError("Registered pool is missing from the reference")
    rows=reference["phase_diagnostics"][pool]
    for arm,phase in (("online_original","training"),("online_original","test"),("pristine_frozen","test")):
        decisions=rows[arm][phase]["decisions"]
        expected_start=plan["start"]+(0 if phase=="training" else 6*300)
        if (len(decisions)!=4 or [d["decision_ts"] for d in decisions]!=[expected_start+i*300 for i in range(4)]
                or any(not d["available"] or d["neural"] is None or d["terminal"] for d in decisions[:3])
                or not decisions[-1]["terminal"]):
            raise ValueError("Reference must contain all three aligned observed decisions")
        for d in decisions[:3]:
            if d["neural"].get("stimulus") not in ("none","reward","aversive"):
                raise ValueError("Invalid recorded reinforcement")
    if [d["neural"]["stimulus"] for d in rows["online_original"]["test"]["decisions"][:3]]!=["none","aversive","reward"]:
        raise ValueError("Recorded test pulse schedule differs from registration")
    return protocol,reference,plan


def input_sequences(plan, reference, pool):
    ticks=[Tick(**r) for r in plan["series"][pool]]
    news=News(enabled=False)
    sequences={}
    try:
        for phase in ("training","test"):
            pairs=[]
            rows=reference["phase_diagnostics"][pool]["online_original"][phase]["decisions"][:3]
            for row in rows:
                index,quote=quote_at(ticks,row["decision_ts"])
                if not quote.available or quote.ts!=row["quote_ts"]:
                    raise ValueError("Reference quote does not match the sealed history")
                rgb=frame(ticks,index,news)
                if hashlib.sha256(rgb.tobytes()).hexdigest()!=row["neural"]["input_sha256"]:
                    raise ValueError("Rendered input differs from recorded image")
                pairs.append((rgb,row))
            sequences[phase]=pairs
    finally:
        news.db.close()
    return sequences


def same_memory(brain, state):
    return all(np.array_equal(a,b) for a,b in zip(learned_state(brain).values(),state.values()))


def run(payload, data, output):
    protocol,reference,plan=validate(payload)
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    atomic_json(root/"protocol.json",protocol)
    lab=TraceLab(data);b=lab.brain
    if b.build!=reference["native_build"]:
        raise ValueError("Native build differs from reference; controlled replay requires the same build")
    sequences=input_sequences(plan,reference,protocol["pool"])
    deadline=time.monotonic()+420
    b.reset();b.eta=.001;b.weights_frozen=False
    lab.fly.controller.s=replace(lab.fly.controller.s,learning=True)
    training=[]
    for rgb,row in sequences["training"]:
        if time.monotonic()>deadline:raise TimeoutError("Bounded pulse assay expired")
        event=lab.fly.controller.observe(rgb,row["neural"]["stimulus"])
        if event["spike_sha256"]!=row["neural"]["spike_sha256"]:
            raise AssertionError("Original training spike counts did not reproduce")
        training.append(event)
    trained=learned_state(b)
    np.savez_compressed(root/"trained-memory.npz",**trained)
    reports={};summary={}
    for name,arm in ARMS.items():
        if time.monotonic()>deadline:raise TimeoutError("Bounded pulse assay expired")
        b.reset()
        if arm["memory"]=="trained":restore_learned(b,trained)
        before=learned_state(b)
        b.eta=.001;b.weights_frozen=not arm["learning"]
        lab.fly.controller.s=replace(lab.fly.controller.s,learning=arm["learning"])
        folder=root/name;folder.mkdir()
        np.savez_compressed(folder/"initial-memory.npz",**before)
        started=time.monotonic();events=[]
        for i,(rgb,row) in enumerate(sequences["test"]):
            if time.monotonic()>deadline:raise TimeoutError("Bounded pulse assay expired")
            pulse=row["neural"]["stimulus"] if arm["pulses"]=="recorded" else "none"
            event=lab.capture(rgb,pulse,folder,i+1)
            event.update(phase="replay",phase_step=i+1,market_decision_ts=row["decision_ts"],input_preset="recorded_market",input_news="none")
            events.append(event)
            if not arm["learning"] and not same_memory(b,before):
                raise AssertionError("Frozen factorial arm changed synaptic memory")
        if name in ("pristine_frozen_none","trained_online_recorded"):
            ref_arm="pristine_frozen" if name=="pristine_frozen_none" else "online_original"
            expected=reference["phase_diagnostics"][protocol["pool"]][ref_arm]["test"]["decisions"][:3]
            if any(e["spike_sha256"]!=r["neural"]["spike_sha256"] for e,r in zip(events,expected)):
                raise AssertionError(f"Reference control did not reproduce: {name}")
        initial_hash=hashlib.sha256(before["weights"].tobytes()+before["u"].tobytes()+before["w"].tobytes()).hexdigest()
        report={"schema":1,"source":SOURCE,"config":{"preset":"recorded_market","view":"original","news":"none","eta":.001,**arm},
                "graph":{"neurons":b.n,"edges":len(b.post),"plastic_edges":len(b.circuit["edges"])},
                "native_build":b.build,"upstream_commit":UPSTREAM_COMMIT,"seconds":time.monotonic()-started,"events":events,
                "initial_memory_sha256":initial_hash,"reference_report_sha256":protocol["reference_report_sha256"],
                "plan_sha256":protocol["plan_sha256"],"pool":protocol["pool"],"interpretation":protocol["interpretation"]}
        atomic_json(folder/"report.json",report);atomic_json(folder/"view.json",lab.export_view(folder,report))
        reports[name]=report
        summary[name]={"actions":[e["side"] for e in events],"gate_spikes":[e["gate_spikes"] for e in events],
                       "difference_hz":[e["difference_hz"] for e in events],
                       "weight_update_l2":[e["diagnostics"]["weight_delta_l2"] for e in events]}
        print(f"pulse_study arm={name} actions={summary[name]['actions']}",flush=True)
    result={"status":"market_pulse_study_completed","protocol":protocol,"training_events":training,"summary":summary,"reports":reports,
            "verification":{"same_native_build":True,"training_reproduced":True,"both_reference_controls_reproduced":True,
                            "all_frozen_memory_preserved":True,"identical_test_images":True},
            "code_sha256":{name:digest(Path(__file__).with_name(name)) for name in ("fly_market_pulse.py","fly_trace.py","fly.py","fly_market_study.py")}}
    atomic_json(root/"summary.json",result)
    return result


def cloud_run(payload, output):
    """A durable observer: an existing receipt always resumes the same call."""
    import modal
    import uuid
    validate(payload)
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    receipt_path=root/"cloud-call.json"
    if not receipt_path.exists():
        receipt={"run_id":"assay-pulse-"+uuid.uuid4().hex,"status":"submitting","payload_sha256":signature(payload)}
        atomic_json(receipt_path,receipt)
        atomic_json(root/"source-hashes.json",{name:digest(Path(__file__).with_name(name)) for name in
                    ("fly_market_pulse.py","fly_trace.py","fly.py","fly_market_study.py")})
        call=modal.Function.from_name("fly-paper-lab","worker",environment_name="main").spawn(
            debug={"run_id":receipt["run_id"],"pulse_plan":payload})
        receipt.update(status="pending",call_id=call.object_id);atomic_json(receipt_path,receipt)
    receipt=json.loads(receipt_path.read_text())
    if receipt["payload_sha256"]!=signature(payload):raise ValueError("Output directory belongs to another payload")
    if "call_id" not in receipt:raise RuntimeError("Submission outcome unknown; inspect Modal before retrying")
    print(f"Observing saved call {receipt['call_id']}",flush=True)
    try:result=modal.FunctionCall.from_id(receipt["call_id"]).get(timeout=50)
    except TimeoutError:
        print("Still pending. Repeat this command to observe the same call; it will not resubmit.",flush=True)
        return
    atomic_json(root/"cloud-result.json",result)
    if result.get("status")!="market_pulse_study_completed" or result.get("run_id")!=receipt["run_id"]:
        raise RuntimeError("Unexpected result; no automatic resubmission")
    report=result["report"]
    if report["protocol"]!=payload["protocol"] or report["code_sha256"]!=json.loads((root/"source-hashes.json").read_text()):
        raise ValueError("Returned protocol or executed source differs from submission")
    volume=modal.Volume.from_name("fly-paper-lab-state",environment_name="main")
    for arm in ARMS:
        folder=root/arm;folder.mkdir(exist_ok=True)
        remote=result["remote_path"]+"/"+arm
        view=folder/"view.json"
        if not view.exists():
            partial=view.with_suffix(".partial")
            with partial.open("wb") as stream:
                for block in volume.read_file(remote.removeprefix("/state")+"/view.json"):stream.write(block)
            partial.replace(view)
        atomic_json(folder/"report.json",report["reports"][arm])
        atomic_json(folder/"remote.json",{"remote_path":remote,"call_id":receipt["call_id"]})
    atomic_json(root/"summary.json",report)
    receipt.update(status="completed",budget=result["budget"]);atomic_json(receipt_path,receipt)
    print(json.dumps({"summary":report["summary"],"verification":report["verification"],"budget":result["budget"]},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    pack=sub.add_parser("pack")
    for field in ("protocol","reference","plan","out"):pack.add_argument("--"+field,type=Path,required=True)
    for command in ("run","cloud"):
        cmd=sub.add_parser(command);cmd.add_argument("--payload",type=Path,required=True);cmd.add_argument("--out",type=Path,required=True)
        if command=="run":cmd.add_argument("--fly-data",type=Path,required=True)
    a=p.parse_args()
    if a.command=="pack":
        payload={"protocol":json.loads(a.protocol.read_text()),"reference_json":a.reference.read_text(),"market_plan":json.loads(a.plan.read_text())}
        validate(payload);atomic_json(a.out,payload)
    else:
        payload=json.loads(a.payload.read_text())
        if a.command=="cloud":cloud_run(payload,a.out)
        else:run(payload,a.fly_data,a.out)


if __name__=="__main__":main()
