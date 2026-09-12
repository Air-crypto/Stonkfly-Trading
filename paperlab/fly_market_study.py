"""Sealed, chronological fly comparisons on a discovery snapshot. Paper only."""
import argparse
from bisect import bisect_right
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time

import numpy as np

from .core import Broker, Tick, atomic_json, digest
from .fly import frame, UPSTREAM_COMMIT
from .fly_trace import TraceLab
from .fly_visual import apply_view, encoding
from .multi import DEX_COSTS, read_archive
from .news import News
from .universe import MIN_CONTEXT, Pool

ARMS = {
    "pristine_frozen": {"eta": .001, "train": False, "online": False, "view": "original"},
    "trained_frozen": {"eta": .001, "train": True, "online": False, "view": "original"},
    "online_original": {"eta": .001, "train": True, "online": True, "view": "original"},
    "lower_eta_frozen": {"eta": .0001, "train": True, "online": False, "view": "original"},
    "price_only_frozen": {"eta": .001, "train": True, "online": False, "view": "price_only"},
}

REINFORCEMENT_ARMS = {
    "pristine_frozen": ARMS["pristine_frozen"],
    "online_original": ARMS["online_original"],
    "reinforcement_gated": {**ARMS["online_original"], "reinforcement_only": True},
}

VISUAL_ARMS = {
    "pristine_frozen": ARMS["pristine_frozen"],
    "online_original": ARMS["online_original"],
    "fixed_returns_frozen": {**ARMS["pristine_frozen"], "view": "fixed_returns"},
    "fixed_returns_online": {**ARMS["online_original"], "view": "fixed_returns"},
}


def signature(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def seal(archive, output, phase_steps=3):
    """Cohort selection uses past eligibility, never later price/return/survival."""
    if type(phase_steps) is not int or not 2 <= phase_steps <= 4:
        raise ValueError("Use 2–4 decisions in each bounded phase")
    archive=Path(archive)
    if Path(output).exists():
        raise ValueError("Refuse to overwrite a sealed plan")
    db=sqlite3.connect(f"file:{archive.resolve()}?mode=ro",uri=True)
    counts, recent, admitted = {}, {}, {}
    last=0
    try:
        for key,raw in db.execute("SELECT key,payload FROM observations ORDER BY slot,key"):
            p=Pool(**json.loads(raw)); last=max(last,p.observed)
            available=p.rejection() is None
            counts[key]=counts.get(key,0)+int(available)
            recent[key]=(recent.get(key,[])+[available])[-3:]
            if counts[key]>=MIN_CONTEXT and all(recent[key]) and key not in admitted:
                admitted[key]=p.observed
    finally:
        db.close()
    if not admitted:
        raise ValueError("No historically qualified pools yet")
    # Allow a fixed 15-minute cohort formation period, then seal at a 5-minute boundary.
    start=math.ceil((min(admitted.values())+900)/300)*300
    keys=sorted((k for k,t in admitted.items() if t<=start),key=lambda k:(admitted[k],k))[:2]
    end=start+3*phase_steps*300
    if last<end:
        raise ValueError(f"Need observations through {end}; snapshot ends at {last}")
    _,series=read_archive(archive,last)
    sequences={k:[asdict(t) for t in series[k] if t.ts<=end] for k in keys}
    plan={"schema":1,"snapshot_sha256":digest(archive),"snapshot_end":last,"start":start,
          "phase_steps":phase_steps,"decision_seconds":300,"costs":asdict(DEX_COSTS),
          "cohort":keys,"admission_times":{k:admitted[k] for k in keys},"series":sequences,
          "arms":ARMS,"news":"disabled in all arms; no reconstructed or backdated headlines",
          "selection":"Earliest eligible pools within fixed 15-minute formation period, at most two of four sleeves. No future survival filter.",
          "promotion_rule":"Choose by development total equity only, requiring improvement over both cash and pristine frozen; evaluate once on subsequent test. No automatic deployment."}
    validate(plan)
    envelope={"plan":plan,"sha256":signature(plan)}
    atomic_json(output,envelope)
    return envelope


def validate(plan):
    protocols = {1: ARMS, 2: REINFORCEMENT_ARMS, 3: VISUAL_ARMS}
    if not isinstance(plan,dict) or plan.get("schema") not in protocols or plan.get("arms")!=protocols[plan["schema"]]:
        raise ValueError("Unknown study protocol")
    n=plan.get("phase_steps")
    if type(n) is not int or not 2<=n<=4 or plan.get("decision_seconds")!=300:
        raise ValueError("Invalid study horizon")
    if plan.get("costs")!=asdict(DEX_COSTS):
        raise ValueError("Study must retain the declared adverse DEX costs")
    if plan["schema"]==3 and plan.get("visual_encoding")!=encoding("fixed_returns"):
        raise ValueError("Visual protocol must pin the executed sensory transform")
    if not 1<=len(plan.get("cohort",[]))<=2 or len(set(plan["cohort"]))!=len(plan["cohort"]):
        raise ValueError("Use one or two unique admitted pools")
    if set(plan["series"])!=set(plan["cohort"]):
        raise ValueError("Cohort/series mismatch")
    start=plan["start"]
    if not math.isfinite(start) or start<=0 or start%300 or plan["snapshot_end"]<start+3*n*300:
        raise ValueError("Insufficient chronological horizon")
    if plan["schema"]>=2 and (plan.get("previous_test_end")!=start or
                              not isinstance(plan.get("parent_plan_sha256"),str) or len(plan["parent_plan_sha256"])!=64):
        raise ValueError("A follow-up must start after the recorded prior test interval")
    for key,raw in plan["series"].items():
        if not MIN_CONTEXT<=len(raw)<=512:
            raise ValueError("Each study series must have 64–512 bounded observations")
        ticks=[Tick(**r) for r in raw]
        if any(t.product!=key for t in ticks) or any(b.ts<=a.ts for a,b in zip(ticks,ticks[1:])):
            raise ValueError("Mixed or unordered observations")
        past=[t for t in ticks if t.ts<=start]
        if sum(t.available for t in past)<MIN_CONTEXT:
            raise ValueError("Cohort admission must precede the first decision")
        if any(type(t.available) is not bool or t.ts>start+3*n*300 for t in ticks):
            raise ValueError("Invalid availability or future observation")
    return plan


def seal_followup(archive, previous, output, phase_steps=2, protocol="reinforcement"):
    """Preserve the prior cohort and move every decision past its test endpoint."""
    previous = json.loads(Path(previous).read_text()) if isinstance(previous,(str,Path)) else previous
    old=validate(previous["plan"])
    if previous["sha256"]!=signature(old):
        raise ValueError("Previous sealed plan hash mismatch")
    if Path(output).exists():
        raise ValueError("Refuse to overwrite a sealed plan")
    if type(phase_steps) is not int or not 2<=phase_steps<=4:
        raise ValueError("Use 2–4 decisions per phase")
    if protocol not in ("reinforcement", "visual"):
        raise ValueError("Unknown follow-up protocol")
    start=old["start"]+3*old["phase_steps"]*300
    end=start+3*phase_steps*300
    archive=Path(archive)
    db=sqlite3.connect(f"file:{archive.resolve()}?mode=ro",uri=True)
    try:
        last=db.execute("SELECT max(json_extract(payload,'$.observed')) FROM observations").fetchone()[0]
    finally:
        db.close()
    if last is None or last<end:
        raise ValueError(f"Need observations through {end}; snapshot ends at {last}")
    _,series=read_archive(archive,last)
    if not all(key in series for key in old["cohort"]):
        raise ValueError("Prior cohort fell outside the bounded archive; do not replace it with survivors")
    plan={**old,"schema":3 if protocol=="visual" else 2,"start":start,"previous_test_end":start,"parent_plan_sha256":previous["sha256"],
          "phase_steps":phase_steps,"snapshot_sha256":digest(archive),"snapshot_end":last,
          "arms":VISUAL_ARMS if protocol=="visual" else REINFORCEMENT_ARMS,
          "series":{key:[asdict(t) for t in series[key] if t.ts<=end] for key in old["cohort"]},
          "selection":"Same cohort as parent study, regardless of later eligibility or survival. New decision windows start at parent test end.",
          "hypothesis":"Freeze efficacy and weight updates when reinforcement is none; retain neural propagation and rate traces. No decoder or risk-limit change."}
    if protocol=="visual":
        plan["hypothesis"]="Compare fixed return encoding with the unchanged visual adapter, crossed with frozen versus online plasticity. No decoder or risk-limit change."
        plan["visual_encoding"]=encoding("fixed_returns")
    validate(plan)
    envelope={"plan":plan,"sha256":signature(plan)}
    atomic_json(output,envelope)
    return envelope


def quote_at(ticks, stamp):
    index=bisect_right([t.ts for t in ticks],stamp)-1
    if index<0:
        raise ValueError("Missing prior context")
    quote=ticks[index]
    if stamp-quote.ts>180:
        quote=replace(quote,available=False)
    return index,quote


def decision_timeline(rows):
    """Expose every decision slot, including gaps with no neural capture."""
    return [{"decision_ts": r["decision_ts"], "quote_ts": r["quote_ts"],
             "available": r["available"], "terminal": r["terminal"],
             "action": r["event"]["side"] if r["event"] else None,
             "observation": ("terminal_mark" if r["terminal"] else
                             "observed" if r["event"] else
                             "unavailable_quote" if not r["available"] else "no_new_quote"),
             "fill_status": r["fill"]["status"], "fill_reason": r["fill"].get("reason"),
             "equity": r["equity"]} for r in rows]


def phase(lab, ticks, start, steps, arm, learning, deadline, trace_output=None):
    b=lab.brain
    b.weights_frozen=not learning
    b.eta=arm["eta"]
    lab.fly.controller.s=replace(lab.fly.controller.s,learning=learning)
    broker=Broker(DEX_COSTS)
    news=News(enabled=False)
    pending=None; anchor=DEX_COSTS.capital; unpriced=False; last_quote=0; rows=[]
    trace_events=[]; wall_start=time.monotonic()
    if trace_output is not None:
        trace_output.mkdir(parents=True,exist_ok=False)
    try:
        for step in range(steps+1):
            if time.monotonic()>deadline:
                raise TimeoutError("Bounded market study reached its diagnostic time limit")
            stamp=start+step*300
            index,t=quote_at(ticks,stamp)
            fill=broker.execute(*pending,t) if pending else {"status":"hold"}
            pending=None
            equity=broker.equity(t)
            event=None
            if step<steps and t.available and t.ts>last_quote:
                reward=0 if unpriced else equity-anchor
                rgb=apply_view(frame(ticks,index,news),ticks,index,arm["view"])
                stimulus=("reward" if reward>.01 else "aversive" if reward<-.01 else "none") if learning else "none"
                enabled=learning and (not arm.get("reinforcement_only",False) or stimulus!="none")
                b.weights_frozen=not enabled
                lab.fly.controller.s=replace(lab.fly.controller.s,learning=enabled)
                before=b.weight[b.circuit["edges"]].copy()
                event=(lab.capture(rgb,stimulus,trace_output,len(trace_events)+1)
                       if trace_output is not None else lab.fly.controller.observe(rgb,stimulus))
                if trace_output is not None:
                    trace_events.append(event)
                    event["market_decision_ts"]=stamp
                event["equity_reward_usd"]=reward if learning else 0
                event["plasticity_enabled"]=enabled
                event["weight_delta_l2"]=float(np.linalg.norm(b.weight[b.circuit["edges"]]-before))
                if event["side"]!="HOLD":
                    pending=(.5 if event["side"]=="BUY" else 0,stamp)
                last_quote=t.ts
                anchor=equity
            unpriced=not t.available
            rows.append({"decision_ts":stamp,"quote_ts":t.ts,"available":t.available,"equity":equity,
                         "fill":fill,"event":event,"broker":broker.state(),"terminal":step==steps})
        if trace_events:
            report={"schema":1,"source":"sealed_retrospective_market_replay",
                    "config":{"preset":"market_replay","view":arm["view"],"news":"none","eta":arm["eta"],"learning":learning,
                              "reinforcement_only":arm.get("reinforcement_only",False)},
                    "upstream_commit":UPSTREAM_COMMIT,"graph":{"neurons":b.n,"edges":len(b.post),"plastic_edges":len(b.circuit["edges"])},
                    "native_build":b.build,"seconds":time.monotonic()-wall_start,"events":trace_events,
                    "market_timeline":decision_timeline(rows),"visual_encoding":encoding(arm["view"]),
                    "interpretation":"Recorded market replay, isolated paper account. No executable DEX or monthly-return claim."}
            atomic_json(trace_output/"report.json",report)
            atomic_json(trace_output/"view.json",lab.export_view(trace_output,report))
        return {"start":start,"end":start+steps*300,"equity":rows[-1]["equity"],
                "return_pct":100*(rows[-1]["equity"]/DEX_COSTS.capital-1),"fees":float(broker.fees),
                "fills":sum(r["fill"]["status"]=="filled" for r in rows),
                "unavailable_marks":sum(not r["available"] for r in rows),"rows":rows}
    finally:
        news.db.close()


def learned_state(brain):
    return {"weights":brain.weight[brain.circuit["edges"]].copy(),"u":brain.memory_u.copy(),"w":brain.memory_w.copy()}


def restore_learned(brain, state):
    brain.reset()
    brain.weight[brain.circuit["edges"]]=state["weights"]
    brain.memory_u[:]=state["u"];brain.memory_w[:]=state["w"]


def run(envelope, data, output):
    plan=validate(envelope["plan"])
    if envelope.get("sha256")!=signature(plan):
        raise ValueError("Sealed plan hash mismatch")
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    atomic_json(root/"protocol.json",envelope)
    deadline=time.monotonic()+480
    lab=TraceLab(data); n=plan["phase_steps"]; arms=plan["arms"]
    results={}; states={}
    for key in plan["cohort"]:
        ticks=[Tick(**r) for r in plan["series"][key]]
        results[key]={}
        for name,arm in arms.items():
            lab.brain.reset()
            training=phase(lab,ticks,plan["start"],n,arm,arm["train"],deadline)
            state=learned_state(lab.brain);states[key,name]=state
            restore_learned(lab.brain,state)
            development=phase(lab,ticks,plan["start"]+n*300,n,arm,arm["online"],deadline)
            results[key][name]={"training":training,"development":development}
            print(f"market_study development pool={plan['cohort'].index(key)} arm={name} return={development['return_pct']:.4f}%",flush=True)
            atomic_json(root/"development.json",results)
    idle=(4-len(plan["cohort"]))*DEX_COSTS.capital
    dev_equity={name:idle+sum(results[k][name]["development"]["equity"] for k in results) for name in arms}
    candidate=max(arms,key=lambda name:(dev_equity[name],name))
    selected=candidate if dev_equity[candidate]>max(1000,dev_equity["pristine_frozen"])+1e-9 else None
    selection={"selected":selected,"development_equity":dev_equity,"cash_equity":1000,
               "plan_sha256":envelope["sha256"],"test_simulated_before_selection":False,
               "note":"No deployment. Selection fixed from development outcomes before test simulation."}
    # This file is written before the first test-phase model call.
    atomic_json(root/"selection.json",selection)
    for key in plan["cohort"]:
        ticks=[Tick(**r) for r in plan["series"][key]]
        for name,arm in arms.items():
            # Learned weights always come from training, not a test-tuned or development-updated state.
            restore_learned(lab.brain,states[key,name])
            traces=(root/f"pool{plan['cohort'].index(key)}-{name}") if name in ("pristine_frozen","online_original","reinforcement_gated","fixed_returns_frozen","fixed_returns_online") else None
            results[key][name]["test"]=phase(lab,ticks,plan["start"]+2*n*300,n,arm,arm["online"],deadline,traces)
            print(f"market_study test pool={plan['cohort'].index(key)} arm={name} return={results[key][name]['test']['return_pct']:.4f}%",flush=True)
            atomic_json(root/"results.json",results)
    totals={name:{phase_name:idle+sum(results[k][name][phase_name]["equity"] for k in results)
                  for phase_name in ("training","development","test")} for name in arms}
    elapsed=n*300
    summary={"status":"market_study_completed","plan_sha256":envelope["sha256"],"selection":selection,
             "code_sha256":{name:digest(Path(__file__).with_name(name)) for name in ("fly_market_study.py","fly_trace.py","fly.py","fly_visual.py")},
             "total_equity":totals,"initial_capital":1000,"active_sleeves":len(plan["cohort"]),
             "test_net_after_monthly_hosting":{str(cost):{name:totals[name]["test"]-1000-cost*elapsed/(30*86400)
                 for name in arms} for cost in (20,40)},"costs":asdict(DEX_COSTS),"phase_steps":n,
             "graph":{"neurons":lab.brain.n,"edges":len(lab.brain.post)},"native_build":lab.brain.build,
             "limitation":"Short sealed retrospective replay, not an estimate of monthly returns or executable DEX performance."
             " Cash resets at each phase; neural dynamics reset while training weights are restored. News disabled."
             " Test results must not be reused to choose another variant. No policy automatically promoted."}
    atomic_json(root/"summary.json",summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    s=sub.add_parser("seal");s.add_argument("--archive",required=True);s.add_argument("--out",required=True);s.add_argument("--phase-steps",type=int,default=3)
    f=sub.add_parser("followup");f.add_argument("--archive",required=True);f.add_argument("--previous",required=True);f.add_argument("--out",required=True);f.add_argument("--phase-steps",type=int,default=2)
    f.add_argument("--protocol",choices=("reinforcement","visual"),default="reinforcement")
    r=sub.add_parser("run");r.add_argument("--plan",required=True);r.add_argument("--fly-data",required=True);r.add_argument("--out",required=True)
    a=parser.parse_args()
    if a.command=="seal":
        result=seal(a.archive,a.out,a.phase_steps)
    elif a.command=="followup":
        result=seal_followup(a.archive,a.previous,a.out,a.phase_steps,a.protocol)
    else:
        result=run(json.loads(Path(a.plan).read_text()),a.fly_data,a.out)
    print(json.dumps({k:v for k,v in result.items() if k!="plan"},indent=2))


if __name__=="__main__":
    main()
