"""Sealed, chronological fly comparisons on a discovery snapshot. Paper only."""
import argparse
import copy
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
from .fly_market_input import INPUT_ARMS, validate_input, development_choice, source_files

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


INFERENCE_ARMS = {name: ARMS[name] for name in ("pristine_frozen", "trained_frozen", "online_original")}

RESTORATION_ARMS = {"pristine_frozen": ARMS["pristine_frozen"], "trained_frozen": ARMS["trained_frozen"],
                   **{name: {**ARMS["trained_frozen"], "restore_post_ids": ids} for name, ids in
                      (("restore_10704", ["10704"]), ("restore_11402", ["11402"]), ("restore_both", ["10704", "11402"]))}}
RESTORATION_TIMING = "Train normally from pristine. For development and test separately, reset neural activity and restore the saved training memory, then reset incoming weights/u/w of the declared target cells to pristine. Freeze all plasticity and omit reinforcement throughout each inference phase. Development state is never carried into test."


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
    protocols = {1: ARMS, 2: REINFORCEMENT_ARMS, 3: VISUAL_ARMS, 4: INFERENCE_ARMS, 5: RESTORATION_ARMS, 6: INPUT_ARMS}
    if not isinstance(plan,dict) or plan.get("schema") not in protocols or plan.get("arms")!=protocols[plan["schema"]]:
        raise ValueError("Unknown study protocol")
    n=plan.get("phase_steps")
    if type(n) is not int or not 2<=n<=4 or plan.get("decision_seconds")!=300:
        raise ValueError("Invalid study horizon")
    if plan.get("costs")!=asdict(DEX_COSTS):
        raise ValueError("Study must retain the declared adverse DEX costs")
    if plan["schema"]==3 and plan.get("visual_encoding")!=encoding("fixed_returns"):
        raise ValueError("Visual protocol must pin the executed sensory transform")
    if plan["schema"]>=4 and "visual_encoding" in plan:
        raise ValueError("Frozen inference uses the original visual adapter")
    if plan["schema"]==5 and plan.get("restoration_timing")!=RESTORATION_TIMING:
        raise ValueError("Restoration timing must match the registered inference boundary")
    if plan["schema"]!=5 and "restoration_timing" in plan:
        raise ValueError("Unexpected restoration timing metadata")
    if plan["schema"]==6:validate_input(plan)
    elif "activity_reset_timing" in plan:raise ValueError("Unexpected activity reset timing")
    if not 1<=len(plan.get("cohort",[]))<=2 or len(set(plan["cohort"]))!=len(plan["cohort"]):
        raise ValueError("Use one or two unique admitted pools")
    if set(plan["series"])!=set(plan["cohort"]):
        raise ValueError("Cohort/series mismatch")
    start=plan["start"]
    if not math.isfinite(start) or start<=0 or start%300 or plan["snapshot_end"]<start+3*n*300:
        raise ValueError("Insufficient chronological horizon")
    if plan["schema"]>=2 and ((plan.get("previous_test_end")!=start if plan["schema"]!=6 else plan.get("previous_test_end",start+1)>start) or
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
    if protocol not in ("reinforcement", "visual", "frozen-inference", "memory-restoration"):
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
    schema,arms={"reinforcement":(2,REINFORCEMENT_ARMS),"visual":(3,VISUAL_ARMS),"frozen-inference":(4,INFERENCE_ARMS),"memory-restoration":(5,RESTORATION_ARMS)}[protocol]
    plan={**old,"schema":schema,"start":start,"previous_test_end":start,"parent_plan_sha256":previous["sha256"],
          "phase_steps":phase_steps,"snapshot_sha256":digest(archive),"snapshot_end":last,
          "arms":arms,
          "series":{key:[asdict(t) for t in series[key] if t.ts<=end] for key in old["cohort"]},
          "selection":"Same cohort as parent study, regardless of later eligibility or survival. New decision windows start at parent test end.",
          "hypothesis":"Freeze efficacy and weight updates when reinforcement is none; retain neural propagation and rate traces. No decoder or risk-limit change."}
    plan.pop("restoration_timing",None)
    if protocol=="memory-restoration":
        plan["restoration_timing"]=RESTORATION_TIMING
        plan["hypothesis"]="Compare partial and combined restoration of incoming MBON11 memory with trained and pristine frozen controls on a subsequent market window."
    if protocol=="visual":
        plan["hypothesis"]="Compare fixed return encoding with the unchanged visual adapter, crossed with frozen versus online plasticity. No decoder or risk-limit change."
        plan["visual_encoding"]=encoding("fixed_returns")
    else:
        plan.pop("visual_encoding",None)
        if protocol=="frozen-inference":
            plan["hypothesis"]="Train on the prior training phase, then freeze weights and omit reinforcement at inference. Compare with pristine frozen and original online behavior; retain the full graph, decoder, cohort and costs."
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


def phase(lab, ticks, start, steps, arm, learning, deadline, trace_output=None, restoration=None, activity_output=None, news=None, record_end_state=False):
    if record_end_state and (trace_output is None or activity_output is None):
        raise ValueError('End-state recording requires full traces and boundary records')
    if activity_output is not None and learning and arm.get('activity_reset') not in ('carry','reset_rates'):
        raise ValueError('Online boundary recording supports only carry or learning-rate trace resets')
    b=lab.brain
    b.weights_frozen=not learning
    b.eta=arm["eta"]
    initial_memory=learned_state(b)
    lab.fly.controller.s=replace(lab.fly.controller.s,learning=learning)
    broker=Broker(DEX_COSTS)
    owns_news=news is None
    news=News(enabled=False) if owns_news else news
    pending=None; anchor=DEX_COSTS.capital; unpriced=False; last_quote=0; rows=[]
    trace_events=[]; observations=0; wall_start=time.monotonic()
    if activity_output is not None:
        activity_output.mkdir(parents=True,exist_ok=False)
        from .fly_market_activity import apply_boundary, attach_boundary_state
    if trace_output is not None:
        trace_output.mkdir(parents=True,exist_ok=False)
        np.savez_compressed(trace_output/"initial-memory.npz",**initial_memory)
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
                boundary=None
                if activity_output is not None:
                    boundary=apply_boundary(b,arm["activity_reset"],observations+1,activity_output/f"boundary-{observations+1:02}.npz")
                before=b.weight[b.circuit["edges"]].copy()
                frozen_memory=learned_state(b) if not enabled else None
                if 'recipient_current' in arm:
                    from .fly_paper_stimulation import AppliedCurrent
                    if learning or trace_output is None or arm.get('recipient_ids')!=['10704','11402'] or arm['recipient_current'] not in (0,10):
                        raise ValueError('Recipient activation requires bounded frozen recording')
                    targets=[lab.id_index[x] for x in arm['recipient_ids']]
                    if any(str(lab.types[i])!='MBON11' for i in targets):raise ValueError('Recipient identity differs')
                    with AppliedCurrent(b,targets,arm['recipient_current']) as applied:
                        event=lab.capture(rgb,'none',trace_output,len(trace_events)+1,targets,arm['recipient_current'])
                    current_path=trace_output/f'current-{len(trace_events)+1:02}.npz';applied.save(current_path)
                    event['stimulation']={'target_ids':arm['recipient_ids'],'current':arm['recipient_current'],
                                          'duration_ms':500,'artifact_sha256':digest(current_path)}
                else:
                    event=(lab.capture(rgb,stimulus,trace_output,len(trace_events)+1)
                           if trace_output is not None else lab.fly.controller.observe(rgb,stimulus))
                observations+=1
                if record_end_state:
                    from .fly_market_activity import dynamic_state, array_hash
                    end_path=activity_output/f'end-{observations:02}.npz'
                    np.savez_compressed(end_path,**dynamic_state(b))
                    event['end_state_sha256']=digest(end_path)
                    event['all_weight_sha256']=array_hash(b.weight)
                if boundary is not None:event["activity_boundary"]=boundary
                if frozen_memory is not None:
                    after_memory=learned_state(b)
                    if not all(np.array_equal(v,after_memory[k]) for k,v in frozen_memory.items()):
                        raise AssertionError("Frozen market inference changed synaptic memory")
                if trace_output is not None:
                    trace_events.append(event)
                    event["market_decision_ts"]=stamp
                event["equity_reward_usd"]=reward if learning else 0
                if not owns_news:event['news_features']=news.features(t.ts).tolist()
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
                    "config":{"preset":"market_replay","view":arm["view"],"news":getattr(news,'label','none'),"eta":arm["eta"],"learning":learning,
                              "reinforcement_only":arm.get("reinforcement_only",False),"restore_post_ids":(restoration or {}).get("post_ids",[])},
                    "upstream_commit":UPSTREAM_COMMIT,"graph":{"neurons":b.n,"edges":len(b.post),"plastic_edges":len(b.circuit["edges"])},
                    "native_build":b.build,"seconds":time.monotonic()-wall_start,"events":trace_events,
                    "market_timeline":decision_timeline(rows),"visual_encoding":encoding(arm["view"]),"restoration":restoration,
                    "initial_memory_sha256":memory_signature(initial_memory),"final_memory_sha256":memory_signature(learned_state(b)),
                    "interpretation":"Recorded market replay, isolated paper account. No executable DEX or monthly-return claim."}
            requested=[]
            if 'recipient_current' in arm:
                report['config'].update(recipient_current=arm['recipient_current'],recipient_ids=arm['recipient_ids'])
                requested=[lab.id_index[x] for x in arm['recipient_ids']]
            atomic_json(trace_output/"report.json",report)
            view=lab.export_view(trace_output,report,requested)
            if activity_output is not None:
                report["config"]["activity_reset"]=arm["activity_reset"]
                atomic_json(trace_output/"report.json",report)
                attach_boundary_state(view,activity_output)
            atomic_json(trace_output/"view.json",view)
        return {"start":start,"end":start+steps*300,"equity":rows[-1]["equity"],
                "return_pct":100*(rows[-1]["equity"]/DEX_COSTS.capital-1),"fees":float(broker.fees),
                "fills":sum(r["fill"]["status"]=="filled" for r in rows),
                "unavailable_marks":sum(not r["available"] for r in rows),"rows":rows,"restoration":restoration,
                "initial_memory_sha256":memory_signature(initial_memory),"final_memory_sha256":memory_signature(learned_state(b))}
    finally:
        if owns_news:news.db.close()


def learned_state(brain):
    return {"weights":brain.weight[brain.circuit["edges"]].copy(),"u":brain.memory_u.copy(),"w":brain.memory_w.copy()}


def memory_signature(state):
    return hashlib.sha256(state["weights"].tobytes()+state["u"].tobytes()+state["w"].tobytes()).hexdigest()


def restore_learned(brain, state):
    brain.reset()
    brain.weight[brain.circuit["edges"]]=state["weights"]
    brain.memory_u[:]=state["u"];brain.memory_w[:]=state["w"]



def inference_memory(brain, trained, pristine, arm):
    """Derive a new immutable inference checkpoint without altering training memory."""
    from .fly_market_restoration import restore_posts
    targets=arm.get("restore_post_ids",[])
    post_ids=np.array([str(brain.ids[i]) for i in brain.post[brain.circuit["edges"]]])
    initial,mask=restore_posts(pristine,trained,post_ids,targets)
    return initial,{"post_ids":targets,"edge_ids":[str(e) for e in brain.circuit["edges"][mask]],
                   "edge_count":int(mask.sum()),"training_memory_sha256":memory_signature(trained),
                   "inference_memory_sha256":memory_signature(initial)}


def run(envelope, data, output):
    plan=validate(envelope["plan"])
    if envelope.get("sha256")!=signature(plan):
        raise ValueError("Sealed plan hash mismatch")
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    atomic_json(root/"protocol.json",envelope)
    deadline=time.monotonic()+480
    lab=TraceLab(data); n=plan["phase_steps"]; arms=plan["arms"]
    results={}; states={}; restorations={}
    if plan["schema"]==6:
        from .fly_market_activity import dynamic_state
        np.savez_compressed(root/"initial-dynamics.npz",**dynamic_state(lab.brain))
        np.savez_compressed(root/"neuron-ids.npz",neuron_ids=lab.brain.ids)
    if plan["schema"]==5:
        np.savez_compressed(root/"plastic-map.npz",edge_ids=lab.brain.circuit["edges"],
                            post_ids=lab.brain.ids[lab.brain.post[lab.brain.circuit["edges"]]])
    for key in plan["cohort"]:
        ticks=[Tick(**r) for r in plan["series"][key]]
        results[key]={};training_cache={}
        for name,arm in arms.items():
            lab.brain.reset()
            pristine=learned_state(lab.brain)
            recipe=(arm["eta"],arm["train"],arm["view"],arm.get("reinforcement_only",False))
            if plan["schema"] in (5,6) and recipe in training_cache:
                source,recorded,cached=training_cache[recipe]
                training=copy.deepcopy(recorded);trained={k:v.copy() for k,v in cached.items()}
                training["training_compute_source"]=source;training["training_compute_reused"]=True
            else:
                training=phase(lab,ticks,plan["start"],n,arm,arm["train"],deadline)
                trained=learned_state(lab.brain)
                if plan["schema"] in (5,6):
                    training["training_compute_source"]=name;training["training_compute_reused"]=False
                    training_cache[recipe]=(name,copy.deepcopy(training),{k:v.copy() for k,v in trained.items()})
            state=trained;restoration=None
            if plan["schema"]==5:
                state,restoration=inference_memory(lab.brain,trained,pristine,arm)
                np.savez_compressed(root/f"pool{plan['cohort'].index(key)}-{name}-inference-memory.npz",**state)
            states[key,name]=state;restorations[key,name]=restoration
            np.savez_compressed(root/f"pool{plan['cohort'].index(key)}-{name}-training-memory.npz",**trained)
            restore_learned(lab.brain,state)
            development=phase(lab,ticks,plan["start"]+n*300,n,arm,arm["online"],deadline,restoration=restoration,**({"activity_output":root/f"boundaries/pool{plan['cohort'].index(key)}-{name}-development"} if plan["schema"]==6 else {}))
            results[key][name]={"training":training,"development":development}
            print(f"market_study development pool={plan['cohort'].index(key)} arm={name} return={development['return_pct']:.4f}%",flush=True)
            atomic_json(root/"development.json",results)
    idle=(4-len(plan["cohort"]))*DEX_COSTS.capital
    dev_equity={name:idle+sum(results[k][name]["development"]["equity"] for k in results) for name in arms}
    selected=development_choice(plan,dev_equity)
    selection={"selected":selected,"development_equity":dev_equity,"cash_equity":1000,
               "plan_sha256":envelope["sha256"],"test_simulated_before_selection":False,
               "note":"No deployment. Selection fixed from development outcomes before test simulation."}
    # This file is written before the first test-phase model call.
    atomic_json(root/"selection.json",selection)
    for key in plan["cohort"]:
        ticks=[Tick(**r) for r in plan["series"][key]]
        for name,arm in arms.items():
            # Restore the separately prepared inference state; development never enters test.
            restore_learned(lab.brain,states[key,name])
            traces=(root/f"pool{plan['cohort'].index(key)}-{name}") if plan["schema"] in (5,6) or name in ("pristine_frozen","trained_frozen","online_original","reinforcement_gated","fixed_returns_frozen","fixed_returns_online") else None
            results[key][name]["test"]=phase(lab,ticks,plan["start"]+2*n*300,n,arm,arm["online"],deadline,traces,restoration=restorations[key,name],**({"activity_output":root/f"boundaries/pool{plan['cohort'].index(key)}-{name}-test"} if plan["schema"]==6 else {}))
            print(f"market_study test pool={plan['cohort'].index(key)} arm={name} return={results[key][name]['test']['return_pct']:.4f}%",flush=True)
            atomic_json(root/"results.json",results)
    totals={name:{phase_name:idle+sum(results[k][name][phase_name]["equity"] for k in results)
                  for phase_name in ("training","development","test")} for name in arms}
    elapsed=n*300
    summary={"status":"market_study_completed","plan_sha256":envelope["sha256"],"selection":selection,
             "code_sha256":{name:digest(Path(__file__).with_name(name)) for name in source_files(plan["schema"])},
             "restoration_timing":plan.get("restoration_timing"),"total_equity":totals,"initial_capital":1000,"active_sleeves":len(plan["cohort"]),
             "test_net_after_monthly_hosting":{str(cost):{name:totals[name]["test"]-1000-cost*elapsed/(30*86400)
                 for name in arms} for cost in (20,40)},"costs":asdict(DEX_COSTS),"phase_steps":n,
             "graph":{"neurons":lab.brain.n,"edges":len(lab.brain.post)},"native_build":lab.brain.build,
             "limitation":"Short sealed retrospective replay, not an estimate of monthly returns or executable DEX performance."
             " Cash resets at each phase; neural dynamics reset while training weights are restored. News disabled."
             " Test results must not be reused to choose another variant. No policy automatically promoted."}
    if plan["schema"]==6:
        summary.update(activity_reset_timing=plan["activity_reset_timing"],initial_dynamics_sha256=digest(root/"initial-dynamics.npz"),neuron_ids_sha256=digest(root/"neuron-ids.npz"))
    atomic_json(root/"summary.json",summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    s=sub.add_parser("seal");s.add_argument("--archive",required=True);s.add_argument("--out",required=True);s.add_argument("--phase-steps",type=int,default=3)
    f=sub.add_parser("followup");f.add_argument("--archive",required=True);f.add_argument("--previous",required=True);f.add_argument("--out",required=True);f.add_argument("--phase-steps",type=int,default=2)
    f.add_argument("--protocol",choices=("reinforcement","visual","frozen-inference","memory-restoration"),default="reinforcement")
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
