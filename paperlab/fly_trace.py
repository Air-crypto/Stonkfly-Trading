"""Full-network diagnostic assays. Never reads or writes a paper account.

Capture wraps the existing 10 ms integration boundary without changing the native
kernel. Edge highlights mean source spikes, not measured synaptic transmission.
"""
from dataclasses import asdict, dataclass, replace
import base64
import hashlib
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image

from .core import Tick, atomic_json
from .fly import Fly, UPSTREAM_COMMIT, frame
from .news import News


@dataclass(frozen=True)
class Assay:
    preset: str = "rise"
    steps: int = 4
    probe_steps: int = 0
    probe_preset: str = "rise"
    learning: bool = True
    reinforcement_only: bool = False
    eta: float = .001
    reinforcement: str = "none"
    view: str = "original"
    news: str = "none"
    neurons: tuple[str, ...] = ()
    current: float = 0

    def __post_init__(self):
        if self.preset not in ("rise", "fall", "flat", "reversal", "shock"):
            raise ValueError("Unknown price preset")
        if type(self.steps) is not int or not 1 <= self.steps <= 8:
            raise ValueError("Use 1–8 observations per assay")
        if type(self.probe_steps) is not int or not 0 <= self.probe_steps <= 4 or self.steps+self.probe_steps > 8:
            raise ValueError("Use 0–4 probe observations and at most eight observations in total")
        if self.probe_preset not in ("rise", "fall", "flat", "reversal", "shock"):
            raise ValueError("Unknown probe price preset")
        if type(self.learning) is not bool:
            raise ValueError("learning must be a boolean")
        if type(self.reinforcement_only) is not bool:
            raise ValueError("reinforcement_only must be a boolean")
        if not math.isfinite(self.eta) or not 0 <= self.eta <= .002:
            raise ValueError("eta must be between 0 and .002")
        if self.reinforcement not in ("none", "reward", "aversive", "alternating"):
            raise ValueError("Unknown reinforcement")
        if self.view not in ("original", "price_only", "blank"):
            raise ValueError("Unknown visual ablation")
        if self.news not in ("none", "positive", "negative"):
            raise ValueError("Unknown synthetic news preset")
        if not isinstance(self.neurons, (tuple, list)) or len(self.neurons) > 16:
            raise ValueError("At most 16 stimulation neuron IDs")
        if any(not isinstance(x, str) or not x.isdecimal() for x in self.neurons):
            raise ValueError("Neuron IDs must be decimal strings")
        if not math.isfinite(self.current) or not -20 <= self.current <= 20:
            raise ValueError("Diagnostic current must be between -20 and 20")
        if self.current and not self.neurons:
            raise ValueError("Choose neuron IDs for nonzero current")


def synthetic_ticks(preset, steps):
    """Fixed input fixture; prices are never chosen using model outputs."""
    x = np.arange(100 + steps, dtype=float)
    if preset == "rise":
        p = 1 + .002 * x
    elif preset == "fall":
        p = 1.3 - .002 * x
    elif preset == "flat":
        p = np.ones_like(x)
    elif preset == "reversal":
        p = 1 + .003 * np.minimum(x, 96) - .02 * np.maximum(0, x - 96)
    else:
        p = 1 + .001 * x - .18 * (x >= 100)
    return [Tick(1700000000 + i * 300, float(v * .995), float(v * 1.005),
                 product="SYNTHETIC", source="synthetic_assay") for i, v in enumerate(p)]


def input_frames(config):
    ticks = synthetic_ticks(config.preset, config.steps)
    news = News()
    if config.news != "none":
        news.add("https://example.invalid/fixture", "growth rally approved" if config.news == "positive"
                 else "hack crash loss", "synthetic", ticks[0].ts, ticks[0].ts)
    try:
        images = []
        for i in range(99, 99 + config.steps):
            rgb = frame(ticks, i, news)
            if config.view == "blank":
                rgb[:] = 128
            elif config.view == "price_only":
                rgb[:28] = (235, 240, 249)
                rgb[140:] = (235, 240, 249)
            images.append(rgb)
        return images
    finally:
        news.db.close()


class Recorder:
    """Record all neurons and all plastic edges, in exact integration-bin order."""
    def __init__(self, brain, extra_indices=(), current=0):
        self.brain = brain
        self.original = brain.rgb_step
        self.had_override = "rgb_step" in vars(brain)
        self.indices = np.asarray(extra_indices, dtype=np.int32)
        self.current = current
        self.rows = []
        self.initial_weights = brain.weight[brain.circuit["edges"]].copy()

    def __enter__(self):
        self.brain.rgb_step = self.step
        return self

    def __exit__(self, *exc):
        if self.had_override:
            self.brain.rgb_step = self.original
        else:
            # Restore class-method lookup; storing a bound method would retain
            # a brain -> bound method -> brain cycle and all its large arrays.
            del self.brain.rgb_step

    def step(self, rgb, duration_ms, **kwargs):
        if duration_ms > 10 + 1e-8:
            raise ValueError("Trace requires <=10 ms controller bins")
        if len(self.indices) and self.current:
            old = kwargs.get("stimulation")
            pulses = [] if old is None else list(old) if isinstance(old, list) else [old]
            kwargs["stimulation"] = pulses + [(self.indices, self.current)]
        counts, elapsed = self.original(rgb, duration_ms, **kwargs)
        b = self.brain
        self.rows.append({"ms": b.sim_ms, "counts": counts.copy(), "voltage": b.v.copy(),
                          "weights": b.weight[b.circuit["edges"]].copy(),
                          "kc": b.rate_kc.copy(), "dan": b.rate_dan.copy(),
                          "u": b.memory_u.copy(), "w": b.memory_w.copy()})
        return counts, elapsed

    def arrays(self):
        return {key: np.asarray([r[key] for r in self.rows]) for key in self.rows[0]}


class TraceLab:
    def __init__(self, data):
        self.fly = Fly(data)
        self.brain = self.fly.controller.brain
        from stonkfly.neural.common import annotations
        self.annotation = annotations(self.brain.ids)
        self.types = self.annotation.type.fillna("").to_numpy()
        self.id_index = {str(v): i for i, v in enumerate(self.brain.ids)}

    def run(self, config, output):
        """Each run starts at pristine native state, independent of previous runs."""
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        b = self.brain
        extra = []
        for identity in config.neurons:
            if identity not in self.id_index:
                raise ValueError(f"Unknown retained neuron ID: {identity}")
            extra.append(self.id_index[identity])
        b.reset()
        b.eta = config.eta
        b.weights_frozen = not config.learning
        self.fly.controller.s = replace(self.fly.controller.s, learning=config.learning)
        images = input_frames(config)
        events = []
        started = time.monotonic()
        for i, rgb in enumerate(images):
            reinforcement = config.reinforcement
            if reinforcement == "alternating":
                reinforcement = "reward" if i % 2 == 0 else "aversive"
            enabled = config.learning and (not config.reinforcement_only or reinforcement != "none")
            b.weights_frozen = not enabled
            self.fly.controller.s = replace(self.fly.controller.s, learning=enabled)
            event = self.capture(rgb, reinforcement, output, i+1, extra, config.current)
            event.update(phase="training" if config.probe_steps else "assay", phase_step=i+1,
                         input_preset=config.preset, input_news=config.news)
            events.append(event)
            print(f"assay step={i+1}/{config.steps} side={event['side']} "
                  f"spikes={event['total_spikes']} changed={event['diagnostics']['changed_edges']}", flush=True)
        boundary = None
        if config.probe_steps:
            weights=b.weight[b.circuit["edges"]].copy()
            memory=(b.memory_u.copy(),b.memory_w.copy())
            b.reset(keep_memory=True)
            if not np.array_equal(weights,b.weight[b.circuit["edges"]]) or not all(
                    np.array_equal(a,v) for a,v in zip(memory,(b.memory_u,b.memory_w))):
                raise AssertionError("Probe reset changed learned weights or efficacy memory")
            if not all(np.array_equal(getattr(b,k),v) for k,v in b.initial.items()
                       if k not in ("memory_u","memory_w")) or b.sim_ms != 0:
                raise AssertionError("Probe must reset neural dynamics and sensory/rate traces")
            boundary={"after_observation":config.steps,"dynamics_reset":True,
                      "weight_sha256":hashlib.sha256(weights.tobytes()).hexdigest(),
                      "memory_sha256":hashlib.sha256(memory[0].tobytes()+memory[1].tobytes()).hexdigest(),
                      "weight_delta_from_pristine_l2":float(np.linalg.norm(weights-b.baseline_plastic)),
                      "probe_learning":False,"probe_reinforcement":"none","probe_extra_current":0}
            b.weights_frozen=True
            self.fly.controller.s=replace(self.fly.controller.s,learning=False)
            probe=replace(config,preset=config.probe_preset,steps=config.probe_steps,probe_steps=0,
                          learning=False,reinforcement="none",neurons=(),current=0,news="none")
            for i,rgb in enumerate(input_frames(probe)):
                event=self.capture(rgb,"none",output,len(events)+1)
                event.update(phase="probe",phase_step=i+1,input_preset=probe.preset,input_news="none")
                events.append(event)
                if not np.array_equal(weights,b.weight[b.circuit["edges"]]):
                    raise AssertionError("Frozen probe changed trained weights")
                print(f"assay probe={i+1}/{config.probe_steps} side={event['side']} "
                      f"spikes={event['total_spikes']} changed={event['diagnostics']['changed_edges']}",flush=True)
        report = {"schema": 1, "source": "synthetic_full_network_assay",
                  "config": asdict(config), "upstream_commit": UPSTREAM_COMMIT,
                  "graph": {"neurons": b.n, "edges": len(b.post), "plastic_edges": len(b.circuit["edges"])},
                  "native_build": b.build, "seconds": time.monotonic() - started,
                  "events": events,"probe_boundary":boundary,
                  "interpretation": "Synthetic mechanics assay, no trading return. No optimizer loss or backprop gradient. "
                  "Source spike highlights are not a measurement of transmission or causality."}
        atomic_json(output / "report.json", report)
        view = self.export_view(output, report, extra)
        atomic_json(output / "view.json", view)
        return report

    def capture(self, rgb, reinforcement, output, index, extra=(), current=0):
        """Record one native observation without changing its integration semantics."""
        b = self.brain
        with Recorder(b, extra, current) as trace:
            event = self.fly.controller.observe(rgb, reinforcement)
        arrays = trace.arrays()
        if not np.array_equal(arrays["counts"].sum(axis=0), b.counts):
            raise AssertionError("Trace does not reproduce controller spike counts")
        delta = arrays["weights"][-1] - trace.initial_weights
        event["diagnostics"] = {
            "loss": None, "backprop_gradient": None,
            "plasticity_enabled": bool(self.fly.controller.s.learning and not b.weights_frozen),
            "changed_edges": int(np.count_nonzero(delta)),
            "weight_delta_l2": float(np.linalg.norm(delta)),
            "max_abs_delta": float(np.max(np.abs(delta))),
            "efficacy_min": float(1 + b.memory_w.min()),
            "efficacy_max": float(1 + b.memory_w.max()),
            "clipped_edges": int(np.count_nonzero((b.memory_w <= -.9) | (b.memory_w >= 1))),
        }
        np.savez_compressed(output / f"step-{index:02}.npz", **arrays,
                            initial_weights=trace.initial_weights, neuron_ids=b.ids,
                            plastic_edges=b.circuit["edges"], plastic_pre=b.circuit["pre"],
                            plastic_post=b.post[b.circuit["edges"]])
        Image.fromarray(rgb).save(output / f"input-{index:02}.png")
        return event

    def export_view(self, output, report, requested=()):
        """A bounded view of a full-network recording; selection never affects simulation."""
        b = self.brain
        edges = b.circuit["edges"]
        total = np.zeros(b.n, dtype=np.int64)
        for i in range(len(report["events"])):
            with np.load(output / f"step-{i+1:02}.npz") as a:
                total += a["counts"].sum(axis=0)
        chosen = set(int(x) for x in requested)
        groups = {"visual": np.unique(np.r_[b.retina, b.r8]), "KC": b.circuit["kc"],
                  "DAN": b.circuit["dan"], "MBON": b.circuit["mb"],
                  "output": np.unique(np.r_[self.fly.controller.decoder.left,
                      self.fly.controller.decoder.right, self.fly.controller.decoder.gate])}
        group_lookup = {}
        for name, indices in groups.items():
            group_lookup.update({int(k): name for k in indices})
            limit = 18 if name == "visual" else 48 if name == "KC" else 32
            chosen.update(int(k) for k in indices[np.argsort(-total[indices], kind="stable")[:limit]])
        selected = np.array(sorted(chosen), dtype=np.int32)
        ix = {int(v): i for i, v in enumerate(selected)}
        plastic_lookup = {int(e): i for i, e in enumerate(edges)}
        connections = []
        for src in selected:
            for edge in range(b.ptr[src], b.ptr[src + 1]):
                target = int(b.post[edge])
                if target in ix:
                    plastic = plastic_lookup.get(edge)
                    connections.append({"id": int(edge), "source": ix[int(src)], "target": ix[target],
                                        "weight": float(b.weight[edge]) if plastic is None else float(b.baseline_plastic[plastic]),
                                        "plastic_index": plastic})
        plastic_selection = sorted({e["plastic_index"] for e in connections if e["plastic_index"] is not None})
        frames = []
        for i, event in enumerate(report["events"]):
            with np.load(output / f"step-{i+1:02}.npz") as a:
                counts = a["counts"]
                weights = a["weights"]
                initial = a["initial_weights"]
                all_delta = weights - initial
                pristine_delta = weights - b.baseline_plastic
                frames.append({"step": i+1, "event": event,
                    "input_png": base64.b64encode((output / f"input-{i+1:02}.png").read_bytes()).decode(),
                    "times_ms": a["ms"].tolist(), "counts": counts[:, selected].tolist(),
                    "voltage": a["voltage"][:, selected].round(3).tolist(),
                    "total_spikes": counts.sum(axis=1).tolist(),
                    "groups": {g: counts[:, indices].sum(axis=1).tolist() for g, indices in groups.items()},
                    "weight_delta_l2": np.linalg.norm(all_delta, axis=1).tolist(),
                    "changed_edges": np.count_nonzero(all_delta, axis=1).tolist(),
                    "weight_from_pristine_l2":np.linalg.norm(pristine_delta,axis=1).tolist(),
                    "changed_from_pristine":np.count_nonzero(pristine_delta,axis=1).tolist(),
                    "plastic_weights": weights[:, plastic_selection].tolist(),
                    "plastic_initial": initial[plastic_selection].tolist(),
                    "kc_mean_hz": a["kc"].mean(axis=1).tolist(), "dan_mean_hz": a["dan"].mean(axis=1).tolist(),
                    "memory_u_l2": np.linalg.norm(a["u"], axis=1).tolist(),
                    "memory_w_l2": np.linalg.norm(a["w"], axis=1).tolist()})
        return {"report": report, "selection": "Most active 18 visual and 48 KC cells, identified DAN/MBON/output cells, plus requested IDs. "
                "Layout is grouped by function, not anatomical coordinates. Full graph simulated; full neuron traces in NPZ.",
                "nodes": [{"id": str(b.ids[k]), "index": int(k), "type": str(self.types[k]),
                           "group": group_lookup.get(int(k), "other"), "total_spikes": int(total[k])} for k in selected],
                "edges": connections, "plastic_selection": plastic_selection, "frames": frames}


def paired_study(data, output, steps=4):
    """Predeclared mechanics comparisons; all arms reported, never select by P&L."""
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    base = Assay(steps=steps)
    arms = {
        "rise_learning": base,
        "rise_frozen": replace(base, learning=False),
        "fall_learning": replace(base, preset="fall"),
        "fall_frozen": replace(base, preset="fall", learning=False),
        "rise_reward": replace(base, reinforcement="reward"),
        "rise_aversive": replace(base, reinforcement="aversive"),
        "rise_lower_eta": replace(base, eta=.0001),
        "rise_price_only": replace(base, view="price_only"),
    }
    atomic_json(root / "protocol.json", {"arms": {k: asdict(v) for k, v in arms.items()},
                "purpose": "Input/learning/decoder sensitivity, not a trading backtest",
                "comparison": "Identical pristine native state per arm; fixed synthetic inputs; report every arm."})
    lab = TraceLab(data)
    results = {name: lab.run(cfg, root / name) for name, cfg in arms.items()}
    def compare(a, c):
        ea, ec = results[a]["events"], results[c]["events"]
        return {"arms": [a, c], "different_actions": sum(x["side"] != y["side"] for x, y in zip(ea, ec)),
                "different_spike_hashes": sum(x["spike_sha256"] != y["spike_sha256"] for x, y in zip(ea, ec)),
                "decoder_difference_delta_hz": [x["difference_hz"] - y["difference_hz"] for x, y in zip(ea, ec)]}
    summary = {"results": results, "comparisons": [compare(*pair) for pair in [
        ("rise_learning", "rise_frozen"), ("fall_learning", "fall_frozen"),
        ("rise_learning", "fall_learning"), ("rise_reward", "rise_aversive"),
        ("rise_learning", "rise_lower_eta"), ("rise_learning", "rise_price_only")]]}
    atomic_json(root / "study.json", summary)
    return summary
