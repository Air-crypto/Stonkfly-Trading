from dataclasses import replace
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from paperlab.debugger import neuron_trace, safe_run
from paperlab.fly_trace import Assay, Recorder, TraceLab, input_frames


@pytest.mark.parametrize("change", [{"steps":0}, {"steps":9}, {"steps":True}, {"eta":float("nan")},
    {"eta":-.01}, {"eta":.01}, {"current":21}, {"current":1}, {"neurons":["../x"]},
    {"probe_steps":True}, {"probe_steps":5}, {"steps":8,"probe_steps":1}, {"probe_preset":"unknown"}, {"learning":"false"}, {"preset":"future"}, {"reinforcement":"trade"}, {"view":"hidden_policy"}])
def test_bounded_assay_configuration(change):
    with pytest.raises((ValueError, TypeError)):
        Assay(**change)


def test_recorder_preserves_return_and_stimulus_and_restores_on_error():
    seen=[]
    def original(rgb, duration, **kwargs):
        seen.append(kwargs)
        b.sim_ms += duration
        return np.array([3,4,5]), .25
    b=SimpleNamespace(rgb_step=original, sim_ms=0, weight=np.array([1.,2.]),
        circuit={"edges":np.array([1])}, v=np.zeros(3), rate_kc=np.zeros(1),
        rate_dan=np.zeros(1), memory_u=np.zeros(1), memory_w=np.zeros(1))
    pulse=(np.array([2]), 20)
    with pytest.raises(RuntimeError):
        with Recorder(b) as recorder:
            counts,wall=b.rgb_step(None,10,learning=True,stimulation=pulse)
            assert seen[0]["stimulation"] is pulse
            assert wall==.25
            assert np.array_equal(counts,[3,4,5])
            assert np.array_equal(recorder.arrays()["counts"],[[3,4,5]])
            raise RuntimeError("simulated downstream failure")
    assert b.rgb_step is original
    with Recorder(b,[1],-10):
        b.rgb_step(None,10,learning=False,stimulation=pulse)
    assert seen[-1]["stimulation"][0] is pulse
    assert seen[-1]["stimulation"][1][1]==-10


def test_paths_and_full_neuron_lookup(tmp_path):
    with pytest.raises(ValueError): safe_run(tmp_path,"../secret")
    np.savez_compressed(tmp_path/"step-01.npz", neuron_ids=[123,456], ms=[10,20],
        counts=[[2,3],[4,5]], voltage=[[-50,-51],[-49,-48]], plastic_pre=[0],
        plastic_post=[1], plastic_edges=[42], initial_weights=[1.], weights=[[1.1],[1.2]])
    result=neuron_trace(tmp_path,"456")
    assert result["frames"][0]["counts"]==[3,5]
    assert result["frames"][0]["plastic_edges"][0]["source"]=="123"
    with pytest.raises(ValueError): neuron_trace(tmp_path,"789")


def test_recorder_does_not_leave_a_bound_method_cycle():
    import weakref
    class Brain:
        def __init__(self):
            self.weight=np.array([1.]);self.circuit={"edges":np.array([0])}
        def rgb_step(self, *args, **kwargs): pass
    brain=Brain();ref=weakref.ref(brain)
    with Recorder(brain): pass
    assert "rgb_step" not in vars(brain)
    del brain
    assert ref() is None  # Full graph must release without a later garbage-collection cycle.


@pytest.mark.skipif(not os.environ.get("FLY_TRACE_DATA"), reason="requires prepared full retained graph")
def test_full_native_trace_is_observational_and_reset_is_reproducible(tmp_path):
    lab=TraceLab(Path(os.environ["FLY_TRACE_DATA"]))
    config=Assay(steps=1)
    rgb=input_frames(config)[0]
    reference=lab.fly.controller.observe(rgb,"none")
    report=lab.run(config,tmp_path/"captured")
    actual=report["events"][0]
    assert actual["spike_sha256"]==reference["spike_sha256"]
    assert actual["side"]==reference["side"]
    assert actual["memory"]==reference["memory"]
    repeated=lab.run(config,tmp_path/"repeated")["events"][0]
    assert repeated["spike_sha256"]==actual["spike_sha256"]
    frozen=lab.run(replace(config,learning=False),tmp_path/"frozen")["events"][0]
    assert frozen["diagnostics"]["changed_edges"]==0
    assert frozen["diagnostics"]["weight_delta_l2"]==0


@pytest.mark.skipif(not os.environ.get("FLY_TRACE_DATA"), reason="requires prepared full retained graph")
def test_reinforcement_gate_freezes_neutral_updates_but_keeps_spikes(tmp_path):
    lab=TraceLab(Path(os.environ["FLY_TRACE_DATA"]))
    cfg=Assay(steps=2,reinforcement_only=True)
    gated=lab.run(cfg,tmp_path/"gated")
    frozen=lab.run(replace(cfg,learning=False),tmp_path/"frozen")
    assert [e["spike_sha256"] for e in gated["events"]]==[e["spike_sha256"] for e in frozen["events"]]
    assert all(e["total_spikes"]>0 and e["diagnostics"]["weight_delta_l2"]==0 for e in gated["events"])
    assert not any(e["diagnostics"]["plasticity_enabled"] for e in gated["events"])
    reward=lab.run(replace(cfg,steps=1,reinforcement="reward"),tmp_path/"reward")
    assert reward["events"][0]["diagnostics"]["plasticity_enabled"]
    assert reward["events"][0]["diagnostics"]["changed_edges"]>0


@pytest.mark.skipif(not os.environ.get("FLY_TRACE_DATA"), reason="requires prepared full retained graph")
def test_retention_probe_clears_transients_preserves_memory_and_freezes_updates(tmp_path):
    lab=TraceLab(Path(os.environ["FLY_TRACE_DATA"]))
    cfg=Assay(steps=1,probe_steps=1,probe_preset="fall",learning=False)
    baseline=lab.run(cfg,tmp_path/"baseline")
    stimulated=lab.run(replace(cfg,reinforcement="reward",news="positive"),tmp_path/"stimulated")
    reference=lab.run(Assay(preset="fall",steps=1,learning=False),tmp_path/"reference")
    bp=baseline["events"][-1];sp=stimulated["events"][-1]
    assert bp["spike_sha256"]==sp["spike_sha256"]==reference["events"][0]["spike_sha256"]
    assert bp["input_sha256"]==sp["input_sha256"]
    assert sp["stimulus"]=="none" and sp["input_news"]=="none" and sp["phase"]=="probe"
    trained=lab.run(replace(cfg,learning=True,reinforcement="reward"),tmp_path/"trained")
    assert trained["probe_boundary"]["weight_delta_from_pristine_l2"]>0
    assert trained["events"][-1]["diagnostics"]["weight_delta_l2"]==0
    assert not trained["events"][-1]["diagnostics"]["plasticity_enabled"]
    with np.load(tmp_path/"trained/step-01.npz") as a, np.load(tmp_path/"trained/step-02.npz") as b:
        assert np.array_equal(a["weights"][-1],b["initial_weights"])
        assert np.array_equal(a["u"][-1],b["u"][-1])
        assert np.array_equal(a["w"][-1],b["w"][-1])
        assert b["ms"][0]==10
