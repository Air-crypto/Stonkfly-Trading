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
    {"learning":"false"}, {"preset":"future"}, {"reinforcement":"trade"}, {"view":"hidden_policy"}])
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
