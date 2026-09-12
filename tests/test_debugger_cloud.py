from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest

from paperlab.cloud_debug import validate_request
from paperlab.debugger_cloud import CloudJobs
from paperlab.fly_trace import Assay


def test_cloud_requests_are_bounded_and_cannot_select_arbitrary_paths():
    valid={"run_id":"assay-test", "config":asdict(Assay())}
    assert validate_request(valid).steps==4
    for request in ({**valid,"run_id":"../../paper"}, {**valid,"path":"/state/paper.db"},
                    {**valid,"config":{"steps":1000}}):
        with pytest.raises(ValueError): validate_request(request)


def test_cloud_receipt_resumes_same_call_after_observation_timeout(tmp_path, monkeypatch):
    import modal
    calls=[]
    class Remote:
        object_id="fc-test"
        def get(self, timeout=0):
            calls.append(self.object_id)
            raise TimeoutError("still running")
    remote=Remote()
    monkeypatch.setattr(modal.Function,"from_name",lambda *a,**kw:SimpleNamespace(spawn=lambda **kw:remote))
    monkeypatch.setattr(modal.FunctionCall,"from_id",lambda identity:remote)
    jobs=CloudJobs(tmp_path)
    name=jobs.launch(Assay(steps=1))
    restarted=CloudJobs(tmp_path)
    assert restarted.status()["status"]=="running"
    assert restarted.status()["run"]==name
    assert calls==["fc-test","fc-test"]
    with pytest.raises(ValueError): restarted.launch(Assay())
    assert json.loads((tmp_path/"cloud-job.json").read_text())["call_id"]=="fc-test"


def test_ambiguous_submission_blocks_duplicate_compute(tmp_path, monkeypatch):
    import modal
    def fail(**kw): raise ConnectionError("response lost")
    monkeypatch.setattr(modal.Function,"from_name",lambda *a,**kw:SimpleNamespace(spawn=fail))
    jobs=CloudJobs(tmp_path)
    with pytest.raises(ConnectionError): jobs.launch(Assay())
    assert CloudJobs(tmp_path).status()["status"]=="attention"
    with pytest.raises(ValueError): CloudJobs(tmp_path).launch(Assay())


def test_completed_cloud_call_materializes_view_and_allows_next_run(tmp_path, monkeypatch):
    import modal
    jobs=CloudJobs(tmp_path)
    result={"status":"debug_completed","run_id":"assay-test", "remote_path":"/state/fly-debugger/assay-test",
            "view":{"ok":True},"report":{"events":[]}, "budget":{"estimated_compute_usd":.01}}
    (tmp_path/"cloud-job.json").write_text(json.dumps({"run":"assay-test","status":"running","call_id":"fc-test"}))
    monkeypatch.setattr(modal.FunctionCall,"from_id",lambda identity:SimpleNamespace(get=lambda **kw:result))
    assert jobs.status()["status"]=="succeeded"
    assert (tmp_path/"assay-test/view.json").exists()
