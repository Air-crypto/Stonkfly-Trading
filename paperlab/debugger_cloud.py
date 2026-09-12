"""Authenticated Modal bridge. Credentials stay in the local SDK, never the UI."""
from dataclasses import asdict
import json
from pathlib import Path
import threading
import uuid

from .core import atomic_json


class CloudJobs:
    def __init__(self, root, data=None):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.receipt = self.root / "cloud-job.json"
        self.lock = threading.Lock()
        self.data = None
        self.can_run = True
        self.backend = "modal"

    def saved(self):
        return json.loads(self.receipt.read_text()) if self.receipt.exists() else None

    def launch(self, config):
        import modal
        with self.lock:
            old = self.saved()
            if old and old["status"] not in ("succeeded", "not_started"):
                raise ValueError("Previous cloud call remains unresolved. Inspect its saved call ID; do not resubmit it")
            name = "assay-" + uuid.uuid4().hex
            receipt = {"run": name, "status": "submitting", "config": asdict(config)}
            atomic_json(self.receipt, receipt)
            # A lost submission response stays unresolved; never retry paid work blindly.
            call = modal.Function.from_name("fly-paper-lab", "worker", environment_name="main").spawn(
                debug={"run_id": name, "config": asdict(config)})
            receipt.update(status="running", call_id=call.object_id)
            atomic_json(self.receipt, receipt)
            (self.root / (name + ".log")).write_text(f"Modal call {call.object_id}\nCloud execution continues if the laptop sleeps.\n")
            return name

    def status(self):
        import modal
        with self.lock:
            receipt = self.saved()
            if not receipt:
                return {"status": "idle", "run": None}
            if receipt["status"] in ("succeeded", "not_started"):
                return receipt
            if "call_id" not in receipt:
                return {**receipt, "status": "attention", "error": "Submission outcome unknown; inspect Modal before another run"}
            try:
                result = modal.FunctionCall.from_id(receipt["call_id"]).get(timeout=0)
            except TimeoutError:
                return {**receipt, "status": "running"}
            except Exception as exc:
                # Transport failures, expired result retention and remote exceptions are
                # not treated as permission to duplicate an expensive cloud invocation.
                return {**receipt, "status": "attention", "error": f"{type(exc).__name__}: {exc}"[:500]}
            if result.get("status") in ("writer_busy", "budget_stopped"):
                receipt.update(status="not_started", result=result)
                atomic_json(self.receipt, receipt)
                return receipt
            if result.get("status") != "debug_completed" or result.get("run_id") != receipt["run"]:
                return {**receipt, "status": "attention", "error": "Unexpected cloud result"}
            output = self.root / receipt["run"]
            output.mkdir(exist_ok=True)
            atomic_json(output / "view.json", result["view"])
            atomic_json(output / "report.json", result["report"])
            atomic_json(output / "remote.json", {"remote_path": result["remote_path"], "call_id": receipt["call_id"]})
            receipt.update(status="succeeded", budget=result["budget"])
            atomic_json(self.receipt, receipt)
            with (self.root / (receipt["run"] + ".log")).open("a") as log:
                log.write(f"Completed. Estimated compute: {result['budget']['estimated_compute_usd']:.6f} USD\n")
            return receipt

    def ensure_recordings(self, output):
        """Download full traces only when the user asks for an arbitrary neuron."""
        metadata = output / "remote.json"
        if not metadata.exists():
            return
        import modal
        report = json.loads((output / "report.json").read_text())
        remote = json.loads(metadata.read_text())["remote_path"]
        volume = modal.Volume.from_name("fly-paper-lab-state", environment_name="main")
        for i in range(len(report["events"])):
            name = f"step-{i+1:02}.npz"
            target = output / name
            if target.exists():
                continue
            temporary = target.with_suffix(".partial")
            with temporary.open("wb") as stream:
                for block in volume.read_file(remote.removeprefix("/state") + "/" + name):
                    stream.write(block)
            temporary.replace(target)
