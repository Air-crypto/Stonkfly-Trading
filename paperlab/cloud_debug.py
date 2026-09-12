"""Bounded cloud assays run under the existing worker lease and compute budget."""
import json
from pathlib import Path
import re

from .fly_trace import Assay, TraceLab


def validate_request(request):
    if not isinstance(request, dict) or set(request) not in ({"run_id", "config"}, {"run_id", "market_plan"}):
        raise ValueError("Expected run_id and assay config")
    if not isinstance(request["run_id"], str) or not re.fullmatch(r"assay-[a-z0-9-]{1,80}", request["run_id"]):
        raise ValueError("Invalid diagnostic run ID")
    if "market_plan" in request:
        from .fly_market_study import validate, signature
        envelope=request["market_plan"]
        if not isinstance(envelope,dict) or set(envelope)!={"plan","sha256"}:
            raise ValueError("Expected a sealed market plan")
        validate(envelope["plan"])
        if envelope["sha256"]!=signature(envelope["plan"]):
            raise ValueError("Sealed market plan hash mismatch")
        return envelope
    if not isinstance(request["config"], dict):
        raise ValueError("Expected an assay configuration")
    return Assay(**request["config"])


def run(request, root, data):
    config = validate_request(request)
    output = Path(root) / request["run_id"]
    if "market_plan" in request:
        from .fly_market_study import run as market_run
        report=market_run(config,data,output)
        return {"status":"market_study_completed","run_id":request["run_id"],
                "remote_path":str(output),"report":report}
    # Refuse reuse rather than resetting or overwriting a previous experiment.
    lab = TraceLab(data)
    report = lab.run(config, output)
    return {"status": "debug_completed", "run_id": request["run_id"],
            "remote_path": str(output), "report": report,
            "view": json.loads((output / "view.json").read_text())}
