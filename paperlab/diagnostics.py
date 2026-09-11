"""One bounded historical assay, distinct from the untouched forward paper accounts."""
from pathlib import Path
import json

from .compact import evaluate, load_policy, train
from .core import Costs, atomic_json, digest, load_ticks
from .data import historical
from .fly import replay
from .news import News
from .telemetry import emit


def run(root, fly_data, full_fly=True, product="BTC-USD"):
    root = Path(root)
    output = root / "diagnostics"
    result_path = output / "completed.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    path = root / "bootstrap.jsonl"
    if not path.exists():
        historical(path, product=product, granularity=300)
    ticks = load_ticks(path)
    news = News(root / "news.db")
    try:
        if not (output / "compact/metrics.json").exists():
            train(ticks, news, output / "compact", steps=8192, dataset_sha=digest(path))
        metrics = json.loads((output / "compact/metrics.json").read_text())
        active = root / "active-policy.pt"
        if not active.exists():
            tmp = active.with_suffix(".partial")
            tmp.write_bytes((output / "compact/policy.pt").read_bytes())
            tmp.replace(active)
        policy, _ = load_policy(output / "compact/policy.pt")
        start = int(len(ticks) * .85)
        compact = evaluate(policy, ticks, news, Costs(), start, start + 32)
        atomic_json(output / "compact/replay.json", compact)
        fly = replay(ticks, news, fly_data, output / "fly", steps=32, start=start, diagnostics=True) if full_fly else None
        result = {"status": "diagnostics_completed", "source": ticks[0].source, "interval_seconds": 300,
                  "dataset_sha256": digest(path), "observations": len(ticks), "paired_replay_steps": 32,
                  "replay_start": ticks[start].ts, "replay_end": ticks[start+32].ts,
                  "compact_test": metrics["splits"]["test"], "compact_replay": {k:v for k,v in compact.items() if k != "ledger"},
                  "fly_replay": fly, "note": "Historical replay with modeled fills. Does not change either forward account. No parameter selection by test performance."}
        atomic_json(result_path, result)
        emit("diagnostics_completed", **result)
        return result
    finally:
        news.db.close()
