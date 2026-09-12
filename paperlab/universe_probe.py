"""Bounded native checkpoint-isolation smoke test; synthetic data in its own root."""
from dataclasses import replace
from pathlib import Path
import time

from .core import atomic_json
from .multi import cycle
from .universe import Pool,Store


def run(root,fly_data):
    root=Path(root)/str(time.time_ns())
    root.mkdir(parents=True)
    store=Store(root/"synthetic.db")
    end=time.time()-301
    for i in range(64):
        stamp=end-(63-i)*60
        store.add([Pool(f"synthetic:pool{k}","synthetic",f"pool{k}",f"token{k}",f"TEST{k}",
                        .01*(1+.0005*i*(k+1)),100000,10000,20,20,stamp-3600,stamp,
                        source="synthetic_native_smoke") for k in range(4)])
    store.db.close()
    started=time.monotonic()
    first=cycle(root/"paper",root/"synthetic.db",fly_data,ingest=False,now=end)
    store=Store(root/"synthetic.db")
    for raw, in store.db.execute("SELECT payload FROM pools").fetchall():
        import json
        store.add([replace(Pool(**json.loads(raw)),observed=end+300)])
    store.db.close()
    second=cycle(root/"paper",root/"synthetic.db",fly_data,ingest=False,now=end+300)
    assert all(s["detail"]["brain_ms"]==500 for s in first["policies"]["fly"]["sleeves"])
    assert all(s["detail"]["brain_ms"]==1000 for s in second["policies"]["fly"]["sleeves"])
    result={"status":"native_universe_smoke_passed","source":"synthetic_native_smoke",
            "independent_fly_contexts":4,"observations_per_context":2,"seconds":time.monotonic()-started,
            "note":"Verifies independent native state restore and online updates. No market-performance claim."}
    import resource
    result["peak_process_rss_kib"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    atomic_json(root/"result.json",result)
    return result
