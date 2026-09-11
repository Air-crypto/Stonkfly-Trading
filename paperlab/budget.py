"""Conservative compute reservation; provider billing controls remain authoritative."""
from datetime import datetime, timezone
from pathlib import Path
import time

from .core import atomic_json


def reserve(path, full_fly, now=None, seconds=600):
    import json
    now = time.time() if now is None else now
    path = Path(path)
    month = datetime.fromtimestamp(now, timezone.utc).strftime("%Y-%m")
    state = json.loads(path.read_text()) if path.exists() else {"first_month": "2026-09", "months": {}}
    limit = 100 if month == state["first_month"] else 40
    # Current public CPU/RAM rates with 2x safety factor and 30s startup allowance.
    rate = 2 * (.0000131 * 2 + .00000222 * (16 if full_fly else 4))
    charge = (seconds + 30) * rate
    spent = state["months"].get(month, 0)
    if spent + charge > .75 * limit:
        return None
    state["months"][month] = spent + charge
    atomic_json(path, state)
    return {"month": month, "reserve": charge, "rate": rate, "limit": limit, "started": now}


def settle(path, reservation, elapsed):
    import json
    path = Path(path)
    state = json.loads(path.read_text())
    charge = (max(0, elapsed) + 30) * reservation["rate"]
    state["months"][reservation["month"]] += charge - reservation["reserve"]
    atomic_json(path, state)
    return {"estimated_compute_usd": charge, "monthly_reserved_usd": state["months"][reservation["month"]], "monthly_limit_usd": reservation["limit"], "provider_bill": False}
