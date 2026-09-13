"""Conservative compute reservation; provider billing controls remain authoritative."""
from datetime import datetime, timezone
from pathlib import Path
import time

from .core import atomic_json


def reserve(path, full_fly, now=None, seconds=600, limit_override=None, startup_seconds=30, cpu=None, memory_gib=None,
            authorized_monthly_limit=None):
    import json
    now = time.time() if now is None else now
    path = Path(path)
    month = datetime.fromtimestamp(now, timezone.utc).strftime("%Y-%m")
    state = json.loads(path.read_text()) if path.exists() else {"first_month": "2026-09", "months": {}}
    limit = 100 if month == state["first_month"] else 40
    # Explicit caller opt-in for the user's revised $100/month authorization.
    # Existing experiments retain their earlier first-month/steady-state limits.
    if authorized_monthly_limit is not None:
        import math
        if not math.isfinite(authorized_monthly_limit) or not 0 < authorized_monthly_limit <= 100:
            raise ValueError("Monthly authorization must stay within $100")
        limit = authorized_monthly_limit
    if limit_override is not None:
        import math
        if not math.isfinite(limit_override) or not 0 < limit_override <= limit:
            raise ValueError("Override must stay within the authorized monthly budget")
        limit = limit_override
    # Current public CPU/RAM rates with 2x safety factor and 30s startup allowance.
    cpu = 2 if cpu is None else cpu
    memory_gib = (16 if full_fly else 4) if memory_gib is None else memory_gib
    import math
    if not all(math.isfinite(x) for x in (cpu,memory_gib)) or not 0 < cpu <= 2 or not 0 < memory_gib <= 16:
        raise ValueError("Invalid budget resource allocation")
    rate = 2 * (.0000131 * cpu + .00000222 * memory_gib)
    if not 0 <= startup_seconds <= 60 or not 0 < seconds <= 3600:
        raise ValueError("Invalid reservation duration")
    charge = (seconds + startup_seconds) * rate
    spent = state["months"].get(month, 0)
    if spent + charge > .75 * limit:
        return None
    state["months"][month] = spent + charge
    atomic_json(path, state)
    return {"month": month, "reserve": charge, "rate": rate, "limit": limit, "started": now, "startup_seconds": startup_seconds}


def settle(path, reservation, elapsed):
    import json
    path = Path(path)
    state = json.loads(path.read_text())
    charge = (max(0, elapsed) + reservation.get("startup_seconds", 30)) * reservation["rate"]
    state["months"][reservation["month"]] += charge - reservation["reserve"]
    atomic_json(path, state)
    return {"estimated_compute_usd": charge, "monthly_reserved_usd": state["months"][reservation["month"]], "monthly_limit_usd": reservation["limit"], "provider_bill": False}
