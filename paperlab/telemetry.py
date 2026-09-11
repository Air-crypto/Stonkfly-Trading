"""Small JSON events for remote logs; detailed arrays remain in durable artifacts."""
from datetime import datetime, timezone
import json


def emit(event, **fields):
    print(json.dumps({"at": datetime.now(timezone.utc).isoformat(), "event": event, **fields},
                     allow_nan=False), flush=True)
