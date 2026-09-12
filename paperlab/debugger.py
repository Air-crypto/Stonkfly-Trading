"""Local UI for isolated full-network assays: python -m paperlab.debugger --help."""
import argparse
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from urllib.parse import parse_qs, urlsplit

import numpy as np

from .core import atomic_json
from .fly_trace import Assay, TraceLab, paired_study

WEB = Path(__file__).with_name("debugger_web")


def safe_run(root, name):
    if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", name):
        raise ValueError("Invalid run name")
    return root / name


def neuron_trace(root, identity):
    if not identity.isdecimal():
        raise ValueError("Use an exact neuron ID")
    frames = []
    for path in sorted(root.glob("step-*.npz")):
        with np.load(path, allow_pickle=False) as a:
            matches = np.flatnonzero(a["neuron_ids"] == int(identity))
            if len(matches) != 1:
                raise ValueError("Neuron not in the retained graph")
            index = int(matches[0])
            edge_ix = np.flatnonzero((a["plastic_pre"] == index) | (a["plastic_post"] == index))
            frames.append({"step": len(frames)+1, "times_ms": a["ms"].tolist(),
                           "counts": a["counts"][:, index].tolist(), "voltage": a["voltage"][:, index].tolist(),
                           "plastic_edges": [{"id": int(a["plastic_edges"][e]),
                              "source": str(a["neuron_ids"][a["plastic_pre"][e]]),
                              "target": str(a["neuron_ids"][a["plastic_post"][e]]),
                              "initial": float(a["initial_weights"][e]),
                              "final": float(a["weights"][-1, e])} for e in edge_ix]})
    if not frames:
        raise ValueError("Full neuron recordings unavailable in this run")
    return {"id": identity, "frames": frames}


class Jobs:
    def __init__(self, root, data):
        self.root, self.data = root.resolve(), data.resolve() if data else None
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.process = None
        self.name = None
        self.can_run = self.data is not None
        self.backend = "local" if self.data else "recordings"

    def launch(self, config):
        if self.data is None:
            raise ValueError("Read-only mode. Start with --fly-data to run the native model")
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                raise ValueError("An assay is already running; wait for its result")
            self.name = "assay-" + str(time.time_ns())
            spec = self.root / (self.name + "-config.json")
            atomic_json(spec, asdict(config))
            with (self.root / (self.name + ".log")).open("wb") as log:
                self.process = subprocess.Popen([sys.executable, "-m", "paperlab.debugger", "capture",
                    "--fly-data", str(self.data), "--out", str(self.root / self.name), "--config", str(spec)],
                    stdout=log, stderr=subprocess.STDOUT)
            return self.name

    def status(self):
        with self.lock:
            code = self.process.poll() if self.process else None
            return {"run": self.name, "status": "idle" if self.process is None else
                    "running" if code is None else "succeeded" if code == 0 else "failed",
                    "exit_code": code}


def serve(root, data, port, backend="local"):
    if backend == "modal":
        from .debugger_cloud import CloudJobs
        jobs = CloudJobs(root)
    else:
        jobs = Jobs(root, data)

    class Handler(BaseHTTPRequestHandler):
        def send(self, status, value, content_type="application/json"):
            payload = json.dumps(value, allow_nan=False).encode() if content_type == "application/json" else value
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; object-src 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(payload)

        def allowed(self, write=False):
            hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
            host = self.headers.get("Host")
            if host not in hosts:
                return False
            origin = self.headers.get("Origin")
            return origin == f"http://{host}" if write else origin in (None, f"http://{host}")

        def do_GET(self):
            if not self.allowed():
                return self.send(403, {"error": "Local same-origin requests only"})
            url = urlsplit(self.path)
            q = parse_qs(url.query)
            try:
                if url.path == "/api/runs":
                    runs = sorted(p.parent.name for p in jobs.root.glob("*/view.json"))
                    return self.send(200, {"runs": runs, "can_run": jobs.can_run, "backend": jobs.backend})
                if url.path == "/api/status":
                    return self.send(200, jobs.status())
                if url.path == "/api/view":
                    path = safe_run(jobs.root, q.get("run", [""])[0]) / "view.json"
                    return self.send(200, json.loads(path.read_text()))
                if url.path == "/api/neuron":
                    path = safe_run(jobs.root, q.get("run", [""])[0])
                    if jobs.backend == "modal":
                        jobs.ensure_recordings(path)
                    return self.send(200, neuron_trace(path, q.get("id", [""])[0]))
                if url.path == "/api/log":
                    name = q.get("run", [""])[0]
                    path = safe_run(jobs.root, name).with_suffix(".log")
                    return self.send(200, {"log": path.read_text()[-8000:]})
                names = {"/": ("index.html", "text/html"), "/app.js": ("app.js", "text/javascript"),
                         "/style.css": ("style.css", "text/css")}
                if url.path not in names:
                    return self.send(404, {"error": "Not found"})
                name, mime = names[url.path]
                self.send(200, (WEB / name).read_bytes(), mime)
            except (ValueError, FileNotFoundError, OverflowError) as exc:
                self.send(400, {"error": str(exc)})
            except Exception as exc:
                self.send(502, {"error": f"{type(exc).__name__}: {exc}"[:500]})

        def do_POST(self):
            if not self.allowed(write=True):
                return self.send(403, {"error": "Local same-origin requests only"})
            if self.path != "/api/run":
                return self.send(404, {"error": "Not found"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 8192 or self.headers.get("Content-Type") != "application/json":
                    raise ValueError("Expected a small JSON assay configuration")
                config = Assay(**json.loads(self.rfile.read(length)))
                self.send(202, {"run": jobs.launch(config)})
            except (ValueError, TypeError) as exc:
                self.send(400, {"error": str(exc)})
            except Exception as exc:
                self.send(502, {"error": f"{type(exc).__name__}: {exc}"[:500]})

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Fly debugger: http://127.0.0.1:{port} (isolated synthetic assays)", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("capture", "study", "serve"):
        sub = commands.add_parser(name)
        sub.add_argument("--fly-data", type=Path, required=name != "serve")
        sub.add_argument("--out", type=Path, required=True, help="Isolated output directory, never a paper account")
        if name == "serve":
            sub.add_argument("--port", type=int, default=8765)
            sub.add_argument("--backend", choices=["local", "modal"], default="local")
        else:
            sub.add_argument("--steps", type=int, default=4)
            if name == "capture":
                sub.add_argument("--config", type=Path)
                sub.add_argument("--preset", choices=["rise", "fall", "flat", "reversal", "shock"], default="rise")
                sub.add_argument("--frozen", action="store_true")
    args = parser.parse_args()
    if args.command == "serve":
        serve(args.out, args.fly_data, args.port, args.backend)
    elif args.command == "study":
        paired_study(args.fly_data, args.out, args.steps)
    else:
        config = Assay(**json.loads(args.config.read_text())) if args.config else Assay(
            preset=args.preset, steps=args.steps, learning=not args.frozen)
        TraceLab(args.fly_data).run(config, args.out)


if __name__ == "__main__":
    main()
