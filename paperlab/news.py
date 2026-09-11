"""Timestamped, append-only news revisions. Text is data, never an instruction."""
import calendar
import hashlib
import json
import re
import sqlite3
import time
from urllib.parse import urlsplit

from defusedxml import ElementTree
import feedparser
import numpy as np
import requests


DEFAULT_FEEDS = ["https://www.coindesk.com/arc/outboundfeeds/rss/", "https://www.federalreserve.gov/feeds/press_all.xml"]
FINBERT_REVISION = "4556d13015211d73dccd3fdd39d39232506f3e43"
POS = {"approval", "approved", "growth", "gain", "gains", "surge", "record", "rise", "rally"}
NEG = {"hack", "hacked", "fraud", "loss", "losses", "ban", "crash", "fall", "lawsuit", "liquidation"}


class News:
    def __init__(self, path=":memory:", enabled=True):
        self.db = sqlite3.connect(path)
        self.enabled = enabled
        self.db.execute("CREATE TABLE IF NOT EXISTS news (id TEXT PRIMARY KEY, url TEXT, title TEXT, source TEXT, published REAL, seen REAL, encoded REAL, encoder TEXT, scores TEXT)")

    def add(self, url, title, source, published, seen, scores=None, encoder="lexical-v1", encoded=None):
        title = str(title)[:2000]
        if not np.isfinite([published, seen, encoded or seen]).all() or min(published, seen) <= 0:
            raise ValueError("Invalid news timestamps")
        identity = hashlib.sha256((url + "\0" + title + "\0" + encoder).encode()).hexdigest()
        if scores is None:
            words = set(re.findall(r"[a-z]+", title.lower()))
            pos, neg = len(words & POS), len(words & NEG)
            scores = [pos / (pos + neg + 1), neg / (pos + neg + 1), 1 / (pos + neg + 1)]
        if len(scores) != 3 or not np.isfinite(scores).all() or min(scores) < 0 or abs(sum(scores) - 1) > .01:
            raise ValueError("News scores must be positive/negative/neutral probabilities")
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO news VALUES (?,?,?,?,?,?,?,?,?)", (identity, url, title, source, published, seen, encoded or seen, encoder, json.dumps(scores)))
        return identity

    def rows(self, now):
        # Availability uses all three clocks; original publication time alone is unsafe.
        return self.db.execute("SELECT url,title,scores,published,seen,encoded,encoder FROM news WHERE published<=? AND seen<=? AND encoded<=? AND published>=? ORDER BY max(published,seen,encoded) DESC LIMIT 200", (now, now, now, now - 86400)).fetchall()

    def features(self, now):
        f = np.zeros(50, dtype=np.float32)
        if not self.enabled:
            return f
        rows = self.rows(now)
        # Each URL contributes only its newest available revision. Encoder changes never rewrite history.
        seen_urls = set()
        weights = 0
        for url, title, scores, published, seen, encoded, _ in rows:
            if url in seen_urls:
                continue
            seen_urls.add(url)
            age = max(0, now - max(published, seen, encoded))
            w = float(np.exp(-age / 21600))
            f[:3] += w * np.array(json.loads(scores))
            for token in set(re.findall(r"[a-z]{2,}", title.lower())):
                h = hashlib.blake2b(token.encode(), digest_size=4).digest()
                f[3 + int.from_bytes(h[:2], "little") % 45] += w * (1 if h[2] % 2 else -1)
            weights += w
        if weights:
            f[:48] /= weights
            f[48] = np.log1p(len(seen_urls))
            f[49] = min(1, (now - max(rows[0][3:6])) / 86400)
        return f

    def ingest(self, feeds=DEFAULT_FEEDS, encoder=None):
        report = []
        for url in feeds:
            if urlsplit(url).scheme != "https":
                raise ValueError("RSS feeds must use HTTPS")
            try:
                response = requests.get(url, timeout=20, headers={"User-Agent": "FlyPaperLab/0.1 (research; hourly RSS)"}, stream=True)
                response.raise_for_status()
                raw = bytearray()
                for block in response.iter_content(65536):
                    raw.extend(block)
                    if len(raw) > 4_000_000:
                        raise ValueError("Oversized RSS feed")
                seen = time.time()
                ElementTree.fromstring(raw)  # Reject DTD/entity expansion before feed parsing.
                parsed = feedparser.parse(bytes(raw))
                count = 0
                for entry in parsed.entries[:100]:
                    title = str(entry.get("title", ""))[:2000]
                    link = str(entry.get("link", ""))[:2000]
                    if not title or not link:
                        continue
                    pub = entry.get("published_parsed") or entry.get("updated_parsed")
                    published = float(calendar.timegm(pub)) if pub else seen
                    encoder_name = encoder.name if encoder else "lexical-v1"
                    identity = hashlib.sha256((link + "\0" + title + "\0" + encoder_name).encode()).hexdigest()
                    if self.db.execute("SELECT 1 FROM news WHERE id=?", (identity,)).fetchone():
                        continue
                    scores = encoder(title) if encoder else None
                    self.add(link, title, url, published, seen, scores, encoder_name, time.time())
                    count += 1
                report.append({"feed": url, "entries": count, "status": "ok"})
            except Exception as exc:
                # A failed feed remains visible in the report; no fabricated neutral article.
                report.append({"feed": url, "status": "error", "error": str(exc)[:300]})
        return report


class FinBERT:
    """Optional independent BERT encoder; pinned revision is required for reproducibility."""
    def __init__(self, revision):
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Pin FinBERT to a full Hugging Face commit hash")
        from transformers import pipeline
        import torch
        torch.set_num_threads(2)
        self.name = "ProsusAI/finbert@" + revision
        self.pipe = pipeline("text-classification", model="ProsusAI/finbert", revision=revision, device=-1, top_k=None)

    def __call__(self, title):
        result = self.pipe(title, truncation=True, max_length=128)
        if result and isinstance(result[0], list):
            result = result[0]
        scores = {r["label"].lower(): r["score"] for r in result}
        return [scores[k] for k in ("positive", "negative", "neutral")]
