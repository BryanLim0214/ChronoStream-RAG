"""Shared experiment plumbing: load a benchmark, run one (method, seed) replay, analyze logs."""
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from csrag.store import QDRANT_HTTP

import numpy as np
import psutil
import torch

from csrag import paths
from csrag.types import GTRevision, Revision


# ---------------------------------------------------------------- provenance
def provenance(extra: Optional[dict] = None) -> dict:
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=paths.ROOT, text=True).strip()
        except Exception:
            return None
    import importlib.metadata as md
    pkgs = {}
    for p in ["torch", "transformers", "sentence-transformers", "qdrant-client", "pyarrow", "numpy", "scipy", "pandas"]:
        try:
            pkgs[p] = md.version(p)
        except Exception:
            pkgs[p] = None
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    try:
        import urllib.request
        qv = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{QDRANT_HTTP}/", timeout=5).read())["version"]
    except Exception:
        qv = None
    mv = paths.MODELS / "model_revisions.json"
    return {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
            "python": sys.version.split()[0], "platform": platform.platform(),
            "cpu": platform.processor(), "cpu_logical": psutil.cpu_count(), "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
            "gpu": gpu, "cuda": torch.version.cuda, "packages": pkgs, "qdrant_server": qv,
            "models": json.loads(mv.read_text()) if mv.exists() else None, **(extra or {})}


def file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- Benchmark A
def load_benchA() -> Tuple[List[Revision], Dict[str, List[GTRevision]], dict, list]:
    d = paths.DATA / "benchA"
    revs = []
    for line in open(d / "revisions.jsonl", encoding="utf-8"):
        r = json.loads(line)
        revs.append(Revision(r["doc_id"], r["rev_id"], r["t_event"], r["content_hash"], tuple(r["chunks"]),
                             tuple(r["chunk_hashes"]), r["risky"]))
    gt_raw = json.load(open(d / "gt.json", encoding="utf-8"))
    gt = {doc: [GTRevision(doc, x["rev_id"], x["t_event"], x["content_hash"], frozenset(x["chunk_hashes"]),
                           x["reverted"], None, x["fact"], x["is_vandal"]) for x in g["revisions"]]
          for doc, g in gt_raw.items()}
    queries = json.load(open(d / "queries.json", encoding="utf-8"))
    return revs, gt, gt_raw, queries


# ---------------------------------------------------------------- commit-log analysis
def head_intervals(head_log, end_t: float) -> Dict[str, List[Tuple[float, float, Optional[int]]]]:
    """Per document: list of (start, end, head_rev) intervals from a head-change log."""
    by_doc: Dict[str, List[Tuple[float, Optional[int]]]] = {}
    for t, doc, rid in head_log:
        by_doc.setdefault(doc, []).append((t, rid))
    out = {}
    for doc, ev in by_doc.items():
        ev.sort(key=lambda x: x[0])
        iv = []
        for (t0, r), nxt in zip(ev, ev[1:] + [(end_t, None)]):
            if nxt[0] > t0:
                iv.append((t0, nxt[0], r))
        out[doc] = iv
    return out


def exposure_and_lag(head_log, gt: Dict[str, List[GTRevision]], truth, end_t: float) -> dict:
    """Exact exposure (seconds a later-reverted revision was the served head) and freshness lag."""
    iv = head_intervals(head_log, end_t)
    exp_secs, exp_any, n_rev = [], 0, 0
    lag_event, lag_avail, never = [], [], 0
    for doc, revs in gt.items():
        ivs = iv.get(doc, [])
        first_head: Dict[int, float] = {}
        secs: Dict[int, float] = {}
        for s, e, r in ivs:
            if r is None:
                continue
            first_head.setdefault(r, s)
            secs[r] = secs.get(r, 0.0) + (e - s)
        dt = truth[doc]
        cur = [r for r in revs if not r.reverted]
        for r in revs:
            if r.reverted:
                n_rev += 1
                x = secs.get(r.rev_id, 0.0)
                exp_secs.append(x)
                exp_any += x > 0
        for j, r in enumerate(cur):
            # only revisions that should have become head (not superseded before availability)
            nxt_ev = cur[j + 1].t_event if j + 1 < len(cur) else float("inf")
            if dt.first_avail[r.rev_id] >= nxt_ev:
                continue
            if r.rev_id in first_head:
                lag_event.append(first_head[r.rev_id] - r.t_event)
                lag_avail.append(first_head[r.rev_id] - dt.first_avail[r.rev_id])
            else:
                never += 1
    f = lambda a, q: float(np.percentile(a, q)) if len(a) else float("nan")
    return {"reverted_revisions": n_rev, "reverted_ever_served": int(exp_any),
            "reverted_served_share": exp_any / n_rev if n_rev else float("nan"),
            "exposure_seconds_total": float(np.sum(exp_secs)) if exp_secs else 0.0,
            "lag_event_p50_s": f(lag_event, 50), "lag_event_p95_s": f(lag_event, 95),
            "lag_added_p50_s": f(lag_avail, 50), "lag_added_p95_s": f(lag_avail, 95),
            "lag_added_mean_s": float(np.mean(lag_avail)) if lag_avail else float("nan"),
            "heads_never_served": never, "heads_considered": len(lag_event) + never}
