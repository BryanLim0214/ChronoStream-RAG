"""B4: readers running concurrently with the writer. Does any read see a half-applied update?

python experiments/run_concurrency.py --exp B4_concurrency --tier_gb 1 --readers 4
For each method, the writer replays the tier as fast as it can while reader threads repeatedly
read the current page of one of the 8 documents written most recently (documents under update; a
document joins this set only after its first head has been applied). The history methods run
without holdback here (holdback does not change what a read can see of an applied update). A read is
  torn     if its chunk set equals no revision of that document that ever existed,
  missing  if it is empty although the document had a head before the read started.
Snapshot methods pin the published commit number at the start of each read.
"""
import argparse
import json
import random
import sys
import threading
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qdrant_client import models

from csrag import paths
from csrag.embed import Embedder
from csrag.methods import MAXTX, TINF, MethodConfig, make_method
from csrag.store import QDRANT_GRPC, QDRANT_HTTP, Store
from csrag.stream import DisorderConfig, simulate_deliveries
from experiments.run_benchB import load_tier, restart_qdrant


def run(exp, tier_gb, n_readers, seed=0, methods=None):
    out = paths.RESULTS / exp
    out.mkdir(parents=True, exist_ok=True)
    revs, gt, raw = load_tier(tier_gb)
    sets = {doc: {r.chunk_set for r in rs} for doc, rs in gt.items()}
    dis = DisorderConfig(p_ooo=0.25, delay_median_s=60, delay_sigma=0.65, dup_rate=0.15, embed_s_per_chunk=0.02,
                         straggler_p=0.28, straggler_scale_s=18, chunk_jitter_p=0.30, chunk_jitter_scale_s=15)
    deliveries = simulate_deliveries(revs, dis, seed)
    cfgs = {
        "incremental": MethodConfig("incremental", "incremental", archive=False),
        "doc_guard": MethodConfig("doc_guard", "doc_guard"),
        "proposed_inplace_commit": MethodConfig("proposed_inplace_commit", "history", holdback_s=0.0, hot_k=2, commit="inplace"),
        "proposed": MethodConfig("proposed", "history", holdback_s=0.0, hot_k=2, commit="mvcc"),
    }
    results = []
    if methods:                # rerun only some policies (their records are appended to the results file)
        cfgs = {k: v for k, v in cfgs.items() if k in methods}
    for name, cfg in cfgs.items():
        restart_qdrant()
        emb = Embedder(reuse=True)
        store = Store(f"{exp}_{name}".lower()[:60])
        m = make_method(cfg, emb, store, paths.ARCHIVES / exp / name)
        active = {}
        lock = threading.Lock()
        if cfg.kind == "history":
            m.reader_min_snapshot = lambda: min(list(active.values()) or [m.S])
            m.concurrent_readers = True
        recent = deque(maxlen=8)   # documents written most recently: reads target documents under update
        headed_set = set()
        stop = threading.Event()
        counts = {"reads": 0, "torn": 0, "missing": 0}

        def reader(i):
            rng = random.Random(1000 + i)
            rd = Store.__new__(Store)               # second client set on the same collection
            rd.__dict__.update(store.__dict__)
            rd.client = type(store.client)(host="127.0.0.1", port=QDRANT_HTTP, grpc_port=QDRANT_GRPC, prefer_grpc=True, timeout=600)
            rd.dirty = False
            while not stop.is_set():
                try:
                    doc = recent[rng.randrange(len(recent))] if recent else None
                except IndexError:          # the deque changed between len() and indexing
                    doc = None
                if doc is None:
                    time.sleep(0.001)
                    continue
                tid = threading.get_ident()
                if cfg.kind == "history" and cfg.commit == "mvcc":
                    with m.lock:
                        S = m.S
                        active[tid] = S
                    flt = Store.flt(doc, [models.FieldCondition(key="valid_to", range=models.Range(gte=TINF)),
                                          models.FieldCondition(key="tx_from", range=models.Range(lte=S)),
                                          models.FieldCondition(key="tx_to", range=models.Range(gt=S))])
                elif cfg.kind == "history":
                    flt = Store.flt(doc, [models.FieldCondition(key="valid_to", range=models.Range(gte=TINF))])
                elif cfg.kind == "doc_guard":
                    flt = Store.flt(doc, [models.FieldCondition(key="tx_to", range=models.Range(gte=MAXTX))])
                else:
                    flt = Store.flt(doc)
                try:
                    pts, _ = rd.client.scroll(store.name, scroll_filter=flt, limit=4096, with_payload=["chunk_hash"],
                                              with_vectors=False)
                except Exception:
                    active.pop(tid, None)
                    continue
                active.pop(tid, None)
                V = frozenset(p.payload["chunk_hash"] for p in pts)
                with lock:
                    counts["reads"] += 1
                    if not V:
                        counts["missing"] += 1
                    elif V not in sets[doc]:
                        counts["torn"] += 1

        threads = [threading.Thread(target=reader, args=(i,), daemon=True) for i in range(n_readers)]
        for t in threads:
            t.start()
        t0 = time.perf_counter()
        store.wait = False
        for k, d in enumerate(deliveries):
            m.on_delivery(d, d.t_avail, lambda *a: None)
            doc = d.rev.doc_id
            if m.head_of(doc) is not None:
                if doc not in headed_set:      # first head: make sure it is applied before anyone reads it
                    store.barrier()
                    headed_set.add(doc)
                recent.append(doc)
        stop.set()
        for t in threads:
            t.join()
        wall = time.perf_counter() - t0
        rec = {"method": name, "tier_gb": tier_gb, "readers": n_readers, "deliveries": len(deliveries), "wall_s": wall,
               **counts, "torn_per_10k": 1e4 * counts["torn"] / max(counts["reads"], 1),
               "missing_per_10k": 1e4 * counts["missing"] / max(counts["reads"], 1)}
        results.append(rec)
        print(json.dumps(rec), flush=True)
        store.drop()
    with open(out / f"concurrency_tier{tier_gb:g}.jsonl", "a" if methods else "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="B4_concurrency")
    ap.add_argument("--tier_gb", type=float, default=1.0)
    ap.add_argument("--readers", type=int, default=4)
    ap.add_argument("--methods", default="", help="comma-separated subset; results are appended")
    a = ap.parse_args()
    run(a.exp, a.tier_gb, a.readers, methods=[m for m in a.methods.split(",") if m])
