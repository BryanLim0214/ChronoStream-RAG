"""Discrete-event replay of a delivery stream through one update policy.

Simulated time decides ORDER (when each revision or chunk becomes available, when a
holdback timer fires, when a probe looks at the index). Wall-clock time measures COST
(embedding, index writes, archive writes). Probes see the index after every event with
an earlier or equal simulated time has been applied.
"""
import heapq
import itertools
import time
from typing import Callable, Dict, List, Tuple

import numpy as np

from csrag.methods import Base, chunk_unit
from csrag.types import Delivery

PRI_DELIVERY, PRI_TIMER, PRI_PROBE = 0, 1, 2


def replay(method: Base, deliveries: List[Delivery], probes: List[Tuple[float, object]],
           on_probe: Callable[[Base, object, float], None], progress_every: int = 0) -> Dict:
    seq = itertools.count()
    heap: list = []
    if chunk_unit(method.cfg):
        for d in deliveries:
            for ci, t in enumerate(d.chunk_avail):
                heap.append((t, PRI_DELIVERY, next(seq), "chunk", (d, ci)))
            heap.append((d.t_avail, PRI_DELIVERY, next(seq), "count", d))
    else:
        for d in deliveries:
            heap.append((d.t_avail, PRI_DELIVERY, next(seq), "deliv", d))
    for t, p in probes:
        heap.append((t, PRI_PROBE, next(seq), "probe", p))
    heapq.heapify(heap)

    def sched(t, key):
        heapq.heappush(heap, (t, PRI_TIMER, next(seq), "timer", key))

    ingest_s = probe_s = 0.0
    lat_ms: List[float] = []
    n_deliv = n_events = 0
    t_start = time.perf_counter()
    dirty = False
    while heap:
        t, _, _, kind, item = heapq.heappop(heap)
        n_events += 1
        if kind == "probe":
            t0 = time.perf_counter()
            if dirty:
                method.flush(t)
                dirty = False
            ingest_s += time.perf_counter() - t0
            t1 = time.perf_counter()
            on_probe(method, item, t)
            probe_s += time.perf_counter() - t1
            continue
        t0 = time.perf_counter()
        if kind == "deliv":
            method.on_delivery(item, t, sched)
            n_deliv += 1
        elif kind == "chunk":
            method.on_chunk(item[0], item[1], t)
            dirty = True
        elif kind == "count":
            method.counters["deliveries"] += 1
            n_deliv += 1
        elif kind == "timer":
            method.on_timer(item, t, sched)
        dt = time.perf_counter() - t0
        ingest_s += dt
        if kind == "deliv":
            lat_ms.append(dt * 1000.0)
        if progress_every and n_deliv and n_deliv % progress_every == 0 and kind in ("deliv", "count"):
            print(f"  [{method.name}] deliveries={n_deliv} ingest_s={ingest_s:.1f} probe_s={probe_s:.1f}", flush=True)
    t0 = time.perf_counter()
    method.flush(float("inf"))
    method.finish()
    ingest_s += time.perf_counter() - t0
    lat = np.array(lat_ms) if lat_ms else np.array([np.nan])
    return {"deliveries": n_deliv, "events": n_events, "ingest_wall_s": ingest_s, "probe_wall_s": probe_s,
            "total_wall_s": time.perf_counter() - t_start,
            "ingest_latency_ms_p50": float(np.nanpercentile(lat, 50)),
            "ingest_latency_ms_p95": float(np.nanpercentile(lat, 95)),
            "ingest_latency_ms_p99": float(np.nanpercentile(lat, 99)),
            "ingest_latency_ms_mean": float(np.nanmean(lat))}
