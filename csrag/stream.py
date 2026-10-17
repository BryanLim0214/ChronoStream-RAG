"""Delivery simulator: network disorder, duplicate delivery, and embedding-worker delays.

Everything here is SIMULATED and labeled as such in the paper. Revisions keep their real
(Benchmark B) or synthetic (Benchmark A) event times; this module only decides when each
revision reaches the consumer (t_arrival) and when its chunk embeddings finish (t_avail).

Disorder model (all parameters in the experiment config):
  * network delay: with probability p_ooo a message is delayed by a log-normal amount
    (median `delay_median_s`, shape `delay_sigma`); otherwise 50-300 ms.
  * duplicates: with probability `dup_rate` a second copy arrives 35-140 s later
    (at-least-once delivery).
  * embedding: each chunk takes `embed_s_per_chunk`; with probability `straggler_p` the
    whole revision waits an extra exponential(`straggler_scale_s`) (slow worker); with
    probability `chunk_jitter_p` an individual chunk waits an extra
    exponential(`chunk_jitter_scale_s`) (chunks embedded by separate workers).
"""
from dataclasses import dataclass, asdict
from typing import List

import numpy as np

from csrag.types import Delivery, Revision


@dataclass
class DisorderConfig:
    p_ooo: float = 0.25
    delay_median_s: float = 95.0
    delay_sigma: float = 0.65
    dup_rate: float = 0.15
    dup_offset_min_s: float = 35.0
    dup_offset_max_s: float = 140.0
    embed_s_per_chunk: float = 0.12
    straggler_p: float = 0.28
    straggler_scale_s: float = 18.0
    chunk_jitter_p: float = 0.30
    chunk_jitter_scale_s: float = 15.3

    def as_dict(self):
        return asdict(self)


def simulate_deliveries(revisions: List[Revision], cfg: DisorderConfig, seed: int) -> List[Delivery]:
    """Return deliveries sorted by t_avail (the order a pipeline would process them)."""
    rng = np.random.default_rng(seed)
    out: List[Delivery] = []
    # Iterate in a fixed order so the random stream is reproducible.
    for rev in sorted(revisions, key=lambda r: (r.doc_id, r.t_event, r.rev_id)):
        net = float(rng.uniform(0.05, 0.30))
        if rng.random() < cfg.p_ooo:
            net += float(rng.lognormal(mean=np.log(max(cfg.delay_median_s, 1e-3)), sigma=cfg.delay_sigma))
        t_arr = rev.t_event + net
        n = len(rev.chunks)
        straggle = float(rng.exponential(cfg.straggler_scale_s)) if (cfg.straggler_p > 0 and rng.random() < cfg.straggler_p) else 0.0
        base = cfg.embed_s_per_chunk * n
        jit = []
        for _ in range(n):
            if cfg.chunk_jitter_p > 0 and rng.random() < cfg.chunk_jitter_p:
                jit.append(float(rng.exponential(cfg.chunk_jitter_scale_s)))
            else:
                jit.append(float(rng.uniform(0.01, 0.08)))
        chunk_avail = tuple(t_arr + base + straggle + j for j in jit)
        t_avail = max(chunk_avail) if chunk_avail else t_arr + base + straggle
        out.append(Delivery(rev, f"{rev.doc_id}:{rev.rev_id}:0", False, t_arr, t_avail, chunk_avail))
        if cfg.dup_rate > 0 and rng.random() < cfg.dup_rate:
            off = float(rng.uniform(cfg.dup_offset_min_s, cfg.dup_offset_max_s))
            out.append(Delivery(rev, f"{rev.doc_id}:{rev.rev_id}:1", True, t_arr + off, t_avail + off,
                                tuple(t + off for t in chunk_avail)))
    out.sort(key=lambda d: (d.t_avail, d.msg_id))
    return out


def availability_inversion_rate(deliveries: List[Delivery]) -> float:
    """Share of same-document revision pairs whose processing order differs from event order.

    Uses the first delivery of each revision. Pairs are consecutive in event time.
    """
    first = {}
    for d in deliveries:
        k = (d.rev.doc_id, d.rev.rev_id)
        if k not in first or d.t_avail < first[k].t_avail:
            first[k] = d
    by_doc = {}
    for d in first.values():
        by_doc.setdefault(d.rev.doc_id, []).append(d)
    inv = tot = 0
    for ds in by_doc.values():
        ds.sort(key=lambda d: (d.rev.t_event, d.rev.rev_id))
        for a, b in zip(ds, ds[1:]):
            tot += 1
            if b.t_avail < a.t_avail:
                inv += 1
    return inv / tot if tot else 0.0
