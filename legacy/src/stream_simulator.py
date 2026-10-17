"""
ChronoStream-RAG: Stream Disorder, Compute-Lag, Duplicate & Interruption Simulator
==================================================================================
Implements a realistic partitioned stream ingestion and distributed embedding worker
pipeline that generates and logs:
  1. Source Version & Event Time (t_event)
  2. Broker Arrival Time (t_arrival) under controlled network out-of-order delays (p_ooo)
  3. At-least-once Duplicate Deliveries (p_dup)
  4. Endogenous Embedding Compute-Lag Disorder (t_embed_done) across parallel worker pools,
     including per-chunk asynchronous completion jitter (causing Cross-Chunk Version Tearing!)
  5. Controlled Stream Worker Interruptions & Parquet Checkpoint Recovery
"""

import copy
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class StreamConfig:
    seed: int = 42
    ooo_rate: float = 0.25          # Fraction of events delayed in network/Kafka partitions
    mean_delay_sec: float = 95.0    # Mean delay for out-of-order network arrivals
    delay_sigma: float = 0.65       # Log-normal dispersion of network delay
    dup_rate: float = 0.15          # At-least-once duplicate message rate
    compute_jitter_scale: float = 18.0  # Parallel embedding worker straggler scale (sec)
    chunk_tearing_prob: float = 0.30    # Prob that parallel chunk workers finish out-of-sync
    interruption_at_frac: float = 0.60  # Inject worker crash at 60% of stream progress
    enable_interruption: bool = True
    watermark_window_sec: float = 12.0  # Event-time watermark holdback window (for M4)


def generate_disordered_event_stream(
    revisions: List[Dict[str, Any]],
    cfg: StreamConfig
) -> List[Dict[str, Any]]:
    """
    Takes the chronological Wikimedia revision stream and produces a realistic
    disordered, duplicated, and compute-lagged stream of delivery events.
    Each event tracks:
      - event_time: when the edit occurred at the Wikimedia source
      - arrival_time: when the message arrived at the Kafka/Spark consumer
      - chunk_completion_times: when each chunk [0..k-1] finished embedding across worker pool
      - is_duplicate_delivery: whether this is a repeated broker delivery
    """
    rng = np.random.default_rng(cfg.seed)
    stream_events = []

    for rev in revisions:
        ev = copy.deepcopy(rev)
        t_event = float(ev["event_time_epoch"])

        # Base network transit latency (50ms - 300ms)
        base_net_lat = float(rng.uniform(0.05, 0.30))

        # Never delay the very first baseline revision of a doc so every doc starts initialized,
        # but apply controlled out-of-order network delay to subsequent updates
        is_delayed_net = False
        if ev["rev_num"] > 1 and rng.random() < cfg.ooo_rate:
            # Log-normal delay that pushes arrival_time past the next 1-2 revisions (spaced by 60s)
            net_delay = float(rng.lognormal(
                mean=np.log(max(cfg.mean_delay_sec, 1.0)),
                sigma=cfg.delay_sigma
            ))
            is_delayed_net = True
        else:
            net_delay = 0.0

        t_arrival = t_event + base_net_lat + net_delay

        # Simulate parallel worker pool embedding time:
        # Longer revisions (6 chunks) take longer to batch/embed than compact revisions (4 chunks).
        # If an older 6-chunk revision hits a slow worker straggler, its completion time
        # can invert relative to a newer 4-chunk revision EVEN IF arrival_time was in-order!
        num_chunks = ev["num_chunks"]
        base_embed_sec = 0.12 * num_chunks
        worker_straggler = 0.0
        if rng.random() < 0.28 and cfg.compute_jitter_scale > 0:
            # Heavy-tailed worker queue / batching straggler delay
            worker_straggler = float(rng.exponential(scale=cfg.compute_jitter_scale * (num_chunks / 4.0)))

        # Per-chunk completion times across asynchronous worker threads
        chunk_Embed_times = []
        for c_idx in range(num_chunks):
            if rng.random() < cfg.chunk_tearing_prob and cfg.compute_jitter_scale > 0:
                c_jitter = float(rng.exponential(scale=cfg.compute_jitter_scale * 0.85))
            else:
                c_jitter = float(rng.uniform(0.01, 0.08))
            chunk_Embed_times.append(t_arrival + base_embed_sec + worker_straggler + c_jitter)

        t_embed_complete = max(chunk_Embed_times)

        ev_record = {
            **ev,
            "stream_msg_id": f"msg_{ev['doc_id']}_v{ev['rev_num']}_orig",
            "is_duplicate_delivery": False,
            "is_network_delayed": is_delayed_net,
            "network_delay_sec": round(net_delay, 4),
            "worker_straggler_sec": round(worker_straggler, 4),
            "t_event": t_event,
            "t_arrival": t_arrival,
            "chunk_complete_times": chunk_Embed_times,
            "t_embed_complete": t_embed_complete
        }
        stream_events.append(ev_record)

        # At-least-once broker retry / duplicate delivery
        if rng.random() < cfg.dup_rate:
            dup_record = copy.deepcopy(ev_record)
            dup_offset = float(rng.uniform(35.0, 140.0))
            dup_record["stream_msg_id"] = f"msg_{ev['doc_id']}_v{ev['rev_num']}_dup1"
            dup_record["is_duplicate_delivery"] = True
            dup_record["t_arrival"] = t_arrival + dup_offset
            dup_record["chunk_complete_times"] = [
                t + dup_offset for t in chunk_Embed_times
            ]
            dup_record["t_embed_complete"] = t_embed_complete + dup_offset
            stream_events.append(dup_record)

    # Sort stream events by broker arrival time (t_arrival)
    stream_events.sort(key=lambda x: x["t_arrival"])
    return stream_events
