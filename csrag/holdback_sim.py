"""Publication-logic replay (no vector index) for sweeping the holdback window H on real data.

It applies exactly the publication rules of `HistoryMethod` (duplicate drop, identity-revert
suppression, retraction when a revert arrives after publication, release at t_event + H,
risk gating) and records when each revision becomes the served head. Because it skips
embedding and index writes, it can sweep many H values over every fetched revision.
A gate in experiments/run_holdback.py checks it against full-system runs.
"""
import heapq
import itertools
from typing import Dict, List, Optional, Tuple

import numpy as np

from csrag.revert import RevertTracker
from csrag.types import Delivery


def publication_replay(deliveries: List[Delivery], holdback_s: float, risk_gated: bool = False,
                       revert_detection: bool = True) -> List[Tuple[float, str, Optional[int]]]:
    """Return the head-change log (t_proc, doc, head rev_id) for one H setting."""
    tracker = RevertTracker()
    curated: Dict[str, List[Tuple[float, int]]] = {}
    published: Dict[str, set] = {}
    pending: Dict[Tuple[str, int], Delivery] = {}
    head_log = []
    seq = itertools.count()
    heap = [(d.t_avail, 0, next(seq), "d", d) for d in deliveries]
    heapq.heapify(heap)

    def head(doc):
        c = curated.get(doc)
        return c[-1][1] if c else None

    def publish(doc, t_ev, rid, now):
        before = head(doc)
        c = curated.setdefault(doc, [])
        c.append((t_ev, rid))
        c.sort()
        published.setdefault(doc, set()).add(rid)
        after = head(doc)
        if after != before:
            head_log.append((now, doc, after))

    def retract(doc, rid, now):
        c = curated.get(doc, [])
        before = head(doc)
        curated[doc] = [x for x in c if x[1] != rid]
        published[doc].discard(rid)
        after = head(doc)
        if after != before:
            head_log.append((now, doc, after))

    while heap:
        now, _, _, kind, item = heapq.heappop(heap)
        if kind == "d":
            r = item.rev
            if tracker.known(r.doc_id, r.rev_id, r.t_event):
                continue
            changed = tracker.add(r.doc_id, r.rev_id, r.t_event, r.content_hash)
            if not revert_detection:
                changed = set()
            for rid in changed:
                if rid != r.rev_id and tracker.is_reverted(r.doc_id, rid) and rid in published.get(r.doc_id, ()):
                    retract(r.doc_id, rid, now)
            if revert_detection and tracker.is_reverted(r.doc_id, r.rev_id):
                continue
            H = holdback_s if (not risk_gated or r.risky) else 0.0
            release = r.t_event + H
            if release <= now:
                publish(r.doc_id, r.t_event, r.rev_id, now)
            else:
                pending[(r.doc_id, r.rev_id)] = item
                heapq.heappush(heap, (release, 1, next(seq), "t", (r.doc_id, r.rev_id)))
        else:
            d = pending.pop(item, None)
            if d is None:
                continue
            r = d.rev
            if revert_detection and tracker.is_reverted(r.doc_id, r.rev_id):
                continue
            publish(r.doc_id, r.t_event, r.rev_id, now)
    return head_log
