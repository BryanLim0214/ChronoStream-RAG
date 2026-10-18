"""Publication-logic replay (no vector index) for sweeping the holdback window H on real data.

It applies exactly the publication rules of `HistoryMethod` (duplicate drop, identity-revert
suppression, retraction when a revert arrives after publication, re-serving a revision that a later
arrival shows is not reverted, release at t_event + H, risk gating) and records when each revision
becomes the served head. It skips embedding and index writes, so it can sweep many H values over
every fetched revision. `experiments/run_holdback.py --gate_tier` checks that its head-change log
equals the full system's on a Benchmark B tier.
"""
import heapq
import itertools
from typing import Dict, List, Optional, Tuple

from csrag.revert import RevertTracker
from csrag.types import Delivery


def publication_replay(deliveries: List[Delivery], holdback_s: float, risk_gated: bool = False,
                       revert_detection: bool = True, supersede: bool = False) -> List[Tuple[float, str, Optional[int]]]:
    """Return the head-change log (t_proc, doc, head rev_id) for one H setting.

    supersede=True (simulator only, not part of HistoryMethod): a revision whose hold ends, or that is
    ready after its hold has ended, is skipped if a newer revision of the same document (by edit time)
    has already arrived. This needs no revert detector and is the baseline 'hold, then serve only the
    newest' that a reviewer asked for. With supersede=False the replay equals HistoryMethod.
    """
    newest_arrived: Dict[str, float] = {}
    tracker = RevertTracker()
    curated: Dict[str, List[Tuple[float, int]]] = {}
    published: Dict[str, set] = {}
    pending: Dict[Tuple[str, int], Delivery] = {}
    suppressed: Dict[Tuple[str, int], Delivery] = {}
    retracted: Dict[Tuple[str, int], Delivery] = {}
    by_key: Dict[Tuple[str, int], Delivery] = {}
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

    def superseded(r):
        return supersede and newest_arrived.get(r.doc_id, float("-inf")) > r.t_event

    def admit(d, now):
        r = d.rev
        H = holdback_s if (not risk_gated or r.risky) else 0.0
        release = r.t_event + H
        if release <= now:
            if not superseded(r):
                publish(r.doc_id, r.t_event, r.rev_id, now)
        else:
            pending[(r.doc_id, r.rev_id)] = d
            heapq.heappush(heap, (release, 1, next(seq), "t", (r.doc_id, r.rev_id)))

    while heap:
        now, _, _, kind, item = heapq.heappop(heap)
        if kind == "d":
            r = item.rev
            if tracker.known(r.doc_id, r.rev_id, r.t_event):
                continue
            by_key[(r.doc_id, r.rev_id)] = item
            newest_arrived[r.doc_id] = max(newest_arrived.get(r.doc_id, float("-inf")), r.t_event)
            changed = tracker.add(r.doc_id, r.rev_id, r.t_event, r.content_hash)
            if not revert_detection:
                changed = set()
            for rid in changed:
                if rid == r.rev_id:
                    continue
                key = (r.doc_id, rid)
                if tracker.is_reverted(r.doc_id, rid):
                    if rid in published.get(r.doc_id, ()):
                        retracted[key] = by_key[key]
                        retract(r.doc_id, rid, now)
                else:
                    d0 = suppressed.pop(key, None) or retracted.pop(key, None)
                    if d0 is not None:
                        admit(d0, now)
            if revert_detection and tracker.is_reverted(r.doc_id, r.rev_id):
                suppressed[(r.doc_id, r.rev_id)] = item
                continue
            admit(item, now)
        else:
            d = pending.pop(item, None)
            if d is None:
                continue
            r = d.rev
            if revert_detection and tracker.is_reverted(r.doc_id, r.rev_id):
                suppressed[(r.doc_id, r.rev_id)] = d
                continue
            if superseded(r):
                continue
            publish(r.doc_id, r.t_event, r.rev_id, now)
    return head_log
