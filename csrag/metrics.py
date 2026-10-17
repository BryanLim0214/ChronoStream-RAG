"""Reader-free consistency measures, computed against ground truth.

For a document and a probe time p, an index exposes a set V of chunk hashes (what a
query restricted to that document can find). We compare V with the chunk sets of the
document's real revisions:

  exact    V equals the target revision's chunks.
  exposed  V equals a revision that was later undone by a revert, or contains a chunk that
           only ever appeared in such revisions.
  stale    V equals an older (not reverted) revision than the target.
  other    V equals some other single revision (for example a newer one, for as-of probes).
  torn     V equals no revision that ever existed (a mix of revisions, or orphan chunks).
  missing  V is empty although a target exists.

Targets:
  deliverable   latest non-reverted revision whose embeddings had finished by p
                (the best a pipeline could show without knowing the future).
  source        latest non-reverted revision with edit time <= p (what the page "should" say).
  as of tau     latest non-reverted revision with edit time <= tau.
"""
import bisect
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from csrag.types import Delivery, GTRevision


@dataclass
class DocTruth:
    revs: List[GTRevision]                       # sorted by t_event
    t_event_curated: List[float]
    curated: List[GTRevision]
    set_index: Dict[frozenset, List[int]]        # chunk set -> indices into revs
    reverted_only: Set[str]
    first_arrival: Dict[int, float]
    first_avail: Dict[int, float]
    avail_sorted: List[Tuple[float, float, int]]  # (t_avail, t_event, idx in curated) for curated


def build_truth(gt_docs: Dict[str, List[GTRevision]], deliveries: List[Delivery]) -> Dict[str, DocTruth]:
    first_arr: Dict[Tuple[str, int], float] = {}
    first_av: Dict[Tuple[str, int], float] = {}
    for d in deliveries:
        k = (d.rev.doc_id, d.rev.rev_id)
        first_arr[k] = min(first_arr.get(k, float("inf")), d.t_arrival)
        first_av[k] = min(first_av.get(k, float("inf")), d.t_avail)
    out = {}
    for doc, revs in gt_docs.items():
        set_index: Dict[frozenset, List[int]] = {}
        good_chunks, bad_chunks = set(), set()
        for i, r in enumerate(revs):
            set_index.setdefault(r.chunk_set, []).append(i)
            (bad_chunks if r.reverted else good_chunks).update(r.chunk_set)
        curated = [r for r in revs if not r.reverted]
        fa = {r.rev_id: first_arr.get((doc, r.rev_id), float("inf")) for r in revs}
        fv = {r.rev_id: first_av.get((doc, r.rev_id), float("inf")) for r in revs}
        avail_sorted = sorted((fv[r.rev_id], r.t_event, j) for j, r in enumerate(curated))
        out[doc] = DocTruth(revs, [r.t_event for r in curated], curated, set_index,
                            bad_chunks - good_chunks, fa, fv, avail_sorted)
    return out


def source_target(dt: DocTruth, p: float) -> Optional[GTRevision]:
    i = bisect.bisect_right(dt.t_event_curated, p) - 1
    return dt.curated[i] if i >= 0 else None


def deliverable_target(dt: DocTruth, p: float) -> Optional[GTRevision]:
    """Latest (by edit time) non-reverted revision available by p."""
    best = None
    for t_av, t_ev, j in dt.avail_sorted:
        if t_av > p:
            break
        if best is None or t_ev > best.t_event:
            best = dt.curated[j]
    return best


def classify(dt: DocTruth, V: Set[str], target: Optional[GTRevision]) -> str:
    if target is None:
        return "no_target"
    if not V:
        return "missing"
    fv = frozenset(V)
    if fv == target.chunk_set:
        return "exact"
    idx = dt.set_index.get(fv)
    if V & dt.reverted_only or (idx and any(dt.revs[i].reverted for i in idx)):
        return "exposed"
    if idx:
        if any((not dt.revs[i].reverted) and dt.revs[i].t_event < target.t_event for i in idx):
            return "stale"
        return "other"
    return "torn"


def attribute(dt: DocTruth, V: Set[str], p: float, held: bool) -> str:
    """Three-clock attribution of a current-state probe against the source target."""
    s = source_target(dt, p)
    if s is None:
        return "no_target"
    if V and frozenset(V) == s.chunk_set:
        return "correct"
    if dt.first_arrival[s.rev_id] > p:
        return "not_arrived"
    if dt.first_avail[s.rev_id] > p:
        return "in_flight"
    if held:
        return "held"
    return "index_error"
