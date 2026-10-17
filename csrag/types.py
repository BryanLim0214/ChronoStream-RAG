"""Core data types.

`Revision` and `Delivery` are the only objects an update policy (method) may read.
Ground truth (facts, vandalism labels, revert labels) lives in `GroundTruth` and is
never passed to a method; the evaluator joins it by (doc_id, rev_id).
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

INF = 1e18  # open-ended time


@dataclass(frozen=True)
class Revision:
    doc_id: str
    rev_id: int
    t_event: float              # when the edit was made at the source (seconds since epoch)
    content_hash: str           # hash of the full revision text (Wikipedia: API sha1)
    chunks: Tuple[str, ...]
    chunk_hashes: Tuple[str, ...]
    risky: bool = False         # edit made by an anonymous or temporary account (visible metadata)


@dataclass(frozen=True)
class Delivery:
    rev: Revision
    msg_id: str
    is_dup: bool
    t_arrival: float                     # message reached the consumer
    t_avail: float                       # all chunk embeddings finished (revision ready)
    chunk_avail: Tuple[float, ...]       # per-chunk embedding completion times


@dataclass
class GTRevision:
    doc_id: str
    rev_id: int
    t_event: float
    content_hash: str
    chunk_set: frozenset
    reverted: bool                       # undone by a later identity revert (offline detector)
    reverted_tag: Optional[bool] = None  # MediaWiki `mw-reverted` tag (Benchmark B only)
    fact: Optional[str] = None           # Benchmark A only
    is_vandal: bool = False              # Benchmark A only (hand label)


@dataclass
class GroundTruth:
    docs: Dict[str, List[GTRevision]] = field(default_factory=dict)  # sorted by t_event

    def curated(self, doc_id: str) -> List[GTRevision]:
        return [r for r in self.docs[doc_id] if not r.reverted]
