"""Identity-revert detection from content hashes only.

A revision is an identity revert when its full-text hash equals the hash of one of the
previous `radius` revisions, other than the immediately previous one (that would be a null
edit). Every revision between the restored revision and the reverting revision is
"reverted". This follows the checksum method used in Wikipedia research (the semantics of
the `mwreverts` detector, radius 15). No edit comments, user names or labels are read.

`RevertTracker` is the online version used inside a running pipeline. Revisions may arrive
in any order. An arrival that lands at the end of a document's event-time order is
processed incrementally. An arrival that lands earlier triggers a recompute of that
document, so the result always equals the offline detector over the revisions seen so far,
in event-time order.
"""
import bisect
from typing import Dict, List, Set, Tuple

DEFAULT_RADIUS = 15


def _step(history, reverted, reverter_of, h, j, radius):
    match = None
    for k in range(len(history) - 1, -1, -1):
        if history[k][0] == h:
            match = k
            break
    if match is not None:
        for _, pos in history[match + 1:]:   # empty when the match is the previous revision (null edit)
            reverted.add(pos)
            reverter_of.setdefault(pos, j)
    history.append((h, j))
    if len(history) > radius + 1:
        history.pop(0)


def detect_reverts(hashes: List[str], radius: int = DEFAULT_RADIUS) -> Tuple[Set[int], Dict[int, int]]:
    """Offline detector over one document's revisions in event-time order.

    Returns (reverted_positions, reverter_of) where reverter_of[pos] is the position of the
    first revision that reverted it. Same semantics as mwreverts.Detector: a buffer of the
    last radius+1 (hash, position) entries; the most recent earlier occurrence of the same
    hash is the restored revision; every entry inserted after it is reverted. Entries are not
    removed early, so reverts of reverts (edit wars) are also detected.
    """
    history: List[Tuple[str, int]] = []
    reverted: Set[int] = set()
    reverter_of: Dict[int, int] = {}
    for j, h in enumerate(hashes):
        _step(history, reverted, reverter_of, h, j, radius)
    return reverted, reverter_of


class _DocState:
    __slots__ = ("keys", "hashes", "history", "reverted_pos", "reverter_of", "rev_ids")

    def __init__(self):
        self.keys: List[Tuple[float, int]] = []
        self.hashes: List[str] = []
        self.history: List[Tuple[str, int]] = []
        self.reverted_pos: Set[int] = set()
        self.reverter_of: Dict[int, int] = {}
        self.rev_ids: Set[int] = set()


class RevertTracker:
    """Online identity-revert detection, robust to out-of-order arrival."""

    def __init__(self, radius: int = DEFAULT_RADIUS):
        self.radius = radius
        self.docs: Dict[str, _DocState] = {}
        self.n_incremental = 0
        self.n_recomputes = 0

    def known(self, doc_id: str, rev_id: int, t_event: float) -> bool:
        st = self.docs.get(doc_id)
        if st is None:
            return False
        i = bisect.bisect_left(st.keys, (t_event, rev_id))
        return i < len(st.keys) and st.keys[i] == (t_event, rev_id)

    def _ids(self, st: _DocState) -> Set[int]:
        return {st.keys[p][1] for p in st.reverted_pos}

    def add(self, doc_id: str, rev_id: int, t_event: float, content_hash: str) -> Set[int]:
        """Insert a revision. Returns the rev_ids whose reverted status changed."""
        st = self.docs.setdefault(doc_id, _DocState())
        key = (t_event, rev_id)
        i = bisect.bisect_left(st.keys, key)
        if i < len(st.keys) and st.keys[i] == key:
            return set()
        before = st.rev_ids
        if i == len(st.keys):                       # in-order arrival: incremental step
            st.keys.append(key)
            st.hashes.append(content_hash)
            _step(st.history, st.reverted_pos, st.reverter_of, content_hash, i, self.radius)
            self.n_incremental += 1
        else:                                       # late arrival: recompute this document
            st.keys.insert(i, key)
            st.hashes.insert(i, content_hash)
            st.history, st.reverted_pos, st.reverter_of = [], set(), {}
            for j, h in enumerate(st.hashes):
                _step(st.history, st.reverted_pos, st.reverter_of, h, j, self.radius)
            self.n_recomputes += 1
        st.rev_ids = self._ids(st)
        return before.symmetric_difference(st.rev_ids)

    def is_reverted(self, doc_id: str, rev_id: int) -> bool:
        st = self.docs.get(doc_id)
        return st is not None and rev_id in st.rev_ids

    def reverted_ids(self, doc_id: str) -> Set[int]:
        st = self.docs.get(doc_id)
        return set(st.rev_ids) if st else set()
