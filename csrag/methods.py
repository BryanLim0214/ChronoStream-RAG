"""Update policies that keep a chunked vector index in step with a stream of revisions.

Naive baselines (process each chunk as soon as its embedding finishes):
  overwrite      - writes chunk i of a document into slot i, no checks (arrival order wins).
  chunk_lww      - per-slot version check: a slot keeps the chunk with the newest edit time.
                   Slots that a shorter revision no longer uses are never cleared.

Realistic baselines (process a whole revision when all its embeddings finish):
  incremental    - like common indexing tools: chunk ids derived from content, insert the
                   new chunks, then delete the document's chunks not in this revision.
                   No version check, so an older revision that finishes late wins.
  doc_guard      - document version guard (drop a revision older than the current head),
                   then delete-then-insert of the document's chunks. Keeps the head only.
  tx_travel      - doc_guard that keeps old versions tagged with commit numbers, so it can
                   answer "what did the index hold at time t" (transaction time).

Valid-time history (this paper's composed design; every mechanism is a switch):
  full_history   - keeps every published revision with its edit time, splices late
                   revisions into place, retracts revisions undone by an identity revert,
                   multi-version (snapshot) commits so readers never see half an update.
  proposed       - full_history + holdback window H + hot window of K revisions per
                   document in the vector index, older revisions in a Parquet cold archive.

Methods never see ground truth. They read only Revision/Delivery fields.
"""
import bisect
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from qdrant_client import models

from csrag.cold import ColdArchive
from csrag.embed import Embedder
from csrag.revert import RevertTracker
from csrag.store import Store, point_id
from csrag.types import Delivery, Revision

MAXTX = 2 ** 62


@dataclass
class MethodConfig:
    name: str
    kind: str                       # overwrite | chunk_lww | incremental | doc_guard | tx_travel | history
    reuse: bool = True              # content-hash embedding reuse
    holdback_s: float = 0.0         # H
    risk_gated: bool = False        # apply H only to edits from anonymous/temporary accounts
    revert_detection: bool = True   # identity-revert retraction/suppression (history kind)
    hot_k: Optional[int] = None     # None = keep all published revisions in the vector index
    commit: str = "mvcc"            # mvcc | inplace
    archive: bool = True            # write the Parquet archive (needed for cold reads and recovery)

    def as_dict(self):
        return asdict(self)


def chunk_unit(cfg: MethodConfig) -> bool:
    return cfg.kind in ("overwrite", "chunk_lww")


class Base:
    def __init__(self, cfg: MethodConfig, embedder: Embedder, store: Store, archive_dir: Optional[Path]):
        self.cfg = cfg
        self.name = cfg.name
        self.emb = embedder
        self.store = store
        self.archive = ColdArchive(archive_dir) if (archive_dir and cfg.archive) else None
        self.head_log: List[Tuple[float, str, Optional[int]]] = []   # (t_proc, doc, head rev after change)
        self.counters = {"deliveries": 0, "duplicates_dropped": 0, "stale_dropped": 0,
                         "published": 0, "suppressed_reverted": 0, "retracted": 0,
                         "late_spliced": 0, "held": 0, "evicted_to_cold": 0, "cold_reads": 0,
                         "unreverted_lost": 0}
        self.lock = threading.Lock()

    # engine hooks
    def on_delivery(self, d: Delivery, now: float, sched: Callable):
        raise NotImplementedError

    def on_chunk(self, d: Delivery, ci: int, now: float):
        raise NotImplementedError

    def on_timer(self, key, now: float, sched: Callable):
        pass

    def flush(self, now: float):
        pass

    def finish(self):
        if self.archive:
            self.archive.flush_all()

    # reads
    def visible(self, doc_id: str, mode: str = "current", tau: Optional[float] = None) -> Set[str]:
        raise NotImplementedError

    def search(self, doc_id: str, qvec: np.ndarray, k: int, mode: str = "current",
               tau: Optional[float] = None) -> List[dict]:
        raise NotImplementedError

    def head_of(self, doc_id: str) -> Optional[int]:
        return None

    def memory_objects(self) -> dict:
        return {}


# ---------------------------------------------------------------------------
# Naive chunk-slot baselines
# ---------------------------------------------------------------------------
class SlotMethod(Base):
    """overwrite / chunk_lww. Writes are buffered per slot and applied before each probe;
    because the buffer keeps the last write per slot in processing order, the result equals
    applying each chunk write immediately."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.slot_version: Dict[Tuple[str, int], Tuple[float, int]] = {}
        self.pending: Dict[Tuple[str, int], Tuple[str, str, int, float]] = {}  # slot -> (hash, text, rev, t_event)

    def on_chunk(self, d: Delivery, ci: int, now: float):
        rev = d.rev
        key = (rev.doc_id, ci)
        ver = (rev.t_event, rev.rev_id)
        if self.cfg.kind == "chunk_lww":
            prev = self.slot_version.get(key)
            if prev is not None and ver <= prev:
                self.counters["stale_dropped" if ver < prev else "duplicates_dropped"] += 1
                return
        self.slot_version[key] = ver
        self.pending[key] = (rev.chunk_hashes[ci], rev.chunks[ci], rev.rev_id, rev.t_event)

    def on_delivery(self, d, now, sched):
        self.counters["deliveries"] += 1

    def flush(self, now):
        if not self.pending:
            return
        keys = list(self.pending)
        hs = [self.pending[k][0] for k in keys]
        txt = [self.pending[k][1] for k in keys]
        vecs = self.emb.embed(txt, hs)
        ids = [point_id(f"{k[0]}#{k[1]}") for k in keys]
        pls = [{"doc_id": k[0], "slot": k[1], "chunk_hash": h, "rev_id": self.pending[k][2],
                **({"text": t} if self.store.store_text else {})} for k, h, t in zip(keys, hs, txt)]
        self.store.upsert(ids, vecs, pls)
        self.pending.clear()

    def visible(self, doc_id, mode="current", tau=None):
        return {p["chunk_hash"] for p in self.store.scroll_payloads(Store.flt(doc_id))}

    def search(self, doc_id, qvec, k, mode="current", tau=None):
        return self.store.search(qvec, Store.flt(doc_id), k)


# ---------------------------------------------------------------------------
# Revision-level head-only baselines
# ---------------------------------------------------------------------------
class IncrementalMethod(Base):
    """Content-derived chunk ids; insert new chunks then clean up the document's other chunks.
    No version check: whichever revision is processed last becomes the head."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.records: Dict[str, Set[int]] = {}      # the "record manager": doc -> ids in the index
        self.head: Dict[str, int] = {}

    def on_delivery(self, d, now, sched):
        self.counters["deliveries"] += 1
        rev = d.rev
        ids = [point_id(f"{rev.doc_id}|{h}") for h in rev.chunk_hashes]
        have = self.records.get(rev.doc_id, set())
        new_pos, seen = [], set()
        for i, pid in enumerate(ids):
            if pid not in have and pid not in seen:
                seen.add(pid)
                new_pos.append(i)
        if new_pos:
            hs = [rev.chunk_hashes[i] for i in new_pos]
            txt = [rev.chunks[i] for i in new_pos]
            vecs = self.emb.embed(txt, hs)
            pls = [{"doc_id": rev.doc_id, "chunk_hash": h, "rev_id": rev.rev_id,
                    **({"text": t} if self.store.store_text else {})} for h, t in zip(hs, txt)]
            self.store.upsert([ids[i] for i in new_pos], vecs, pls)          # 1) insert
        keep = set(ids)
        stale = [pid for pid in have if pid not in keep]
        self.store.delete(stale)                                             # 2) clean up
        self.records[rev.doc_id] = keep
        if self.head.get(rev.doc_id) != rev.rev_id:
            self.head[rev.doc_id] = rev.rev_id
            self.head_log.append((now, rev.doc_id, rev.rev_id))
        if self.archive:
            self.archive.log_revision(rev.doc_id, rev.rev_id, rev.t_event, rev.chunk_hashes, "published", now)
        self.counters["published"] += 1

    def visible(self, doc_id, mode="current", tau=None):
        return {p["chunk_hash"] for p in self.store.scroll_payloads(Store.flt(doc_id))}

    def search(self, doc_id, qvec, k, mode="current", tau=None):
        return self.store.search(qvec, Store.flt(doc_id), k)

    def head_of(self, doc_id):
        return self.head.get(doc_id)


class DocGuardMethod(Base):
    """Document version guard + delete-then-insert. Optionally keeps old versions with commit
    numbers (tx_travel) to answer as-of questions in transaction time."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.head: Dict[str, Tuple[float, int, List[int]]] = {}
        self.S = 0
        self.commit_times: List[float] = []           # processing time of commit S (index S-1)
        self.keep_versions = self.cfg.kind == "tx_travel"

    def on_delivery(self, d, now, sched):
        self.counters["deliveries"] += 1
        rev = d.rev
        h = self.head.get(rev.doc_id)
        ver = (rev.t_event, rev.rev_id)
        if h is not None and ver <= (h[0], h[1]):
            self.counters["stale_dropped" if ver < (h[0], h[1]) else "duplicates_dropped"] += 1
            return
        vecs = self.emb.embed(rev.chunks, rev.chunk_hashes)
        ids = [point_id(f"{rev.doc_id}|{rev.rev_id}|{i}") for i in range(len(rev.chunks))]
        nxt = self.S + 1
        pls = [{"doc_id": rev.doc_id, "chunk_hash": hh, "rev_id": rev.rev_id, "tx_from": nxt, "tx_to": MAXTX,
                **({"text": t} if self.store.store_text else {})} for hh, t in zip(rev.chunk_hashes, rev.chunks)]
        if self.keep_versions:
            self.store.upsert(ids, vecs, pls)
            if h is not None:
                self.store.set_payload(h[2], {"tx_to": nxt})
        else:
            if h is not None:
                self.store.delete(h[2])                                       # 1) delete
            self.store.upsert(ids, vecs, pls)                                 # 2) insert
        self.S = nxt
        self.commit_times.append(now)
        self.head[rev.doc_id] = (rev.t_event, rev.rev_id, ids)
        self.head_log.append((now, rev.doc_id, rev.rev_id))
        self.counters["published"] += 1
        if self.archive:
            self.archive.add_chunks(rev.doc_id, rev.chunk_hashes, rev.chunks, vecs)
            self.archive.log_revision(rev.doc_id, rev.rev_id, rev.t_event, rev.chunk_hashes, "published", now)

    def _tx_filter(self, S):
        return [models.FieldCondition(key="tx_from", range=models.Range(lte=S)),
                models.FieldCondition(key="tx_to", range=models.Range(gt=S))]

    def _snapshot_at(self, tau):
        return bisect.bisect_right(self.commit_times, tau)  # number of commits with t_proc <= tau

    def _filter(self, doc_id, mode, tau):
        if self.keep_versions and mode != "current" and tau is not None:
            return Store.flt(doc_id, self._tx_filter(self._snapshot_at(tau)))
        return Store.flt(doc_id, [models.FieldCondition(key="tx_to", range=models.Range(gte=MAXTX))])

    def visible(self, doc_id, mode="current", tau=None):
        return {p["chunk_hash"] for p in self.store.scroll_payloads(self._filter(doc_id, mode, tau))}

    def search(self, doc_id, qvec, k, mode="current", tau=None):
        return self.store.search(qvec, self._filter(doc_id, mode, tau), k)

    def head_of(self, doc_id):
        h = self.head.get(doc_id)
        return h[1] if h else None

    def recover(self) -> dict:
        """Rebuild the head index from the archive (crash recovery)."""
        import time
        t0 = time.perf_counter()
        log = self.archive.read_revision_log().to_pylist()
        heads: Dict[str, dict] = {}
        for r in log:
            if r["status"] != "published":
                continue
            cur = heads.get(r["doc_id"])
            if cur is None or (r["t_event"], r["rev_id"]) > (cur["t_event"], cur["rev_id"]):
                heads[r["doc_id"]] = r
        self.store.recreate()
        n = 0
        for doc, r in heads.items():
            hs, txt, vecs = self.archive.chunk_vectors(doc, r["chunk_hashes"])
            pos = {h: i for i, h in enumerate(hs)}
            order = [pos[h] for h in r["chunk_hashes"]]
            ids = [point_id(f"{doc}|{r['rev_id']}|{i}") for i in range(len(order))]
            pls = [{"doc_id": doc, "chunk_hash": r["chunk_hashes"][i], "rev_id": r["rev_id"], "tx_from": 0,
                    "tx_to": MAXTX, **({"text": txt[order[i]]} if self.store.store_text else {})}
                   for i in range(len(order))]
            self.store.upsert(ids, vecs[order], pls)
            n += len(ids)
        t_ready = time.perf_counter() - t0
        t_green = self.store.wait_green()
        return {"recovery_serving_ready_s": t_ready, "recovery_index_green_s": t_ready + t_green,
                "recovered_points": n}


# ---------------------------------------------------------------------------
# Valid-time history (full_history / proposed and their ablations)
# ---------------------------------------------------------------------------
class _Doc:
    __slots__ = ("curated", "meta", "pending", "hot")

    def __init__(self):
        self.curated: List[Tuple[float, int]] = []          # published, not reverted, sorted
        self.meta: Dict[int, Tuple[float, int]] = {}         # rev_id -> (t_event, n_chunks), published
        self.pending: Dict[int, Tuple[Revision, np.ndarray]] = {}
        self.hot: Set[int] = set()                           # rev_ids with points in the vector index


class HistoryMethod(Base):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.tracker = RevertTracker()
        self.docs: Dict[str, _Doc] = {}
        self.S = 0                                  # published snapshot number
        self.closed: List[Tuple[int, List[int]]] = []   # (closed at S, ids) awaiting vacuum
        self.reader_min_snapshot: Optional[Callable[[], int]] = None
        self.suppressed_meta: Dict[Tuple[str, int], Revision] = {}
        if self.cfg.hot_k is not None:
            assert self.archive is not None, "a bounded hot window needs the cold archive"

    # ----- helpers -----
    def _doc(self, doc_id) -> _Doc:
        st = self.docs.get(doc_id)
        if st is None:
            st = self.docs[doc_id] = _Doc()
        return st

    def _ids(self, doc_id, rev_id, n):
        return [point_id(f"{doc_id}|{rev_id}|{i}") for i in range(n)]

    def _insert_points(self, rev: Revision, vecs: np.ndarray):
        nxt = self.S + 1 if self.cfg.commit == "mvcc" else 0
        ids = self._ids(rev.doc_id, rev.rev_id, len(rev.chunks))
        pls = [{"doc_id": rev.doc_id, "chunk_hash": h, "rev_id": rev.rev_id, "valid_from": rev.t_event,
                "tx_from": nxt, "tx_to": MAXTX, **({"text": t} if self.store.store_text else {})}
               for h, t in zip(rev.chunk_hashes, rev.chunks)]
        self.store.upsert(ids, vecs, pls)

    def _close_points(self, doc_id, rev_id, n):
        ids = self._ids(doc_id, rev_id, n)
        if self.cfg.commit == "mvcc":
            self.store.set_payload(ids, {"tx_to": self.S + 1})
            self.closed.append((self.S + 1, ids))
        else:
            self.store.delete(ids)

    def _commit(self, now):
        with self.lock:
            self.S += 1
        self.vacuum()

    def vacuum(self):
        if self.cfg.commit != "mvcc" or not self.closed:
            return
        horizon = self.S if self.reader_min_snapshot is None else min(self.S, self.reader_min_snapshot())
        keep, drop = [], []
        for s, ids in self.closed:
            (drop if s <= horizon else keep).append((s, ids))
        self.closed = keep
        for _, ids in drop:
            self.store.delete(ids)

    def _head(self, st: _Doc) -> Optional[int]:
        return st.curated[-1][1] if st.curated else None

    def _log_head(self, doc_id, before, now):
        after = self._head(self.docs[doc_id])
        if after != before:
            self.head_log.append((now, doc_id, after))

    def _enforce_hot_window(self, doc_id, st: _Doc):
        k = self.cfg.hot_k
        if k is None:
            return
        want = {rid for _, rid in st.curated[-k:]}
        for rid in list(st.hot - want):
            self._close_points(doc_id, rid, st.meta[rid][1])
            st.hot.discard(rid)
            self.counters["evicted_to_cold"] += 1

    # ----- publication -----
    def _publish(self, rev: Revision, vecs: np.ndarray, now: float):
        st = self._doc(rev.doc_id)
        before = self._head(st)
        key = (rev.t_event, rev.rev_id)
        if st.curated and key < st.curated[-1]:
            self.counters["late_spliced"] += 1
        bisect.insort(st.curated, key)
        st.meta[rev.rev_id] = (rev.t_event, len(rev.chunks))
        in_hot = self.cfg.hot_k is None or key in st.curated[-self.cfg.hot_k:]
        if in_hot:
            self._insert_points(rev, vecs)
            st.hot.add(rev.rev_id)
        if self.archive:
            self.archive.log_revision(rev.doc_id, rev.rev_id, rev.t_event, rev.chunk_hashes, "published", now)
        self._enforce_hot_window(rev.doc_id, st)
        self._commit(now)
        self.counters["published"] += 1
        self._log_head(rev.doc_id, before, now)

    def _retract(self, doc_id: str, rev_id: int, now: float):
        st = self._doc(doc_id)
        if rev_id not in st.meta:
            return
        before = self._head(st)
        t_ev, n = st.meta.pop(rev_id)
        st.curated.remove((t_ev, rev_id))
        if rev_id in st.hot:
            self._close_points(doc_id, rev_id, n)
            st.hot.discard(rev_id)
        if self.archive:
            self.archive.log_revision(doc_id, rev_id, t_ev, [], "retracted", now)
        # a retraction can bring an older revision back into the hot window
        if self.cfg.hot_k is not None:
            for t_k, rid in st.curated[-self.cfg.hot_k:]:
                if rid not in st.hot:
                    hs = self.archive.revision_hashes(doc_id, rid)
                    hs2, txt, vecs = self.archive.chunk_vectors(doc_id, hs)
                    pos = {h: i for i, h in enumerate(hs2)}
                    order = [pos[h] for h in hs]
                    self._insert_points(Revision(doc_id, rid, t_k, "", tuple(txt[i] for i in order), tuple(hs)),
                                        vecs[order])
                    st.hot.add(rid)
        self._commit(now)
        self.counters["retracted"] += 1
        self._log_head(doc_id, before, now)

    def on_delivery(self, d, now, sched):
        self.counters["deliveries"] += 1
        rev = d.rev
        if self.tracker.known(rev.doc_id, rev.rev_id, rev.t_event):
            self.counters["duplicates_dropped"] += 1
            return
        changed = self.tracker.add(rev.doc_id, rev.rev_id, rev.t_event, rev.content_hash)
        if not self.cfg.revert_detection:
            changed = set()
        vecs = self.emb.embed(rev.chunks, rev.chunk_hashes)
        if self.archive:
            self.archive.add_chunks(rev.doc_id, rev.chunk_hashes, rev.chunks, vecs)
        st = self._doc(rev.doc_id)
        # 1) revisions whose status changed because of this arrival
        for rid in changed:
            if rid == rev.rev_id:
                continue
            if self.tracker.is_reverted(rev.doc_id, rid):
                if rid in st.meta:
                    self._retract(rev.doc_id, rid, now)
                # pending ones are suppressed when their timer fires
            else:
                lost = self.suppressed_meta.pop((rev.doc_id, rid), None)
                if lost is not None:
                    self.counters["unreverted_lost"] += 1
        # 2) the new revision itself
        if self.cfg.revert_detection and self.tracker.is_reverted(rev.doc_id, rev.rev_id):
            self.counters["suppressed_reverted"] += 1
            self.suppressed_meta[(rev.doc_id, rev.rev_id)] = rev
            if self.archive:
                self.archive.log_revision(rev.doc_id, rev.rev_id, rev.t_event, rev.chunk_hashes, "suppressed", now)
            return
        H = self.cfg.holdback_s if (not self.cfg.risk_gated or rev.risky) else 0.0
        release = rev.t_event + H
        if release <= now:
            self._publish(rev, vecs, now)
        else:
            st.pending[rev.rev_id] = (rev, vecs)
            self.counters["held"] += 1
            sched(release, (rev.doc_id, rev.rev_id))

    def on_timer(self, key, now, sched):
        doc_id, rev_id = key
        st = self._doc(doc_id)
        item = st.pending.pop(rev_id, None)
        if item is None:
            return
        rev, vecs = item
        if self.cfg.revert_detection and self.tracker.is_reverted(doc_id, rev_id):
            self.counters["suppressed_reverted"] += 1
            self.suppressed_meta[(doc_id, rev_id)] = rev
            if self.archive:
                self.archive.log_revision(doc_id, rev_id, rev.t_event, rev.chunk_hashes, "suppressed", now)
            return
        self._publish(rev, vecs, now)

    # ----- reads -----
    def _vis(self):
        if self.cfg.commit != "mvcc":
            return []
        S = self.S
        return [models.FieldCondition(key="tx_from", range=models.Range(lte=S)),
                models.FieldCondition(key="tx_to", range=models.Range(gt=S))]

    def target_rev(self, doc_id, mode, tau) -> Optional[Tuple[float, int]]:
        st = self.docs.get(doc_id)
        if not st or not st.curated:
            return None
        if mode == "current" or tau is None:
            return st.curated[-1]
        i = bisect.bisect_right(st.curated, (tau, float("inf"))) - 1
        return st.curated[i] if i >= 0 else None

    def visible(self, doc_id, mode="current", tau=None):
        tr = self.target_rev(doc_id, mode, tau)
        if tr is None:
            return set()
        st = self.docs[doc_id]
        if tr[1] in st.hot:
            flt = Store.flt(doc_id, [models.FieldCondition(key="rev_id", match=models.MatchValue(value=tr[1]))] + self._vis())
            return {p["chunk_hash"] for p in self.store.scroll_payloads(flt)}
        self.counters["cold_reads"] += 1
        return set(self.archive.revision_hashes(doc_id, tr[1]))

    def search(self, doc_id, qvec, k, mode="current", tau=None):
        tr = self.target_rev(doc_id, mode, tau)
        if tr is None:
            return []
        st = self.docs[doc_id]
        if tr[1] in st.hot:
            flt = Store.flt(doc_id, [models.FieldCondition(key="rev_id", match=models.MatchValue(value=tr[1]))] + self._vis())
            return self.store.search(qvec, flt, k)
        self.counters["cold_reads"] += 1
        hs = self.archive.revision_hashes(doc_id, tr[1])
        hs2, txt, vecs = self.archive.chunk_vectors(doc_id, hs)
        if not hs2:
            return []
        sims = vecs @ qvec
        top = np.argsort(-sims)[:k]
        return [{"chunk_hash": hs2[i], "text": txt[i], "rev_id": tr[1], "score": float(sims[i]), "cold": True}
                for i in top]

    def head_of(self, doc_id):
        st = self.docs.get(doc_id)
        return self._head(st) if st else None

    def recover(self) -> dict:
        """Crash recovery: rebuild the in-memory timelines and the hot index from the archive."""
        import time
        t0 = time.perf_counter()
        log = self.archive.read_revision_log().sort_by([("t_proc", "ascending")]).to_pylist()
        curated: Dict[str, Dict[int, dict]] = {}
        for r in log:
            dct = curated.setdefault(r["doc_id"], {})
            if r["status"] == "published":
                dct[r["rev_id"]] = r
            elif r["status"] == "retracted":
                dct.pop(r["rev_id"], None)
        self.store.recreate()
        self.docs = {}
        self.closed = []
        self.S = 0
        n = 0
        for doc, revs in curated.items():
            st = self._doc(doc)
            for r in revs.values():
                st.curated.append((r["t_event"], r["rev_id"]))
                st.meta[r["rev_id"]] = (r["t_event"], len(r["chunk_hashes"]))
            st.curated.sort()
            hot = st.curated if self.cfg.hot_k is None else st.curated[-self.cfg.hot_k:]
            for t_k, rid in hot:
                hs = revs[rid]["chunk_hashes"]
                hs2, txt, vecs = self.archive.chunk_vectors(doc, hs)
                pos = {h: i for i, h in enumerate(hs2)}
                order = [pos[h] for h in hs]
                self._insert_points(Revision(doc, rid, t_k, "", tuple(txt[i] for i in order), tuple(hs)), vecs[order])
                st.hot.add(rid)
                n += len(hs)
        self.S = 1
        t_ready = time.perf_counter() - t0
        t_green = self.store.wait_green()
        return {"recovery_serving_ready_s": t_ready, "recovery_index_green_s": t_ready + t_green,
                "recovered_points": n}

    def memory_objects(self) -> dict:
        pend = sum(len(st.pending) for st in self.docs.values())
        hot = sum(len(st.hot) for st in self.docs.values())
        cur = sum(len(st.curated) for st in self.docs.values())
        return {"pending_revisions": pend, "hot_revisions": hot, "published_revisions": cur}


def make_method(cfg: MethodConfig, embedder: Embedder, store: Store, archive_dir: Optional[Path]) -> Base:
    cls = {"overwrite": SlotMethod, "chunk_lww": SlotMethod, "incremental": IncrementalMethod,
           "doc_guard": DocGuardMethod, "tx_travel": DocGuardMethod, "history": HistoryMethod}[cfg.kind]
    return cls(cfg, embedder, store, archive_dir)


# Named configurations used in the paper (descriptive names, not M1-M4)
def standard_methods(holdback_s: float = 60.0, hot_k: int = 2) -> Dict[str, MethodConfig]:
    return {
        "overwrite": MethodConfig("overwrite", "overwrite", archive=False),
        "chunk_lww": MethodConfig("chunk_lww", "chunk_lww", archive=False),
        "incremental": MethodConfig("incremental", "incremental", archive=False),
        "doc_guard": MethodConfig("doc_guard", "doc_guard"),
        "tx_travel": MethodConfig("tx_travel", "tx_travel", archive=False),
        "full_history": MethodConfig("full_history", "history", holdback_s=0.0, hot_k=None),
        "proposed": MethodConfig("proposed", "history", holdback_s=holdback_s, hot_k=hot_k),
    }
