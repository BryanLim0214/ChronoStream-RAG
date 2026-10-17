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
    layout: str = "interval"        # interval | revision (history kind)
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
TINF = 1e18   # open-ended valid_to


class _Doc:
    __slots__ = ("curated", "meta", "pending", "hot", "sets", "intervals", "by_hash")

    def __init__(self):
        self.curated: List[Tuple[float, int]] = []          # published, not reverted, sorted by (t_event, rev_id)
        self.meta: Dict[int, Tuple[float, int]] = {}         # rev_id -> (t_event, n_chunks), published
        self.pending: Dict[int, Tuple[Revision, np.ndarray]] = {}
        self.hot: Set[int] = set()                           # revision layout: rev_ids with points in the index
        self.sets: Dict[int, frozenset] = {}                 # interval layout: chunk-hash set per published revision
        self.intervals: Dict[Tuple[str, float, float], int] = {}   # interval layout: (hash, from, to) -> point id (hot)
        self.by_hash: Dict[str, Set[Tuple[float, float]]] = {}


class HistoryMethod(Base):
    """Valid-time history with optional holdback, revert retraction, hot window and cold archive.

    layout = "revision": every published revision stores all of its chunks as points.
    layout = "interval": every chunk is stored once per maximal run of consecutive published
                         revisions that contain it, as a point valid in [valid_from, valid_to).
    commit = "mvcc":     new points are written with tx_from = S+1 (invisible), replaced points get
                         tx_to = S+1, then the snapshot number S is advanced in one step, so a reader
                         at snapshot S sees either the whole update or none of it.
    commit = "inplace":  points are inserted and deleted directly (no snapshot isolation).
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.tracker = RevertTracker()
        self.docs: Dict[str, _Doc] = {}
        self.S = 0
        self.closed: List[Tuple[int, List[int]]] = []
        self.reader_min_snapshot: Optional[Callable[[], int]] = None
        self.suppressed_meta: Dict[Tuple[str, int], Revision] = {}
        self.chunk_text: Dict[str, str] = {}
        self.layout = getattr(self.cfg, "layout", "interval")
        if self.cfg.hot_k is not None:
            assert self.archive is not None, "a bounded hot window needs the cold archive"

    # ------------------------------------------------------------ shared helpers
    def _doc(self, doc_id) -> _Doc:
        st = self.docs.get(doc_id)
        if st is None:
            st = self.docs[doc_id] = _Doc()
        return st

    def _tx_new(self):
        return self.S + 1 if self.cfg.commit == "mvcc" else 0

    def _close_ids(self, ids: List[int]):
        if not ids:
            return
        if self.cfg.commit == "mvcc":
            self.store.set_payload(ids, {"tx_to": self.S + 1})
            self.closed.append((self.S + 1, ids))
        else:
            self.store.delete(ids)

    def _commit(self):
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
        ids = [i for _, x in drop for i in x]
        if ids:
            self.store.delete(ids)

    def _vis(self):
        if self.cfg.commit != "mvcc":
            return []
        S = self.S
        return [models.FieldCondition(key="tx_from", range=models.Range(lte=S)),
                models.FieldCondition(key="tx_to", range=models.Range(gt=S))]

    def _head(self, st: _Doc) -> Optional[int]:
        return st.curated[-1][1] if st.curated else None

    def _log_head(self, doc_id, before, now):
        after = self._head(self.docs[doc_id])
        if after != before:
            self.head_log.append((now, doc_id, after))

    def _boundary(self, st: _Doc) -> float:
        """Edit time from which history is kept in the vector index (older -> cold archive)."""
        k = self.cfg.hot_k
        if k is None or len(st.curated) <= k:
            return -TINF
        return st.curated[-k][0]

    def _vec_text(self, doc_id: str, hashes: List[str], fresh: Dict[str, Tuple[str, np.ndarray]]):
        """Vectors and texts for chunk hashes: new revision first, then caches, then the archive."""
        out, missing = {}, []
        for h in hashes:
            if h in fresh:
                out[h] = fresh[h]
                continue
            v = self.emb.cache.get(h) if self.emb.reuse else None
            t = self.chunk_text.get(h)
            if v is not None and t is not None:
                out[h] = (t, v)
            else:
                missing.append(h)
        if missing:
            hs, txt, vecs = self.archive.chunk_vectors(doc_id, missing)
            for h, t, v in zip(hs, txt, vecs):
                out[h] = (t, v)
        return out

    # ------------------------------------------------------------ revision layout
    def _rev_ids(self, doc_id, rev_id, n):
        return [point_id(f"{doc_id}|{rev_id}|{i}") for i in range(n)]

    def _rev_insert(self, doc_id, rev_id, t_event, hashes, texts, vecs):
        ids = self._rev_ids(doc_id, rev_id, len(hashes))
        nxt = self._tx_new()
        pls = [{"doc_id": doc_id, "chunk_hash": h, "rev_id": rev_id, "valid_from": t_event,
                "tx_from": nxt, "tx_to": MAXTX, **({"text": t} if self.store.store_text else {})}
               for h, t in zip(hashes, texts)]
        self.store.upsert(ids, vecs, pls)

    def _rev_sync_hot(self, doc_id, st: _Doc, fresh_rev: Optional[Revision] = None, fresh_vecs=None):
        b = self._boundary(st)
        want = {rid for t, rid in st.curated if t >= b}
        for rid in list(st.hot - want):
            self._close_ids(self._rev_ids(doc_id, rid, st.meta[rid][1]))
            st.hot.discard(rid)
            self.counters["evicted_to_cold"] += 1
        for t, rid in st.curated:
            if t >= b and rid not in st.hot:
                if fresh_rev is not None and rid == fresh_rev.rev_id:
                    self._rev_insert(doc_id, rid, t, fresh_rev.chunk_hashes, fresh_rev.chunks, fresh_vecs)
                else:
                    hs = self.archive.revision_hashes(doc_id, rid)
                    vt = self._vec_text(doc_id, list(dict.fromkeys(hs)), {})
                    self._rev_insert(doc_id, rid, t, hs, [vt[h][0] for h in hs], np.stack([vt[h][1] for h in hs]))
                st.hot.add(rid)

    # ------------------------------------------------------------ interval layout
    def _runs(self, st: _Doc, h: str) -> List[Tuple[float, float]]:
        runs, start = [], None
        for t, rid in st.curated:
            inside = h in st.sets[rid]
            if inside and start is None:
                start = t
            elif not inside and start is not None:
                runs.append((start, t))
                start = None
        if start is not None:
            runs.append((start, TINF))
        return runs

    def _iv_sync(self, doc_id: str, st: _Doc, hashes, fresh: Dict[str, Tuple[str, np.ndarray]]):
        """Make the hot intervals of `hashes` equal to the desired runs (only hot ones: valid_to > boundary)."""
        b = self._boundary(st)
        to_close, to_insert = [], []
        for h in hashes:
            desired = {r for r in self._runs(st, h) if r[1] > b}
            current = st.by_hash.get(h, set())
            for r in current - desired:
                to_close.append((h, r))
            for r in desired - current:
                to_insert.append((h, r))
        for h, r in to_close:
            pid = st.intervals.pop((h, r[0], r[1]))
            st.by_hash[h].discard(r)
            if not st.by_hash[h]:
                del st.by_hash[h]
        self._close_ids([point_id(f"{doc_id}|{h}|{r[0]}|{r[1]}") for h, r in to_close])
        if to_insert:
            vt = self._vec_text(doc_id, list({h for h, _ in to_insert}), fresh)
            nxt = self._tx_new()
            ids, vecs, pls = [], [], []
            for h, r in to_insert:
                pid = point_id(f"{doc_id}|{h}|{r[0]}|{r[1]}")
                st.intervals[(h, r[0], r[1])] = pid
                st.by_hash.setdefault(h, set()).add(r)
                t, v = vt[h]
                self.chunk_text[h] = t
                ids.append(pid)
                vecs.append(v)
                pls.append({"doc_id": doc_id, "chunk_hash": h, "valid_from": r[0], "valid_to": r[1],
                            "tx_from": nxt, "tx_to": MAXTX, **({"text": t} if self.store.store_text else {})})
            self.store.upsert(ids, np.stack(vecs), pls)

    def _all_runs(self, st: _Doc) -> List[Tuple[str, float, float]]:
        """All maximal runs in one pass over the published timeline."""
        open_: Dict[str, float] = {}
        out = []
        for t, rid in st.curated:
            cur = st.sets[rid]
            for h in [h for h in open_ if h not in cur]:
                out.append((h, open_.pop(h), t))
            for h in cur:
                if h not in open_:
                    open_[h] = t
        out.extend((h, f, TINF) for h, f in open_.items())
        return out

    def _iv_rebuild(self, doc_id: str, st: _Doc):
        """Insert every hot interval of a document from scratch (used by crash recovery)."""
        b = self._boundary(st)
        runs = [r for r in self._all_runs(st) if r[2] > b]
        if not runs:
            return
        vt = self._vec_text(doc_id, list({h for h, _, _ in runs}), {})
        nxt = self._tx_new()
        ids, vecs, pls = [], [], []
        for h, f, t in runs:
            pid = point_id(f"{doc_id}|{h}|{f}|{t}")
            st.intervals[(h, f, t)] = pid
            st.by_hash.setdefault(h, set()).add((f, t))
            txt, v = vt[h]
            ids.append(pid)
            vecs.append(v)
            pls.append({"doc_id": doc_id, "chunk_hash": h, "valid_from": f, "valid_to": t,
                        "tx_from": nxt, "tx_to": MAXTX, **({"text": txt} if self.store.store_text else {})})
        self.store.upsert(ids, np.stack(vecs), pls)

    def _iv_evict(self, doc_id: str, st: _Doc):
        b = self._boundary(st)
        if b <= -TINF:
            return
        gone = [(h, f, t) for (h, f, t) in st.intervals if t <= b]
        if not gone:
            return
        for h, f, t in gone:
            st.intervals.pop((h, f, t))
            st.by_hash[h].discard((f, t))
            if not st.by_hash[h]:
                del st.by_hash[h]
        self._close_ids([point_id(f"{doc_id}|{h}|{f}|{t}") for h, f, t in gone])
        self.counters["evicted_to_cold"] += len(gone)

    @staticmethod
    def _affected(sets_before_after) -> Set[str]:
        """Hashes whose membership differs among neighbouring revisions (only their runs can change)."""
        present = [s for s in sets_before_after if s is not None]
        if not present:
            return set()
        union = set().union(*present)
        inter = set(present[0]).intersection(*present[1:]) if len(present) > 1 else set()
        if len(present) < 3:                     # at an edge: everything in the union may change
            return union - inter if len(present) == 2 else union
        return union - inter

    # ------------------------------------------------------------ publication
    def _publish(self, rev: Revision, vecs: np.ndarray, now: float):
        st = self._doc(rev.doc_id)
        before = self._head(st)
        key = (rev.t_event, rev.rev_id)
        if st.curated and key < st.curated[-1]:
            self.counters["late_spliced"] += 1
        old_b = self._boundary(st)
        bisect.insort(st.curated, key)
        p = st.curated.index(key)
        st.meta[rev.rev_id] = (rev.t_event, len(rev.chunks))
        hset = frozenset(rev.chunk_hashes)
        if self.archive:
            self.archive.log_revision(rev.doc_id, rev.rev_id, rev.t_event, rev.chunk_hashes, "published", now)
        if self.layout == "revision":
            self._rev_sync_hot(rev.doc_id, st, rev, vecs)
        else:
            st.sets[rev.rev_id] = hset
            A = st.sets[st.curated[p - 1][1]] if p > 0 else None
            B = st.sets[st.curated[p + 1][1]] if p + 1 < len(st.curated) else None
            fresh = {h: (t, v) for h, t, v in zip(rev.chunk_hashes, rev.chunks, vecs)}
            if p > 0 and B is None:
                aff = set(A ^ hset)                                  # append at the end
            else:
                aff = self._affected([A, hset, B]) if (A is not None and B is not None) else set(hset | (A or set()) | (B or set()))
            if self._boundary(st) < old_b:
                aff |= set().union(*[st.sets[r] for t, r in st.curated if t >= self._boundary(st)])
            self._iv_sync(rev.doc_id, st, aff, fresh)
            self._iv_evict(rev.doc_id, st)
        self._commit()
        self.counters["published"] += 1
        self._log_head(rev.doc_id, before, now)

    def _retract(self, doc_id: str, rev_id: int, now: float):
        st = self._doc(doc_id)
        if rev_id not in st.meta:
            return
        before = self._head(st)
        t_ev, n = st.meta.pop(rev_id)
        p = st.curated.index((t_ev, rev_id))
        old_b = self._boundary(st)
        st.curated.pop(p)
        if self.archive:
            self.archive.log_revision(doc_id, rev_id, t_ev, [], "retracted", now)
        if self.layout == "revision":
            if rev_id in st.hot:
                self._close_ids(self._rev_ids(doc_id, rev_id, n))
                st.hot.discard(rev_id)
            self._rev_sync_hot(doc_id, st)
        else:
            X = st.sets.pop(rev_id)
            A = st.sets[st.curated[p - 1][1]] if p > 0 else None
            B = st.sets[st.curated[p][1]] if p < len(st.curated) else None
            aff = set(X) | (A ^ X if A is not None else set()) | (B ^ X if B is not None else set())
            new_b = self._boundary(st)
            if new_b < old_b:                     # older history moves back into the hot window
                aff |= set().union(*[st.sets[r] for t, r in st.curated if t >= new_b]) if st.curated else set()
            self._iv_sync(doc_id, st, aff, {})
        self._commit()
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
        for rid in changed:
            if rid == rev.rev_id:
                continue
            if self.tracker.is_reverted(rev.doc_id, rid):
                if rid in st.meta:
                    self._retract(rev.doc_id, rid, now)
            else:
                if self.suppressed_meta.pop((rev.doc_id, rid), None) is not None:
                    self.counters["unreverted_lost"] += 1
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

    # ------------------------------------------------------------ reads
    def target_rev(self, doc_id, mode, tau) -> Optional[Tuple[float, int]]:
        st = self.docs.get(doc_id)
        if not st or not st.curated:
            return None
        if mode == "current" or tau is None:
            return st.curated[-1]
        i = bisect.bisect_right(st.curated, (tau, float("inf"))) - 1
        return st.curated[i] if i >= 0 else None

    def _hot_filter(self, doc_id, st, tr, mode, tau):
        if self.layout == "revision":
            return Store.flt(doc_id, [models.FieldCondition(key="rev_id", match=models.MatchValue(value=tr[1]))] + self._vis())
        if mode == "current" or tau is None:
            cond = [models.FieldCondition(key="valid_to", range=models.Range(gte=TINF))]
        else:
            cond = [models.FieldCondition(key="valid_from", range=models.Range(lte=tau)),
                    models.FieldCondition(key="valid_to", range=models.Range(gt=tau))]
        return Store.flt(doc_id, cond + self._vis())

    def _is_hot(self, st, tr, mode, tau):
        if self.layout == "revision":
            return tr[1] in st.hot
        if mode == "current" or tau is None:
            return True
        return tau >= self._boundary(st)

    def visible(self, doc_id, mode="current", tau=None):
        tr = self.target_rev(doc_id, mode, tau)
        if tr is None:
            return set()
        st = self.docs[doc_id]
        if self._is_hot(st, tr, mode, tau):
            return {p["chunk_hash"] for p in self.store.scroll_payloads(self._hot_filter(doc_id, st, tr, mode, tau))}
        self.counters["cold_reads"] += 1
        return set(self.archive.revision_hashes(doc_id, tr[1]))

    def search(self, doc_id, qvec, k, mode="current", tau=None):
        tr = self.target_rev(doc_id, mode, tau)
        if tr is None:
            return []
        st = self.docs[doc_id]
        if self._is_hot(st, tr, mode, tau):
            return self.store.search(qvec, self._hot_filter(doc_id, st, tr, mode, tau), k)
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
        """Crash recovery: rebuild timelines and the hot index from the Parquet archive only."""
        import time
        t0 = time.perf_counter()
        log = self.archive.read_revision_log().sort_by([("t_proc", "ascending")]).to_pylist()
        pub: Dict[str, Dict[int, dict]] = {}
        for r in log:
            dct = pub.setdefault(r["doc_id"], {})
            if r["status"] == "published":
                dct[r["rev_id"]] = r
            elif r["status"] == "retracted":
                dct.pop(r["rev_id"], None)
        self.store.recreate()
        self.docs, self.closed, self.S, self.chunk_text = {}, [], 0, {}
        for doc, revs in pub.items():
            st = self._doc(doc)
            for r in revs.values():
                st.curated.append((r["t_event"], r["rev_id"]))
                st.meta[r["rev_id"]] = (r["t_event"], len(r["chunk_hashes"]))
                st.sets[r["rev_id"]] = frozenset(r["chunk_hashes"])
            st.curated.sort()
            if self.layout == "revision":
                self._rev_sync_hot(doc, st)
            else:
                self._iv_rebuild(doc, st)
            if self.layout == "revision":
                st.sets = {}
        self.S = 1
        t_ready = time.perf_counter() - t0
        t_green = self.store.wait_green()
        return {"recovery_serving_ready_s": t_ready, "recovery_index_green_s": t_ready + t_green,
                "recovered_points": self.store.count()}

    def memory_objects(self) -> dict:
        return {"pending_revisions": sum(len(st.pending) for st in self.docs.values()),
                "published_revisions": sum(len(st.curated) for st in self.docs.values()),
                "hot_intervals": sum(len(st.intervals) for st in self.docs.values()),
                "hot_revisions": sum(len(st.hot) for st in self.docs.values())}



def make_method(cfg: MethodConfig, embedder: Embedder, store: Store, archive_dir: Optional[Path]) -> Base:
    cls = {"overwrite": SlotMethod, "chunk_lww": SlotMethod, "incremental": IncrementalMethod,
           "doc_guard": DocGuardMethod, "tx_travel": DocGuardMethod, "history": HistoryMethod}[cfg.kind]
    return cls(cfg, embedder, store, archive_dir)


# Named configurations used in the paper (descriptive names, not M1-M4)
def standard_methods(holdback_s: float = 60.0, hot_k: Optional[int] = 2) -> Dict[str, MethodConfig]:
    return {
        "overwrite": MethodConfig("overwrite", "overwrite", archive=False),
        "chunk_lww": MethodConfig("chunk_lww", "chunk_lww", archive=False),
        "incremental": MethodConfig("incremental", "incremental", archive=False),
        "doc_guard": MethodConfig("doc_guard", "doc_guard"),
        "tx_travel": MethodConfig("tx_travel", "tx_travel", archive=False),
        "full_history": MethodConfig("full_history", "history", holdback_s=0.0, hot_k=None, layout="interval"),
        "proposed": MethodConfig("proposed", "history", holdback_s=holdback_s, hot_k=hot_k, layout="interval"),
    }


def ablation_methods(holdback_s: float, hot_k: Optional[int]) -> Dict[str, MethodConfig]:
    """Each mechanism of the proposed design switched off (or replaced) one at a time."""
    return {
        "full_history_per_revision": MethodConfig("full_history_per_revision", "history", holdback_s=0.0,
                                                  hot_k=None, layout="revision"),
        "proposed_no_holdback": MethodConfig("proposed_no_holdback", "history", holdback_s=0.0, hot_k=hot_k),
        "proposed_no_revert": MethodConfig("proposed_no_revert", "history", holdback_s=holdback_s, hot_k=hot_k,
                                           revert_detection=False),
        "proposed_inplace_commit": MethodConfig("proposed_inplace_commit", "history", holdback_s=holdback_s,
                                                hot_k=hot_k, commit="inplace"),
    }
