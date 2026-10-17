"""Cold archive on Apache Parquet: chunk text + vectors, and a revision log.

* chunks table: one row per (doc_id, chunk_hash), with text and the 384-d vector.
* revisions table: append-only log of revision decisions
  (doc_id, rev_id, t_event, chunk_hashes, status in {published, suppressed, retracted}).

Rows are buffered per bucket (hash of doc_id) and flushed to new part files; readers look
at the in-memory buffer first, then the bucket's part files, using predicate pushdown.
Writers are closed before any read (one file per flush), which also avoids Windows locks.
"""
import hashlib
import shutil
import time
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

CHUNK_SCHEMA = pa.schema([("doc_id", pa.string()), ("chunk_hash", pa.string()), ("text", pa.string()),
                          ("vector", pa.list_(pa.float32(), 384))])
REV_SCHEMA = pa.schema([("doc_id", pa.string()), ("rev_id", pa.int64()), ("t_event", pa.float64()),
                        ("chunk_hashes", pa.list_(pa.string())), ("status", pa.string()),
                        ("t_proc", pa.float64())])


def _bucket(doc_id: str, n: int) -> int:
    return int(hashlib.blake2b(doc_id.encode(), digest_size=4).hexdigest(), 16) % n


class ColdArchive:
    def __init__(self, root: Path, n_buckets: int = 32, flush_rows: int = 4000, fresh: bool = True):
        self.root = Path(root)
        if fresh and self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.n_buckets = n_buckets
        self.flush_rows = flush_rows
        self.chunk_buf: Dict[int, List[tuple]] = {}
        self.rev_buf: Dict[int, List[tuple]] = {}
        self.files: Dict[Tuple[str, int], List[Path]] = {}
        self.seen_chunks: Dict[str, Set[str]] = {}
        self.part = 0
        self.bytes_written = 0
        self.write_seconds = 0.0
        self.read_seconds = 0.0
        self.n_reads = 0

    # ---------- writes ----------
    def add_chunks(self, doc_id: str, hashes: Sequence[str], texts: Sequence[str], vecs: np.ndarray):
        seen = self.seen_chunks.setdefault(doc_id, set())
        b = _bucket(doc_id, self.n_buckets)
        buf = self.chunk_buf.setdefault(b, [])
        for h, t, v in zip(hashes, texts, vecs):
            if h not in seen:
                seen.add(h)
                buf.append((doc_id, h, t, v))
        if len(buf) >= self.flush_rows:
            self._flush_chunks(b)

    def log_revision(self, doc_id: str, rev_id: int, t_event: float, hashes: Sequence[str], status: str, t_proc: float):
        b = _bucket(doc_id, self.n_buckets)
        buf = self.rev_buf.setdefault(b, [])
        buf.append((doc_id, rev_id, t_event, list(hashes), status, t_proc))
        if len(buf) >= self.flush_rows:
            self._flush_revs(b)

    def _write(self, kind: str, b: int, table: pa.Table):
        t0 = time.perf_counter()
        d = self.root / kind / f"b{b:03d}"
        d.mkdir(parents=True, exist_ok=True)
        self.part += 1
        fn = d / f"part-{self.part:06d}.parquet"
        pq.write_table(table, fn, compression="zstd")
        self.files.setdefault((kind, b), []).append(fn)
        self.bytes_written += fn.stat().st_size
        self.write_seconds += time.perf_counter() - t0

    def _flush_chunks(self, b: int):
        buf = self.chunk_buf.get(b)
        if not buf:
            return
        vec = np.stack([r[3] for r in buf]).astype(np.float32)
        arr = pa.FixedSizeListArray.from_arrays(pa.array(vec.reshape(-1), type=pa.float32()), 384)
        tbl = pa.Table.from_arrays([pa.array([r[0] for r in buf]), pa.array([r[1] for r in buf]),
                                    pa.array([r[2] for r in buf]), arr], schema=CHUNK_SCHEMA)
        self._write("chunks", b, tbl)
        self.chunk_buf[b] = []

    def _flush_revs(self, b: int):
        buf = self.rev_buf.get(b)
        if not buf:
            return
        cols = list(zip(*buf))
        tbl = pa.Table.from_arrays([pa.array(cols[0]), pa.array(cols[1], pa.int64()), pa.array(cols[2]),
                                    pa.array(cols[3], pa.list_(pa.string())), pa.array(cols[4]),
                                    pa.array(cols[5])], schema=REV_SCHEMA)
        self._write("revisions", b, tbl)
        self.rev_buf[b] = []

    def flush_all(self):
        for b in list(self.chunk_buf):
            self._flush_chunks(b)
        for b in list(self.rev_buf):
            self._flush_revs(b)

    # ---------- reads ----------
    def revision_hashes(self, doc_id: str, rev_id: int) -> List[str]:
        """Chunk hashes of a revision, read from the revision log."""
        t0 = time.perf_counter()
        b = _bucket(doc_id, self.n_buckets)
        for r in self.rev_buf.get(b, []):
            if r[0] == doc_id and r[1] == rev_id:
                self.read_seconds += time.perf_counter() - t0
                return list(r[3])
        flt = (pc.field("doc_id") == doc_id) & (pc.field("rev_id") == rev_id)
        for fn in self.files.get(("revisions", b), []):
            t = pq.read_table(fn, filters=flt, columns=["chunk_hashes"])
            if t.num_rows:
                self.read_seconds += time.perf_counter() - t0
                return list(t.column("chunk_hashes")[0].as_py())
        self.read_seconds += time.perf_counter() - t0
        return []

    def chunk_vectors(self, doc_id: str, hashes: Sequence[str]) -> Tuple[List[str], List[str], np.ndarray]:
        """Return (hashes, texts, vectors) for the requested chunk hashes of one document."""
        t0 = time.perf_counter()
        self.n_reads += 1
        want = set(hashes)
        found: Dict[str, tuple] = {}
        b = _bucket(doc_id, self.n_buckets)
        for r in self.chunk_buf.get(b, []):
            if r[0] == doc_id and r[1] in want:
                found[r[1]] = (r[2], r[3])
        missing = want - found.keys()
        if missing:
            flt = (pc.field("doc_id") == doc_id) & pc.field("chunk_hash").isin(list(missing))
            for fn in self.files.get(("chunks", b), []):
                t = pq.read_table(fn, filters=flt)
                if t.num_rows:
                    vecs = t.column("vector").combine_chunks().flatten().to_numpy().reshape(-1, 384)
                    for h, txt, v in zip(t.column("chunk_hash").to_pylist(), t.column("text").to_pylist(), vecs):
                        found[h] = (txt, v)
                if len(found) == len(want):
                    break
        hs = [h for h in hashes if h in found]
        self.read_seconds += time.perf_counter() - t0
        if not hs:
            return [], [], np.zeros((0, 384), np.float32)
        return hs, [found[h][0] for h in hs], np.stack([found[h][1] for h in hs]).astype(np.float32)

    def read_revision_log(self) -> pa.Table:
        self.flush_all()
        tables = [pq.read_table(fn) for (k, _), fns in self.files.items() if k == "revisions" for fn in fns]
        return pa.concat_tables(tables) if tables else REV_SCHEMA.empty_table()

    def disk_bytes(self) -> int:
        return sum(f.stat().st_size for f in self.root.rglob("*.parquet"))
