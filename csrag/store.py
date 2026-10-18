"""Thin wrapper around a Qdrant server collection (real HNSW index, payload indexes).

Writes are sent without waiting (wait=False); Qdrant applies them in arrival order through
its write-ahead log. Before any read that must observe all earlier writes, `barrier()`
sends one write with wait=True, which returns only after every earlier write is applied.
"""
import hashlib
import re
import time
import urllib.request
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from qdrant_client import QdrantClient, models

import os

QDRANT_HOST = "127.0.0.1"
# One Qdrant server per worker when policies run in parallel (separate containers and ports).
QDRANT_HTTP = int(os.environ.get("CSRAG_QDRANT_HTTP", "6333"))
QDRANT_GRPC = int(os.environ.get("CSRAG_QDRANT_GRPC", "6334"))
QDRANT_CONTAINER = os.environ.get("CSRAG_QDRANT_CONTAINER", "csrag-qdrant")


def point_id(key: str) -> int:
    """Deterministic 60-bit integer id."""
    return int(hashlib.blake2b(key.encode("utf-8"), digest_size=8).hexdigest(), 16) >> 4


class Store:
    def __init__(self, name: str, dim: int = 384, hnsw_m: int = 16, ef_construct: int = 100,
                 search_ef: int = 128, store_text: bool = True, wait_writes: bool = False):
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_HTTP, grpc_port=QDRANT_GRPC,
                                   prefer_grpc=True, timeout=600)
        self.rest = QdrantClient(host=QDRANT_HOST, port=QDRANT_HTTP, prefer_grpc=False, timeout=600)
        self.n_read_retries = 0
        self.name = name
        self.dim = dim
        self.store_text = store_text
        self.wait = wait_writes
        self.dirty = False
        self.search_params = models.SearchParams(hnsw_ef=search_ef)
        self.hnsw = models.HnswConfigDiff(m=hnsw_m, ef_construct=ef_construct)
        self.n_upserted = 0
        self.n_deleted = 0
        self.n_payload_updates = 0
        self.write_seconds = 0.0
        self.recreate()

    def recreate(self):
        if self.client.collection_exists(self.name):
            self.client.delete_collection(self.name)
        self.client.create_collection(
            self.name,
            vectors_config=models.VectorParams(size=self.dim, distance=models.Distance.COSINE),
            hnsw_config=self.hnsw)
        for field, schema in [("doc_id", models.PayloadSchemaType.KEYWORD),
                              ("chunk_hash", models.PayloadSchemaType.KEYWORD),
                              ("rev_id", models.PayloadSchemaType.INTEGER),
                              ("slot", models.PayloadSchemaType.INTEGER),
                              ("tx_from", models.PayloadSchemaType.INTEGER),
                              ("tx_to", models.PayloadSchemaType.INTEGER),
                              ("valid_from", models.PayloadSchemaType.FLOAT),
                              ("valid_to", models.PayloadSchemaType.FLOAT)]:
            self.client.create_payload_index(self.name, field, field_schema=schema, wait=False)
        self.dirty = True   # the first barrier waits for the index builds

    def drop(self):
        if self.client.collection_exists(self.name):
            self.client.delete_collection(self.name)

    # ---------- writes ----------
    def upsert(self, ids: Sequence[int], vectors: np.ndarray, payloads: Sequence[dict], batch: int = 512):
        if not len(ids):
            return
        t0 = time.perf_counter()
        for i in range(0, len(ids), batch):
            pts = models.Batch(ids=list(ids[i:i + batch]), vectors=vectors[i:i + batch].tolist(),
                               payloads=list(payloads[i:i + batch]))
            self.client.upsert(self.name, points=pts, wait=self.wait)
        self.dirty = True
        self.n_upserted += len(ids)
        self.write_seconds += time.perf_counter() - t0

    def delete(self, ids: Sequence[int]):
        if not ids:
            return
        t0 = time.perf_counter()
        self.client.delete(self.name, points_selector=models.PointIdsList(points=list(ids)), wait=self.wait)
        self.dirty = True
        self.n_deleted += len(ids)
        self.write_seconds += time.perf_counter() - t0

    def set_payload(self, ids: Sequence[int], payload: dict):
        if not ids:
            return
        t0 = time.perf_counter()
        self.client.set_payload(self.name, payload=payload, points=list(ids), wait=self.wait)
        self.dirty = True
        self.n_payload_updates += len(ids)
        self.write_seconds += time.perf_counter() - t0

    def barrier(self):
        """Block until every earlier write is applied (one tiny write with wait=True)."""
        if not self.dirty:
            return
        t0 = time.perf_counter()
        self.client.upsert(self.name, points=models.Batch(ids=[0], vectors=[[1.0] + [0.0] * (self.dim - 1)],
                                                         payloads=[{"doc_id": "__barrier__"}]), wait=True)
        self.write_seconds += time.perf_counter() - t0
        self.dirty = False

    # ---------- reads ----------
    @staticmethod
    def flt(doc_id: Optional[str] = None, extra: Optional[List[models.Condition]] = None) -> models.Filter:
        must = []
        if doc_id is not None:
            must.append(models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)))
        if extra:
            must.extend(extra)
        return models.Filter(must=must)

    def _read(self, fn, *args, **kw):
        """gRPC read with retries, then a REST fallback (guards against transient gRPC faults)."""
        for attempt in range(3):
            try:
                return fn(self.client, *args, **kw)
            except Exception:
                self.n_read_retries += 1
                time.sleep(0.2 * (attempt + 1))
        return fn(self.rest, *args, **kw)

    def scroll_payloads(self, flt: models.Filter, fields: Sequence[str] = ("chunk_hash",), limit: int = 2048) -> List[dict]:
        self.barrier()
        out, offset = [], None
        while True:
            pts, offset = self._read(lambda c, **kw: c.scroll(**kw), collection_name=self.name, scroll_filter=flt, limit=limit, offset=offset,
                                        with_payload=list(fields), with_vectors=False)
            out.extend(p.payload for p in pts)
            if offset is None:
                break
        return out

    def search(self, vector: np.ndarray, flt: models.Filter, k: int,
               fields: Sequence[str] = ("chunk_hash", "text", "rev_id")) -> List[dict]:
        self.barrier()
        res = self._read(lambda c, **kw: c.query_points(**kw), collection_name=self.name, query=vector.tolist(),
                         query_filter=flt, limit=k, search_params=self.search_params, with_payload=list(fields))
        return [dict(p.payload, score=p.score) for p in res.points]

    def count(self) -> int:
        """Number of real points (excludes the barrier sentinel)."""
        self.barrier()
        n = self.client.count(self.name, exact=True).count
        sentinel = self.client.count(self.name, count_filter=Store.flt("__barrier__"), exact=True).count
        return n - sentinel

    def count_raw(self) -> int:
        return self.client.count(self.name, exact=True).count

    def info(self) -> dict:
        self.barrier()
        ci = self.client.get_collection(self.name)
        return {"points": ci.points_count, "indexed_vectors": ci.indexed_vectors_count,
                "segments": ci.segments_count, "status": str(ci.status)}

    def wait_green(self, timeout_s: float = 3600) -> float:
        self.barrier()
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout_s:
            if str(self.client.get_collection(self.name).status).lower().endswith("green"):
                break
            time.sleep(0.2)
        return time.perf_counter() - t0


def qdrant_memory() -> Dict[str, float]:
    """Qdrant process memory from its Prometheus endpoint (bytes)."""
    try:
        txt = urllib.request.urlopen(f"http://{QDRANT_HOST}:{QDRANT_HTTP}/metrics", timeout=5).read().decode()
    except Exception:
        return {}
    out = {}
    for line in txt.splitlines():
        m = re.match(r"^(memory_\w+_bytes)\s+([0-9.eE+]+)$", line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out
