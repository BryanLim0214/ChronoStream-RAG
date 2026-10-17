"""Real embedding with an optional content-hash reuse cache.

Every call to `embed` runs the sentence encoder on the texts that are not reused; there
is no global pre-warm and no modeled time. GPU work is synchronized before timing.
"""
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from csrag import paths

_MODEL = None


def load_encoder(device: Optional[str] = None) -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        _MODEL = SentenceTransformer(str(paths.MODELS / "embedder"), device=device)
    return _MODEL


class Embedder:
    """Encoder + per-method reuse cache keyed by chunk hash (bounded LRU)."""

    def __init__(self, reuse: bool = True, capacity: int = 5_000_000, device: Optional[str] = None,
                 batch_size: int = 256):
        self.model = load_encoder(device)
        self.device = str(self.model.device)
        self.reuse = reuse
        self.capacity = capacity
        self.batch_size = batch_size
        self.cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self.n_requested = 0
        self.n_encoded = 0
        self.n_reused = 0
        self.encode_seconds = 0.0

    def _encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        v = self.model.encode(texts, batch_size=self.batch_size, convert_to_numpy=True,
                              normalize_embeddings=True, show_progress_bar=False)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        self.encode_seconds += time.perf_counter() - t0
        self.n_encoded += len(texts)
        return v.astype(np.float32, copy=False)

    def embed(self, texts: Sequence[str], hashes: Sequence[str]) -> np.ndarray:
        """Return one vector per text. Reused vectors come from the cache when reuse is on."""
        self.n_requested += len(texts)
        out = np.empty((len(texts), 384), dtype=np.float32)
        if not self.reuse:
            out[:] = self._encode(list(texts))
            return out
        miss_idx: List[int] = []
        miss_txt: List[str] = []
        first_pos: Dict[str, int] = {}
        for i, (t, h) in enumerate(zip(texts, hashes)):
            v = self.cache.get(h)
            if v is not None:
                self.cache.move_to_end(h)
                out[i] = v
                self.n_reused += 1
            elif h in first_pos:            # same chunk twice in one call: encode once
                miss_idx.append(i)
                self.n_reused += 1
            else:
                first_pos[h] = len(miss_txt)
                miss_idx.append(i)
                miss_txt.append(t)
        if miss_txt:
            vecs = self._encode(miss_txt)
            for i in miss_idx:
                h = hashes[i]
                out[i] = vecs[first_pos[h]]
            for h, p in first_pos.items():
                self.cache[h] = vecs[p]
            while len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
        return out

    def query(self, texts: Sequence[str]) -> np.ndarray:
        """Embed questions (not counted as chunk work, not cached)."""
        return self.model.encode(list(texts), batch_size=self.batch_size, convert_to_numpy=True,
                                 normalize_embeddings=True, show_progress_bar=False).astype(np.float32)

    def stats(self) -> dict:
        return {"chunks_requested": self.n_requested, "chunks_encoded": self.n_encoded,
                "chunks_reused": self.n_reused, "encode_seconds": round(self.encode_seconds, 4),
                "reuse_pct": 100.0 * self.n_reused / max(self.n_requested, 1), "device": self.device}
