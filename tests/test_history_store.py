"""Gate tests G6/G7 against the live Qdrant server: the valid-time history must expose exactly
the right revision for current and as-of reads, under any arrival order, with reverts,
for both storage layouts, any hot window, and after crash recovery.

Requires the Qdrant server (docker container csrag-qdrant) to be running.
"""
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from csrag import paths
from csrag.methods import MethodConfig, make_method
from csrag.revert import detect_reverts
from csrag.store import Store
from csrag.types import Delivery, Revision


class FakeEmbedder:
    """Deterministic vectors so the test does not need the GPU model."""
    reuse = True

    def __init__(self):
        self.cache = {}

    def embed(self, texts, hashes):
        out = []
        for h in hashes:
            if h not in self.cache:
                rng = np.random.default_rng(abs(hash(h)) % (2 ** 32))
                v = rng.normal(size=384).astype(np.float32)
                self.cache[h] = v / np.linalg.norm(v)
            out.append(self.cache[h])
        return np.stack(out)

    def stats(self):
        return {}


def make_doc(rng, n_rev, doc="d0"):
    """Random revisions over a small paragraph alphabet, with some identity reverts."""
    base = [f"p{i}" for i in range(6)]
    revs, contents = [], []
    cur = list(base)
    for i in range(n_rev):
        r = rng.random()
        if r < 0.15 and len(contents) >= 2:
            cur = list(contents[-2])                         # identity revert of the last edit
        elif r < 0.35 and len(cur) > 2:
            cur = cur[:-rng.randint(1, 2)]                   # shrink
        else:
            cur = list(cur)
            j = rng.randrange(len(cur))
            cur[j] = f"q{i}_{j}"                             # edit one paragraph
            if rng.random() < 0.3:
                cur.append(f"n{i}")
        contents.append(tuple(cur))
        ch = tuple(f"{doc}:{c}" for c in cur)
        revs.append(Revision(doc, 1000 + i, 100.0 * (i + 1), "|".join(cur), ch, ch))
    return revs


def expected_sets(revs):
    rp, _ = detect_reverts([r.content_hash for r in revs])
    cur = [r for i, r in enumerate(revs) if i not in rp]
    return cur


@pytest.mark.parametrize("layout,hot_k", [("interval", None), ("interval", 2), ("interval", 1),
                                           ("revision", None), ("revision", 2)])
def test_history_reads_are_exact_under_any_order(layout, hot_k, tmp_path):
    rng = random.Random(11)
    for trial in range(4):
        revs = make_doc(rng, rng.randint(6, 14))
        order = list(range(len(revs)))
        rng.shuffle(order)
        cfg = MethodConfig(f"t_{layout}_{hot_k}", "history", holdback_s=0.0, hot_k=hot_k, layout=layout)
        store = Store(f"test_{layout}_{hot_k}_{trial}")
        m = make_method(cfg, FakeEmbedder(), store, tmp_path / f"arch_{trial}")
        for k, i in enumerate(order):
            r = revs[i]
            m.on_delivery(Delivery(r, f"m{i}", False, 0, 1e5 + k, tuple(1e5 + k for _ in r.chunks)), 1e5 + k, None)
        cur = expected_sets(revs)
        assert m.visible("d0", "current") == set(cur[-1].chunk_hashes)
        for r in cur:
            assert m.visible("d0", "asof", r.t_event + 1.0) == set(r.chunk_hashes), (layout, hot_k, trial)
        # crash recovery restores identical reads
        m.finish()
        m.recover()
        assert m.visible("d0", "current") == set(cur[-1].chunk_hashes)
        for r in cur:
            assert m.visible("d0", "asof", r.t_event + 1.0) == set(r.chunk_hashes)
        store.drop()


def test_interval_layout_stores_far_fewer_points_than_revision_layout(tmp_path):
    rng = random.Random(3)
    revs = make_doc(rng, 30)
    counts = {}
    for layout in ("interval", "revision"):
        cfg = MethodConfig(f"c_{layout}", "history", hot_k=None, layout=layout)
        store = Store(f"count_{layout}")
        m = make_method(cfg, FakeEmbedder(), store, tmp_path / layout)
        for k, r in enumerate(revs):
            m.on_delivery(Delivery(r, f"m{k}", False, 0, 1e5 + k, tuple(1e5 + k for _ in r.chunks)), 1e5 + k, None)
        counts[layout] = store.count()
        store.drop()
    assert counts["interval"] < counts["revision"]
