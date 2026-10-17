"""Gate tests G5 (revert detection, holdback) and data-leakage guards. Run: pytest -q tests"""
import itertools
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from csrag.revert import RevertTracker, detect_reverts
from csrag import paths


def test_identity_revert_basic():
    assert detect_reverts(list("ABA"))[0] == {1}
    assert detect_reverts(list("ABCA"))[0] == {1, 2}            # A->B->C->A marks B and C
    assert detect_reverts(list("AAB"))[0] == set()               # null edit is not a revert
    assert detect_reverts(list("ABCB"))[0] == {2}               # restores B: C reverted


def test_radius_limit():
    seq = ["A"] + [f"x{i}" for i in range(16)] + ["A"]          # restore 17 revisions back
    assert detect_reverts(seq, radius=15)[0] == set()
    seq = ["A"] + [f"x{i}" for i in range(14)] + ["A"]
    assert detect_reverts(seq, radius=15)[0] == set(range(1, 15))


def test_online_tracker_matches_offline_under_any_arrival_order():
    rng = random.Random(7)
    for trial in range(200):
        n = rng.randint(2, 12)
        hashes = [rng.choice("ABCD") for _ in range(n)]
        offline = detect_reverts(hashes)[0]
        order = list(range(n))
        rng.shuffle(order)
        tr = RevertTracker()
        for i in order:
            tr.add("d", i, float(i), hashes[i])
        assert tr.reverted_ids("d") == offline, (hashes, order)


def test_benchA_has_no_copies_and_detector_matches_labels():
    gt = json.load(open(paths.DATA / "benchA" / "gt.json", encoding="utf-8"))
    assert len(gt) == 20
    for doc, g in gt.items():
        revs = g["revisions"]
        rev_pos = detect_reverts([r["content_hash"] for r in revs])[0]
        assert rev_pos == {i for i, r in enumerate(revs) if r["is_vandal"]}


def test_methods_never_read_ground_truth_fields():
    import re
    src = (paths.ROOT / "csrag" / "methods.py").read_text(encoding="utf-8")
    for banned in [r"is_vandal", r"reverted_tag", r"\.fact", r"summary_note", r"is_revert", r"gt\."]:
        assert not re.search(banned, src), banned


def test_no_modeled_costs_left_in_new_code():
    for f in (paths.ROOT / "csrag").glob("*.py"):
        s = f.read_text(encoding="utf-8")
        for banned in ["3.4269", "0.00054", "* 0.012", "0.45 if", "+ 12.5", "scale_factor"]:
            assert banned not in s, (f.name, banned)


def test_cluster_bootstrap_known_answer():
    import numpy as np
    from csrag.stats import cluster_mean_ci, paired_diff_ci
    rng = np.random.default_rng(0)
    clusters = [f"c{i}" for i in range(200) for _ in range(5)]
    values = [1.0 if int(c[1:]) % 2 == 0 else 0.0 for c in clusters]   # true cluster mean = 0.5
    m, lo, hi = cluster_mean_ci(values, clusters, n_boot=4000)
    assert abs(m - 0.5) < 1e-9 and lo < 0.5 < hi and hi - lo < 0.2
    a = {f"c{i}": 1.0 for i in range(30)}
    b = {f"c{i}": 1.0 for i in range(30)}
    d, lo, hi, nz = paired_diff_ci(a, b)
    assert d == 0 and lo == 0 and hi == 0 and nz == 0
