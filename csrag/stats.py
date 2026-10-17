"""Cluster statistics: the unit of analysis is a document (timeline or article), never a seed.

* cluster_mean_ci: mean of per-cluster means with a percentile bootstrap over clusters.
* paired_diff_ci: difference between two methods on per-cluster means (paired bootstrap).
* wilcoxon_holm: Wilcoxon signed-rank on per-cluster differences, Holm-corrected.
"""
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats as st


def per_cluster(values: Sequence[float], clusters: Sequence[str]) -> Dict[str, float]:
    acc: Dict[str, List[float]] = {}
    for v, c in zip(values, clusters):
        acc.setdefault(c, []).append(float(v))
    return {c: float(np.mean(v)) for c, v in acc.items()}


def cluster_mean_ci(values, clusters, n_boot: int = 10000, seed: int = 0, alpha: float = 0.05) -> Tuple[float, float, float]:
    pc = per_cluster(values, clusters)
    x = np.array(list(pc.values()))
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = x[idx].mean(axis=1)
    return float(x.mean()), float(np.percentile(boots, 100 * alpha / 2)), float(np.percentile(boots, 100 * (1 - alpha / 2)))


def paired_diff_ci(a: Dict[str, float], b: Dict[str, float], n_boot: int = 10000, seed: int = 0,
                   alpha: float = 0.05) -> Tuple[float, float, float, int]:
    keys = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in keys])
    if len(d) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    return (float(d.mean()), float(np.percentile(boots, 100 * alpha / 2)),
            float(np.percentile(boots, 100 * (1 - alpha / 2))), int(np.sum(d != 0)))


def wilcoxon_holm(pairs: Dict[str, Tuple[Dict[str, float], Dict[str, float]]]) -> Dict[str, dict]:
    raw = {}
    for name, (a, b) in pairs.items():
        keys = sorted(set(a) & set(b))
        d = np.array([a[k] - b[k] for k in keys])
        if np.all(d == 0) or len(d) < 2:
            raw[name] = (1.0, len(d), int(np.sum(d != 0)))
            continue
        p = st.wilcoxon(d, zero_method="zsplit").pvalue
        raw[name] = (float(p), len(d), int(np.sum(d != 0)))
    order = sorted(raw, key=lambda k: raw[k][0])
    m = len(order)
    out, running = {}, 0.0
    for i, k in enumerate(order):
        adj = min(1.0, (m - i) * raw[k][0])
        running = max(running, adj)
        out[k] = {"p_raw": raw[k][0], "p_holm": running, "n_units": raw[k][1], "n_nonzero": raw[k][2]}
    return out
