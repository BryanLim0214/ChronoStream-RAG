"""Paper figures from the result files listed in FIGURE_CONTRACT. A missing input is an error.

python experiments/make_figures.py --out ../paper/figures
Style: IEEE column 3.5 in / page 7.16 in, 8 pt Times-like text, TrueType-embedded vector PDF,
categorical slots in fixed order (validated palette), markers + direct labels on every series,
no in-plot titles, no dual axes, recessive grid.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from csrag import paths

COL_W, PAGE_W = 3.5, 7.16
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
INK, INK2, GRID = "#1b1b1b", "#52514e", "#e2e1dc"
MARK = ["o", "s", "^", "D", "v", "P", "X"]


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "lines.linewidth": 1.4, "lines.markersize": 3.6, "legend.frameon": False})


def need(p: Path) -> Path:
    if not p.exists():
        raise FileNotFoundError(f"figure input missing: {p}")
    return p


def fig_holdback(exp: str, out: Path):
    """Exposure to later-reverted edits vs the delay that holdback adds to every other edit."""
    t = pd.read_csv(need(paths.RESULTS / exp / "tradeoff.csv"))
    g = t.groupby(["risk_gated", "holdback_s"]).agg(served=("reverted_ever_served", "mean"),
                                                     total=("reverted_revisions", "mean"),
                                                     delay=("lag_added_mean_s", "mean")).reset_index()
    g["share"] = 100 * g.served / g.total
    base = g[(g.holdback_s == 0)].iloc[0]
    fig, ax = plt.subplots(figsize=(COL_W, 2.1))
    labels = {600: "10 min", 3600: "1 h", 86400: "1 day"}
    series = [(False, "Every edit waits $H$", SLOT[0], MARK[0], (-6, -11)),
              (True, "Only edits by anonymous or\ntemporary accounts wait $H$", SLOT[1], MARK[1], (4, 3))]
    ax.axhline(base.share, color=INK2, lw=0.8, ls=(0, (3, 2)))
    ax.text(0.12, base.share + 2.0, "no holdback", fontsize=6.5, color=INK2)
    for gated, name, col, mk, off in series:
        s = g[(g.risk_gated == gated) & (g.holdback_s >= 60)].sort_values("holdback_s")
        x, y = s.delay.values / 60.0, s.share.values
        keep = x >= 0.1
        ax.plot(x[keep], y[keep], color=col, marker=mk, label=name)
        for H, xi, yi in zip(s.holdback_s.values[keep], x[keep], y[keep]):
            if H in labels:
                ax.annotate(labels[H], (xi, yi), textcoords="offset points", xytext=off, fontsize=6.5, color=INK2)
    ax.set_xscale("log")
    ax.set_xlim(0.1, 3000)
    ax.set_xlabel("Mean delay added to edits that are not reverted (minutes, log scale)")
    ax.set_ylabel("Later-reverted edits ever\nserved as current (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower left", handlelength=1.6)
    fig.savefig(out / "fig_holdback.pdf")
    plt.close(fig)


FIGURE_CONTRACT = {"fig_holdback.pdf": ("B2 holdback tradeoff.csv", fig_holdback)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(paths.ROOT.parent / "paper" / "figures"))
    ap.add_argument("--holdback_exp", default="B2_holdback")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    style()
    fig_holdback(a.holdback_exp, out)
    print("figures written to", out)
