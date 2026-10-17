"""Central path configuration. No absolute paths are hard-coded anywhere else.

CSRAG_ROOT  -> repository root (defaults to the parent of this package)
CSRAG_CACHE -> large artifacts: fetched Wikipedia histories, Parquet archives, model weights
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("CSRAG_ROOT", Path(__file__).resolve().parent.parent))
CACHE = Path(os.environ.get("CSRAG_CACHE", ROOT.parent / "cache"))
DATA = ROOT / "data"
RESULTS = ROOT / "results_v2"
CONFIGS = ROOT / "configs"
MODELS = Path(os.environ.get("CSRAG_MODELS", CACHE / "models"))
WIKI_CACHE = CACHE / "wiki_history"
BENCHB = CACHE / "benchB"
ARCHIVES = CACHE / "archives"

for _p in (CACHE, RESULTS, MODELS, WIKI_CACHE, BENCHB, ARCHIVES):
    _p.mkdir(parents=True, exist_ok=True)
