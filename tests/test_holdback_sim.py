"""Unit tests for the publication replay used by the holdback sweeps (no Qdrant needed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from csrag.holdback_sim import publication_replay
from csrag.types import Delivery, Revision


def _d(rev_id, t_event, h, avail, risky=False):
    r = Revision("doc", rev_id, t_event, h, ("",), ("",), risky)
    return Delivery(r, f"m{rev_id}", False, avail, avail, (avail,))


def _served(log):
    return {h for _, _, h in log if h is not None}


def test_vandal_reverted_within_hold():
    # r1 good, r2 vandal at t=100, r3 restores r1 at t=105; all ready one second after the edit
    dels = [_d(1, 0, "A", 1), _d(2, 100, "B", 101), _d(3, 105, "A", 106)]
    assert 2 in _served(publication_replay(dels, 0.0))                      # no hold: vandal served
    assert 2 not in _served(publication_replay(dels, 600.0))                # hold + revert detection
    assert 2 in _served(publication_replay(dels, 600.0, revert_detection=False))   # hold, no detector
    assert 2 not in _served(publication_replay(dels, 600.0, revert_detection=False, supersede=True))
    # supersede never hides the newest revision
    assert 3 in _served(publication_replay(dels, 600.0, revert_detection=False, supersede=True))


def test_supersede_off_is_unchanged():
    dels = [_d(1, 0, "A", 1), _d(2, 100, "B", 400), _d(3, 105, "C", 106)]
    a = publication_replay(dels, 600.0)
    b = publication_replay(dels, 600.0, supersede=False)
    assert a == b
