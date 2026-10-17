"""Measured resource use: Python process memory and Qdrant server memory, sampled in a thread."""
import threading
import time

import psutil

from csrag.store import qdrant_memory


class ResourceSampler:
    def __init__(self, py_interval: float = 0.1, qdrant_interval: float = 1.0):
        self.proc = psutil.Process()
        self.py_interval = py_interval
        self.q_interval = qdrant_interval
        self.peak_rss = 0
        self.peak_qdrant = {}
        self.samples = []
        self._stop = threading.Event()
        self._t = None

    def _run(self):
        last_q = 0.0
        while not self._stop.is_set():
            rss = self.proc.memory_info().rss
            self.peak_rss = max(self.peak_rss, rss)
            now = time.perf_counter()
            if now - last_q >= self.q_interval:
                last_q = now
                q = qdrant_memory()
                for k, v in q.items():
                    self.peak_qdrant[k] = max(self.peak_qdrant.get(k, 0.0), v)
                self.samples.append((now, rss, q.get("memory_resident_bytes"), q.get("memory_allocated_bytes")))
            self._stop.wait(self.py_interval)

    def __enter__(self):
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()

    def summary(self) -> dict:
        out = {"py_rss_peak_bytes": self.peak_rss}
        for k, v in self.peak_qdrant.items():
            out[f"qdrant_{k}_peak"] = v
        return out
