"""Measured resource use: Python process memory and Qdrant server memory, sampled in a thread."""
import re
import subprocess
import threading
import time

import psutil

from csrag.store import qdrant_memory


class ResourceSampler:
    def __init__(self, py_interval: float = 0.1, qdrant_interval: float = 1.0, docker_interval: float = 0.0,
                 container: str = "csrag-qdrant"):
        self.proc = psutil.Process()
        self.py_interval = py_interval
        self.q_interval = qdrant_interval
        self.peak_rss = 0
        self.peak_qdrant = {}
        self.docker_interval = docker_interval
        self.container = container
        self.peak_docker_bytes = 0.0
        self.samples = []
        self._stop = threading.Event()
        self._t = None

    def _run(self):
        last_q = last_d = 0.0
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
            if self.docker_interval and now - last_d >= self.docker_interval:
                last_d = now
                b = docker_mem_bytes(self.container)
                if b:
                    self.peak_docker_bytes = max(self.peak_docker_bytes, b)
            self._stop.wait(self.py_interval)

    def __enter__(self):
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()

    def summary(self) -> dict:
        out = {"py_rss_peak_bytes": self.peak_rss, "docker_mem_peak_bytes": self.peak_docker_bytes}
        for k, v in self.peak_qdrant.items():
            out[f"qdrant_{k}_peak"] = v
        return out


_UNITS = {"B": 1, "KiB": 2**10, "MiB": 2**20, "GiB": 2**30, "KB": 1e3, "MB": 1e6, "GB": 1e9}


def docker_mem_bytes(container: str) -> float:
    """Container memory use as reported by `docker stats` (cgroup usage, includes page cache)."""
    try:
        out = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", container],
                             capture_output=True, text=True, timeout=20,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
        m = re.match(r"\s*([0-9.]+)\s*([KMG]i?B|B)", out)
        return float(m.group(1)) * _UNITS[m.group(2)] if m else 0.0
    except Exception:
        return 0.0
