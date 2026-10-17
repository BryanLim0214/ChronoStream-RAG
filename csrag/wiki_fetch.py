"""Fetch real Wikipedia revision histories (full wikitext + metadata + change tags).

Polite by design: strictly serial requests, gzip, a descriptive User-Agent, maxlag,
Retry-After / exponential backoff, and a resumable on-disk cache (one gzipped JSONL
file per article plus a `.done` marker). Nothing is fetched twice.

Usage:
    python -m csrag.wiki_fetch --config configs/benchB_fetch.yaml [--pilot 20]
"""
import argparse
import gzip
import json
import random
import time
from pathlib import Path

import requests
import yaml

from csrag import paths


class Api:
    def __init__(self, cfg):
        self.cfg = cfg
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": cfg["user_agent"], "Accept-Encoding": "gzip"})
        self.n_requests = 0
        self.n_retries = 0

    def get(self, params):
        params = dict(params, format="json", formatversion="2", maxlag=self.cfg["maxlag"])
        delay = 2.0
        for attempt in range(12):
            try:
                r = self.s.get(self.cfg["api"], params=params, timeout=120)
                self.n_requests += 1
                if r.status_code == 429 or r.status_code >= 500:
                    wait = float(r.headers.get("Retry-After", delay))
                    self.n_retries += 1
                    time.sleep(wait)
                    delay = min(delay * 2, 300)
                    continue
                data = r.json()
                if "error" in data:
                    if data["error"].get("code") == "maxlag":
                        self.n_retries += 1
                        time.sleep(float(r.headers.get("Retry-After", 5)))
                        continue
                    raise RuntimeError(f"API error: {data['error']}")
                time.sleep(self.cfg["polite_delay_sec"])
                return data
            except (requests.ConnectionError, requests.Timeout, json.JSONDecodeError):
                self.n_retries += 1
                time.sleep(delay)
                delay = min(delay * 2, 300)
        raise RuntimeError("API request failed after retries")


def vital_l4_titles(api, topic_category):
    """Level-4 vital articles, read from the per-topic tracking categories.

    The categories hold the articles' talk pages (Talk:X); we map them back to X.
    (The Wikipedia:Vital articles/Level/4/* list pages expose no page links via the API.)
    """
    subcats, cont = [], {}
    while True:
        d = api.get({"action": "query", "list": "categorymembers", "cmtitle": topic_category,
                     "cmtype": "subcat", "cmlimit": "max", **cont})
        subcats += [m["title"] for m in d["query"]["categorymembers"]]
        if "continue" not in d:
            break
        cont = d["continue"]
    titles = []
    for cat in subcats:
        cont = {}
        while True:
            d = api.get({"action": "query", "list": "categorymembers", "cmtitle": cat,
                         "cmnamespace": "0|1", "cmlimit": "max", **cont})
            for m in d["query"]["categorymembers"]:
                t = m["title"]
                titles.append(t[5:] if t.startswith("Talk:") else t)
            if "continue" not in d:
                break
            cont = d["continue"]
    seen, out = set(), []
    for t in sorted(titles):
        if t not in seen:
            seen.add(t)
            out.append(t)
    return subcats, out


def page_info(api, title):
    d = api.get({"action": "query", "titles": title, "prop": "info|pageprops",
                 "ppprop": "disambiguation"})
    return d["query"]["pages"][0]


def safe_name(title):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in title)[:120]


def fetch_article(api, cfg, title, out_dir):
    """Fetch the most recent `max_revisions_per_article` revisions inside the window."""
    fn = out_dir / f"{safe_name(title)}.jsonl.gz"
    done = out_dir / f"{safe_name(title)}.done"
    if done.exists():
        return json.loads(done.read_text(encoding="utf-8"))
    revs, cont = [], {}
    rvprop = "ids|timestamp|flags|comment|user|userid|size|sha1|tags|content"
    while len(revs) < cfg["max_revisions_per_article"]:
        d = api.get({"action": "query", "titles": title, "prop": "revisions",
                     "rvprop": rvprop, "rvslots": "main", "rvlimit": 50,
                     "rvdir": "older", "rvstart": cfg["window_end"], "rvend": cfg["window_start"],
                     **cont})
        page = d["query"]["pages"][0]
        for r in page.get("revisions", []):
            slot = r.get("slots", {}).get("main", {})
            revs.append({
                "revid": r["revid"], "parentid": r.get("parentid"),
                "timestamp": r["timestamp"], "user": r.get("user"), "userid": r.get("userid"),
                "anon": bool(r.get("anon", False)), "temp": bool(r.get("temp", False)),
                "minor": bool(r.get("minor", False)),
                "comment": r.get("comment"), "size": r.get("size"), "sha1": r.get("sha1"),
                "sha1hidden": bool(r.get("sha1hidden", False)),
                "texthidden": bool(slot.get("texthidden", False)),
                "tags": r.get("tags", []),
                "content": slot.get("content"),
            })
        if "continue" not in d:
            break
        cont = d["continue"]
    revs = revs[: cfg["max_revisions_per_article"]]
    revs.sort(key=lambda r: (r["timestamp"], r["revid"]))
    raw_bytes = sum(len((r["content"] or "").encode("utf-8")) for r in revs)
    with gzip.open(fn, "wt", encoding="utf-8") as f:
        for r in revs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {"title": title, "file": fn.name, "n_revisions": len(revs), "raw_bytes": raw_bytes,
            "first": revs[0]["timestamp"] if revs else None, "last": revs[-1]["timestamp"] if revs else None}
    done.write_text(json.dumps(meta), encoding="utf-8")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(paths.CONFIGS / "benchB_fetch.yaml"))
    ap.add_argument("--pilot", type=int, default=0, help="fetch only the first N eligible articles")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    out_dir = paths.WIKI_CACHE / "articles"
    out_dir.mkdir(parents=True, exist_ok=True)
    api = Api(cfg)

    order_file = paths.WIKI_CACHE / "article_order.json"
    if order_file.exists():
        order = json.loads(order_file.read_text(encoding="utf-8"))
    else:
        subpages, vital = vital_l4_titles(api, cfg["vital_category"])
        rng = random.Random(cfg["seed"])
        rng.shuffle(vital)
        legacy = []
        if cfg.get("include_legacy_45"):
            legacy = list(json.load(open(paths.DATA / "raw" / "wikimedia_real_revisions.json",
                                         encoding="utf-8"))["articles"].keys())
        seen, titles = set(), []
        for t in legacy + vital:
            if t not in seen:
                seen.add(t)
                titles.append(t)
        order = {"retrieved_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "seed": cfg["seed"], "vital_subpages": subpages, "n_vital": len(vital),
                 "n_legacy": len(legacy), "titles": titles}
        order_file.write_text(json.dumps(order, ensure_ascii=False, indent=1), encoding="utf-8")

    log_file = paths.WIKI_CACHE / "fetch_log.jsonl"
    total_bytes, n_ok = 0, 0
    for title in order["titles"]:
        if total_bytes >= cfg["target_raw_wikitext_bytes"]:
            break
        if args.pilot and n_ok >= args.pilot:
            break
        done = out_dir / f"{safe_name(title)}.done"
        if done.exists():
            meta = json.loads(done.read_text(encoding="utf-8"))
            if meta.get("n_revisions", 0) >= cfg["min_revisions_in_window"]:
                total_bytes += meta["raw_bytes"]
                n_ok += 1
            continue
        skip = out_dir / f"{safe_name(title)}.skip"
        if skip.exists():
            continue
        reason = None
        if any(title.startswith(p) for p in cfg["exclude_title_prefixes"]):
            reason = "title_prefix"
        else:
            info = page_info(api, title)
            if info.get("missing"):
                reason = "missing"
            elif cfg["exclude_redirects"] and info.get("redirect"):
                reason = "redirect"
            elif cfg["exclude_disambiguation"] and "disambiguation" in info.get("pageprops", {}):
                reason = "disambiguation"
            elif info.get("length", 0) > cfg["max_current_length_bytes"]:
                reason = "too_long"
        if reason:
            skip.write_text(reason, encoding="utf-8")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps({"title": title, "skipped": reason}) + "\n")
            continue
        meta = fetch_article(api, cfg, title, out_dir)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({**meta, "requests": api.n_requests, "retries": api.n_retries,
                                "cum_bytes": total_bytes + meta["raw_bytes"]}) + "\n")
        if meta["n_revisions"] >= cfg["min_revisions_in_window"]:
            total_bytes += meta["raw_bytes"]
            n_ok += 1
        print(f"{n_ok:5d} {title[:50]:50s} revs={meta['n_revisions']:4d} "
              f"MB={meta['raw_bytes']/1e6:8.1f} cum_GB={total_bytes/1e9:6.2f} req={api.n_requests}",
              flush=True)
    print(f"DONE articles={n_ok} raw_GB={total_bytes/1e9:.3f} requests={api.n_requests} retries={api.n_retries}")


if __name__ == "__main__":
    main()
