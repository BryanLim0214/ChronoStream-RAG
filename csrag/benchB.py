"""Benchmark B preparation: real Wikipedia revision histories -> chunked revisions + ground truth.

For every fetched article (cache/wiki_history/articles/*.jsonl.gz):
  * revisions whose text or hash was hidden by administrators are dropped (counted);
  * wikitext -> plain text per section (mwparserfromhell), paragraphs packed into chunks
    of about `target_chars` characters; a new chunk always starts at a section heading;
  * each chunk gets a 128-bit BLAKE2b hash; the revision's content hash is the API SHA-1;
  * ground truth: identity reverts over the API SHA-1 (radius 15, offline, full fetched
    history) and, independently, MediaWiki's own `mw-reverted` change tag.

Outputs cache/benchB/articles/<name>.parquet (revisions) and <name>.chunks.parquet
(chunk dictionary), plus cache/benchB/manifest.json and revert_delays.csv.
"""
import argparse
import glob
import gzip
import hashlib
import json
import os
import re
from calendar import timegm
from multiprocessing import Pool
from pathlib import Path
from time import strptime

import pyarrow as pa
import pyarrow.parquet as pq

from csrag import paths
from csrag.revert import detect_reverts

HEADING = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$", re.M)


def to_epoch(ts: str) -> float:
    return float(timegm(strptime(ts, "%Y-%m-%dT%H:%M:%SZ")))


def chunk_hash(s: str) -> str:
    return hashlib.blake2b(s.encode("utf-8"), digest_size=16).hexdigest()


def wikitext_to_chunks(wt: str, target_chars: int = 500):
    import mwparserfromhell as mw
    parts, last, title = [], 0, ""
    for m in HEADING.finditer(wt):
        parts.append((title, wt[last:m.start()]))
        title, last = m.group(2).strip(), m.end()
    parts.append((title, wt[last:]))
    chunks = []
    for title, body in parts:
        txt = mw.parse(body).strip_code(normalize=True, collapse=True)
        paras = [p.strip() for p in re.split(r"\n\s*\n", txt) if p.strip()]
        if not paras:
            continue
        cur = (title + "\n") if title else ""
        base = len(cur)
        for p in paras:
            pieces = [p]
            if len(p) > target_chars:               # split a long paragraph at sentence ends
                pieces, buf = [], ""
                for s in re.split(r"(?<=[.!?])\s+", p):
                    if buf and len(buf) + len(s) + 1 > target_chars:
                        pieces.append(buf)
                        buf = s
                    else:
                        buf = (buf + " " + s).strip()
                if buf:
                    pieces.append(buf)
            for piece in pieces:
                if len(cur) > base and len(cur) + len(piece) + 1 > target_chars:
                    chunks.append(cur.strip())
                    cur = (title + "\n") if title else ""
                    base = len(cur)
                cur += piece + "\n"
        if cur.strip() and len(cur) > base:
            chunks.append(cur.strip())
    return chunks


def prepare_article(fn: str):
    out_dir = paths.BENCHB / "articles"
    name = Path(fn).name.replace(".jsonl.gz", "")
    out = out_dir / f"{name}.parquet"
    if out.exists():
        return json.loads((out_dir / f"{name}.meta.json").read_text(encoding="utf-8"))
    revs = [json.loads(l) for l in gzip.open(fn, "rt", encoding="utf-8")]
    kept = [r for r in revs if r["content"] is not None and not r["texthidden"] and not r["sha1hidden"] and r["sha1"]]
    kept.sort(key=lambda r: (r["timestamp"], r["revid"]))
    cache, chunk_dict = {}, {}
    rows = {"rev_id": [], "t_event": [], "sha1": [], "risky": [], "raw_bytes": [], "plain_bytes": [],
            "tags": [], "chunk_hashes": []}
    for r in kept:
        if r["sha1"] not in cache:
            cs = wikitext_to_chunks(r["content"])
            hs = [chunk_hash(c) for c in cs]
            for h, c in zip(hs, cs):
                chunk_dict.setdefault(h, c)
            cache[r["sha1"]] = (hs, sum(len(c.encode("utf-8")) for c in cs))
        hs, pb = cache[r["sha1"]]
        rows["rev_id"].append(r["revid"]); rows["t_event"].append(to_epoch(r["timestamp"]))
        rows["sha1"].append(r["sha1"]); rows["risky"].append(bool(r["anon"] or r["temp"]))
        rows["raw_bytes"].append(len(r["content"].encode("utf-8"))); rows["plain_bytes"].append(pb)
        rows["tags"].append(r["tags"]); rows["chunk_hashes"].append(hs)
    reverted, reverter = detect_reverts(rows["sha1"])
    rows["reverted"] = [i in reverted for i in range(len(kept))]
    rows["reverted_tag"] = ["mw-reverted" in t for t in rows["tags"]]
    rows["revert_delay_s"] = [(rows["t_event"][reverter[i]] - rows["t_event"][i]) if i in reverter else None
                              for i in range(len(kept))]
    pq.write_table(pa.table(rows), out, compression="zstd")
    pq.write_table(pa.table({"hash": list(chunk_dict), "text": list(chunk_dict.values())}),
                   out_dir / f"{name}.chunks.parquet", compression="zstd")
    meta = {"name": name, "n_fetched": len(revs), "n_kept": len(kept), "n_hidden": len(revs) - len(kept),
            "raw_bytes": sum(rows["raw_bytes"]), "plain_bytes": sum(rows["plain_bytes"]),
            "n_chunks_total": sum(len(h) for h in rows["chunk_hashes"]), "n_chunks_unique": len(chunk_dict),
            "n_reverted": len(reverted), "n_reverted_tag": sum(rows["reverted_tag"]),
            "n_risky": sum(rows["risky"])}
    (out_dir / f"{name}.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    (paths.BENCHB / "articles").mkdir(parents=True, exist_ok=True)
    files = []
    for done in sorted(glob.glob(str(paths.WIKI_CACHE / "articles" / "*.done"))):
        m = json.loads(Path(done).read_text(encoding="utf-8"))
        if m["n_revisions"] >= 20:
            files.append(str(paths.WIKI_CACHE / "articles" / m["file"]))
    if a.limit:
        files = files[:a.limit]
    with Pool(a.procs) as pool:
        metas = pool.map(prepare_article, files, chunksize=1)
    tot = {k: sum(m[k] for m in metas) for k in metas[0] if k != "name"}
    tot["articles"] = len(metas)
    (paths.BENCHB / "manifest.json").write_text(json.dumps({"totals": tot, "articles": metas}, indent=1), encoding="utf-8")
    print(json.dumps(tot, indent=1))


if __name__ == "__main__":
    main()
