"""Gate G3: SHA-1 reported by the MediaWiki API equals SHA-1 of the fetched text, for every revision."""
import glob, gzip, hashlib, json, sys
from multiprocessing import Pool
from csrag import paths

def check(fn):
    tot = ok = hidden = 0
    for line in gzip.open(fn, "rt", encoding="utf-8"):
        r = json.loads(line)
        if r["content"] is None or r["texthidden"] or r["sha1hidden"] or not r["sha1"]:
            hidden += 1
            continue
        tot += 1
        ok += hashlib.sha1(r["content"].encode("utf-8")).hexdigest() == r["sha1"]
    return tot, ok, hidden

if __name__ == "__main__":
    files = sorted(glob.glob(str(paths.WIKI_CACHE / "articles" / "*.jsonl.gz")))
    with Pool(8) as p:
        res = p.map(check, files)
    t, o, h = (sum(x[i] for x in res) for i in range(3))
    out = {"files": len(files), "checked": t, "sha1_match": o, "hidden_or_missing": h}
    (paths.WIKI_CACHE / "sha1_verification.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))
