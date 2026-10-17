"""Answer every unique (question, retrieved context) of a Benchmark A run with three readers, then score.

python experiments/run_readers.py --exp A1_main
Writes results_v2/<exp>/answers.parquet and answers_scored.parquet.
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from csrag import paths
from csrag.readers import READERS, context_text, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--readers", default="flan_t5_small,distilbert_squad,qwen2_5_1_5b_instruct")
    a = ap.parse_args()
    d = paths.RESULTS / a.exp
    ctx = pd.read_parquet(d / "contexts.parquet")
    probes = pd.read_parquet(d / "probes.parquet")
    ans_file = d / "answers.parquet"
    done = pd.read_parquet(ans_file) if ans_file.exists() else pd.DataFrame(columns=["cid", "reader", "answer"])
    rows = [done]
    for rname in a.readers.split(","):
        have = set(done[done.reader == rname].cid)
        todo = ctx[~ctx.cid.isin(have)]
        if not len(todo):
            continue
        t0 = time.perf_counter()
        reader = READERS[rname]()
        answers = reader.answer(todo.question.tolist(), [context_text(t) for t in todo.texts])
        rows.append(pd.DataFrame({"cid": todo.cid.values, "reader": rname, "answer": answers}))
        print(f"{rname}: {len(todo)} contexts in {time.perf_counter() - t0:.1f}s", flush=True)
        del reader
        gc.collect()
        torch.cuda.empty_cache()
    ans = pd.concat(rows, ignore_index=True).drop_duplicates(["cid", "reader"])
    ans.to_parquet(ans_file)
    gt = json.load(open(paths.DATA / "benchA" / "gt.json", encoding="utf-8"))
    facts = {(doc, r["rev_id"]): (r["fact"], r["t_event"], r["is_vandal"]) for doc, g in gt.items() for r in g["revisions"]}
    by_doc = {doc: g["revisions"] for doc, g in gt.items()}
    scored = []
    p = probes[probes.target_rev.notna()].merge(ans, on="cid")
    for row in p.itertuples(index=False):
        tfact, tt, _ = facts[(row.doc, int(row.target_rev))]
        others = {}
        for r in by_doc[row.doc]:
            if r["fact"] == tfact:
                continue
            lab = "vandal" if r["is_vandal"] else ("older" if r["t_event"] < tt else "newer")
            others[f"{lab}|{r['rev_id']}"] = r["fact"]
        sc = score(row.answer, tfact, others)
        scored.append({**row._asdict(), **sc})
    pd.DataFrame(scored).to_parquet(d / "answers_scored.parquet")
    s = pd.DataFrame(scored)
    print(s.groupby(["reader", "method"]).correct.mean().unstack(0).round(3).to_string())
    print(s.groupby(["reader", "ctx_state"]).correct.agg(["mean", "size"]).round(3).to_string())


if __name__ == "__main__":
    main()
