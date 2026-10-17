"""Benchmark A: a small, semi-synthetic question-answering set over 20 fact timelines.

Honest description: each of the 20 timelines is hand-written (facts, one vandal edit and
its revert in 5 timelines, a shortened revision in 7 timelines). Two of each revision's
six chunks are taken from the real lead section of the matching Wikipedia article
(data/raw/wikimedia_real_article_extracts.json). Event times are synthetic and close
together (seconds apart) so that delivery disorder matters. Unlike the v1 corpus, the
20 timelines are NOT copied, and no real Wikipedia edit comments are attached to the
synthetic revisions.

Outputs (data/benchA/): revisions.jsonl (fields a pipeline may see), gt.json (ground
truth), queries.json (as-of questions; current-state questions are generated per probe).
"""
import hashlib
import json
from datetime import datetime, timezone

from csrag import paths
from csrag.revert import detect_reverts

BASE_EPOCH = 1704067200.0  # 2024-01-01T00:00:00Z


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def build_chunks(topic, state, extract):
    title, domain, entity, field, fact = (topic["title"], topic["domain"], topic["entity"],
                                          topic["fact_field"], state["fact_val"])
    ext = " ".join(extract.split()) if extract else ""
    part1 = ext[:420] if len(ext) > 100 else (
        f"{title} has a public record of its history, organization and main activities.")
    part2 = ext[420:840] if len(ext) > 460 else (
        f"Statements about {title} are checked against published sources before they are added.")
    hdr = lambda k, name: f"[{title} | Section {k}: {name}]\n"
    c0 = hdr(1, "Summary") + f"In {title}, for {entity}, the {field} is {fact}."
    c1 = hdr(2, "Background") + part1
    if state.get("is_minor_syntactic", False):
        c2 = hdr(3, "Current status") + f"For {entity} in {title}, the {field} is {fact} (copy edit)."
    else:
        c2 = hdr(3, "Current status") + f"For {entity} in {title}, the {field} is {fact}."
    c3 = hdr(4, "Context") + part2
    chunks = [c0, c1, c2, c3]
    if not state.get("shrink_chunks", False):
        chunks.append(hdr(5, "Details") + f"The latest record of {title} gives the {field} for {entity} as {fact}.")
        chunks.append(hdr(6, "Reference table") + f"{entity}: {field}: {fact}.")
    return chunks


def iso(t):
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def build():
    topics = json.load(open(paths.DATA / "benchA" / "topics.json", encoding="utf-8"))
    extracts = json.load(open(paths.DATA / "raw" / "wikimedia_real_article_extracts.json", encoding="utf-8"))
    revisions, gt, queries = [], {}, []
    for d_idx, topic in enumerate(topics):
        doc_id = f"A{d_idx:02d}"
        t = BASE_EPOCH + d_idx * 1.5
        doc_revs = []
        for s_idx, state in enumerate(topic["v_states"]):
            if s_idx > 0:
                t += 5.0 if state.get("is_revert") else (18.0 if state.get("is_vandalism") else 24.0)
            chunks = build_chunks(topic, state, extracts.get(topic["title"], ""))
            rev = {"doc_id": doc_id, "rev_id": d_idx * 100 + state["rev_num"], "t_event": t,
                   "content_hash": _sha("\n\n".join(chunks)), "chunks": chunks,
                   "chunk_hashes": [_sha(c) for c in chunks], "risky": False}
            doc_revs.append((rev, state))
        hashes = [r["content_hash"] for r, _ in doc_revs]
        reverted_pos, _ = detect_reverts(hashes)
        labels = {i for i, (_, s) in enumerate(doc_revs) if s.get("is_vandalism")}
        assert reverted_pos == labels, (doc_id, reverted_pos, labels)   # detector agrees with hand labels
        gt[doc_id] = {"title": topic["title"], "domain": topic["domain"], "entity": topic["entity"],
                      "fact_field": topic["fact_field"], "revisions": []}
        for i, (rev, state) in enumerate(doc_revs):
            revisions.append(rev)
            gt[doc_id]["revisions"].append({
                "rev_id": rev["rev_id"], "t_event": rev["t_event"], "content_hash": rev["content_hash"],
                "chunk_hashes": rev["chunk_hashes"], "fact": state["fact_val"],
                "is_vandal": bool(state.get("is_vandalism")), "reverted": i in reverted_pos})
        # As-of questions: one per curated revision (midpoint of its valid interval) plus one
        # inside each vandal window (the correct answer is the revision before the vandal edit).
        cur = [r for r in gt[doc_id]["revisions"] if not r["reverted"]]
        for j, r in enumerate(cur):
            tau = (r["t_event"] + cur[j + 1]["t_event"]) / 2 if j + 1 < len(cur) else r["t_event"] + 12.0
            queries.append({"qid": f"{doc_id}-asof-{r['rev_id']}", "doc_id": doc_id, "kind": "asof",
                            "tau": tau, "target_rev": r["rev_id"],
                            "question": f"As of {iso(tau)}, what was the {topic['fact_field']} for {topic['entity']} in {topic['title']}?"})
        for r in gt[doc_id]["revisions"]:
            if r["is_vandal"]:
                tau = r["t_event"] + 2.0
                prev = [c for c in cur if c["t_event"] < r["t_event"]][-1]
                queries.append({"qid": f"{doc_id}-asof-vandalwin-{r['rev_id']}", "doc_id": doc_id, "kind": "asof_vandal_window",
                                "tau": tau, "target_rev": prev["rev_id"],
                                "question": f"As of {iso(tau)}, what was the {topic['fact_field']} for {topic['entity']} in {topic['title']}?"})
        gt[doc_id]["current_question"] = (f"What is the current {topic['fact_field']} for "
                                          f"{topic['entity']} in {topic['title']}?")
    out = paths.DATA / "benchA"
    with open(out / "revisions.jsonl", "w", encoding="utf-8") as f:
        for r in revisions:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(gt, open(out / "gt.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    json.dump(queries, open(out / "queries.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    n_chunks = sum(len(r["chunks"]) for r in revisions)
    uniq = len({h for r in revisions for h in r["chunk_hashes"]})
    print(f"Benchmark A: docs={len(gt)} revisions={len(revisions)} chunks={n_chunks} unique_chunks={uniq} "
          f"asof_queries={len(queries)} reverted={sum(r['reverted'] for g in gt.values() for r in g['revisions'])}")


if __name__ == "__main__":
    build()
