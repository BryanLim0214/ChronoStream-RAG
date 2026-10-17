"""Answer readers and answer scoring for Benchmark A.

Each unique (question, retrieved context) pair is answered once per reader with greedy
decoding. Scoring does NOT fail an answer just because its context was torn: an answer is
correct when its token F1 against the target fact is at least 0.5 and strictly higher
than against every other fact of that document (older facts, newer facts, vandal text).
"""
import re
from typing import Dict, List, Sequence

import torch
from transformers import (AutoModelForCausalLM, AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM,
                          AutoTokenizer)

from csrag import paths

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def context_text(texts: Sequence[str]) -> str:
    return "\n\n".join(t for t in texts if t)


class FlanReader:
    name = "flan_t5_small"

    def __init__(self):
        p = paths.MODELS / "flan_t5_small"
        self.tok = AutoTokenizer.from_pretrained(p)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(p).to(DEVICE).eval()

    @torch.no_grad()
    def answer(self, questions: List[str], contexts: List[str], bs: int = 32) -> List[str]:
        out = []
        for i in range(0, len(questions), bs):
            prompts = [f"Read the text and answer the question.\n\nText: {c}\n\nQuestion: {q}\nAnswer:"
                       for q, c in zip(questions[i:i + bs], contexts[i:i + bs])]
            enc = self.tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
            gen = self.model.generate(**enc, max_new_tokens=48, do_sample=False, num_beams=1)
            out += [s.strip() for s in self.tok.batch_decode(gen, skip_special_tokens=True)]
        return out


class DistilBertReader:
    name = "distilbert_squad"

    def __init__(self):
        p = paths.MODELS / "distilbert_squad"
        self.tok = AutoTokenizer.from_pretrained(p)
        self.model = AutoModelForQuestionAnswering.from_pretrained(p).to(DEVICE).eval()

    @torch.no_grad()
    def answer(self, questions: List[str], contexts: List[str], bs: int = 32) -> List[str]:
        out = []
        for i in range(0, len(questions), bs):
            qs, cs = questions[i:i + bs], contexts[i:i + bs]
            enc = self.tok(qs, [c if c else "." for c in cs], return_tensors="pt", padding=True, truncation="only_second",
                           max_length=512, return_offsets_mapping=True)
            offsets = enc.pop("offset_mapping")
            seq_ids = [enc.sequence_ids(j) for j in range(len(qs))]
            res = self.model(**{k: v.to(DEVICE) for k, v in enc.items()})
            st, en = res.start_logits.cpu(), res.end_logits.cpu()
            for j in range(len(qs)):
                ctx_idx = [k for k, s in enumerate(seq_ids[j]) if s == 1]
                if not ctx_idx or not cs[j]:
                    out.append("")
                    continue
                best, span = -1e9, (ctx_idx[0], ctx_idx[0])
                s_log, e_log = st[j], en[j]
                top_s = sorted(ctx_idx, key=lambda k: -float(s_log[k]))[:20]
                top_e = sorted(ctx_idx, key=lambda k: -float(e_log[k]))[:20]
                for a in top_s:
                    for b in top_e:
                        if a <= b <= a + 40:
                            sc = float(s_log[a] + e_log[b])
                            if sc > best:
                                best, span = sc, (a, b)
                a, b = span
                out.append(cs[j][int(offsets[j][a][0]):int(offsets[j][b][1])].strip())
        return out


class QwenReader:
    name = "qwen2_5_1_5b_instruct"

    def __init__(self):
        p = paths.MODELS / "qwen2_5_1_5b_instruct"
        self.tok = AutoTokenizer.from_pretrained(p)
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(p, torch_dtype=torch.bfloat16).to(DEVICE).eval()

    @torch.no_grad()
    def answer(self, questions: List[str], contexts: List[str], bs: int = 16) -> List[str]:
        out = []
        for i in range(0, len(questions), bs):
            prompts = []
            for q, c in zip(questions[i:i + bs], contexts[i:i + bs]):
                msgs = [{"role": "system", "content": "Answer the question using only the given text. "
                                                      "Reply with the answer only, with no explanation."},
                        {"role": "user", "content": f"Text:\n{c}\n\nQuestion: {q}"}]
                prompts.append(self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            enc = self.tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1536).to(DEVICE)
            gen = self.model.generate(**enc, max_new_tokens=64, do_sample=False, num_beams=1,
                                      pad_token_id=self.tok.eos_token_id)
            out += [s.strip() for s in self.tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]
        return out


READERS = {"flan_t5_small": FlanReader, "distilbert_squad": DistilBertReader, "qwen2_5_1_5b_instruct": QwenReader}


# ---------------------------------------------------------------- scoring
def _norm(s: str) -> List[str]:
    s = s.lower()
    s = re.sub(r"[^a-z0-9.\s]", " ", s)
    s = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", s)      # keep decimal points, drop sentence periods
    return s.split()


def token_f1(pred: str, gold: str) -> float:
    p, g = _norm(pred), _norm(gold)
    if not p or not g:
        return 0.0
    common = 0
    gc: Dict[str, int] = {}
    for w in g:
        gc[w] = gc.get(w, 0) + 1
    for w in p:
        if gc.get(w, 0) > 0:
            common += 1
            gc[w] -= 1
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def score(answer: str, target_fact: str, other_facts: Dict[str, str]) -> dict:
    """other_facts: label -> fact text (labels like 'older', 'newer', 'vandal')."""
    ft = token_f1(answer, target_fact)
    best_label, best_f = "target", ft
    for lab, fact in other_facts.items():
        f = token_f1(answer, fact)
        if f > best_f:
            best_label, best_f = lab, f
    tie = any(token_f1(answer, fact) == ft for fact in other_facts.values()) and ft > 0
    correct = ft >= 0.5 and best_label == "target" and not tie
    if best_f < 0.5:
        best_label = "none"
    return {"f1_target": ft, "attributed": best_label if not (tie and best_label == "target") else "ambiguous",
            "correct": correct}
