"""
ChronoStream-RAG: Local Neural Model Downloader & Verifier
==========================================================
Downloads and verifies the 3 local HuggingFace neural models into `models/`:
  1. `sentence-transformers/paraphrase-MiniLM-L6-v2` (384-d Dense Bi-Encoder, ~91 MB)
  2. `google/flan-t5-small` (Instruction-Tuned Seq2Seq Generative SLM, ~308 MB)
  3. `distilbert-base-cased-distilled-squad` (Extractive Span-QA Transformer, ~261 MB)
"""

from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForQuestionAnswering,
)


def download_and_verify_models(base_dir: str = "models"):
    root = Path(base_dir)
    root.mkdir(parents=True, exist_ok=True)

    # 1. Dense Bi-Encoder (384-d)
    p_embed = root / "paraphrase-MiniLM-L6-v2"
    print(f"[1/3] Downloading/verifying SentenceTransformer -> {p_embed}")
    st_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")
    st_model.save(str(p_embed))

    # 2. Generative Seq2Seq SLM (Flan-T5-Small)
    p_flan = root / "flan-t5-small"
    print(f"[2/3] Downloading/verifying Flan-T5-Small -> {p_flan}")
    flan_tok = AutoTokenizer.from_pretrained("google/flan-t5-small")
    flan_mod = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    flan_tok.save_pretrained(str(p_flan))
    flan_mod.save_pretrained(str(p_flan))

    # 3. Extractive Span-QA Transformer (DistilBERT-SQuAD)
    p_distil = root / "distilbert-base-cased-distilled-squad"
    print(f"[3/3] Downloading/verifying DistilBERT-SQuAD -> {p_distil}")
    distil_tok = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
    distil_mod = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")
    distil_tok.save_pretrained(str(p_distil))
    distil_mod.save_pretrained(str(p_distil))

    print("[DONE] All 3 neural models are saved and verified in:", root.resolve())


if __name__ == "__main__":
    download_and_verify_models()
