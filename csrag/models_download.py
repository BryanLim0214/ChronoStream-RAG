"""Download the four models used in the experiments and record their exact revisions."""
import json

from huggingface_hub import snapshot_download, HfApi

from csrag import paths

MODELS = {
    "embedder": "sentence-transformers/paraphrase-MiniLM-L6-v2",
    "flan_t5_small": "google/flan-t5-small",
    "distilbert_squad": "distilbert/distilbert-base-cased-distilled-squad",
    "qwen2_5_1_5b_instruct": "Qwen/Qwen2.5-1.5B-Instruct",
}


def main():
    api = HfApi()
    record = {}
    for key, repo in MODELS.items():
        info = api.model_info(repo)
        local = snapshot_download(repo_id=repo, revision=info.sha, local_dir=paths.MODELS / key,
                                  allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model",
                                                  "1_Pooling/*", "*.py", "tokenizer*", "merges.txt", "vocab*"])
        record[key] = {"repo": repo, "revision": info.sha, "license": (info.card_data or {}).get("license"),
                       "local_dir": str(local)}
        print(key, info.sha, record[key]["license"], flush=True)
    (paths.MODELS / "model_revisions.json").write_text(json.dumps(record, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
