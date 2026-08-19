"""Proxy genre map for worst-genre checkpoint selection.

Checks for a native genre/domain column in the dataset; falls back to
TF-IDF char n-gram + KMeans if none found (sklearn required for fallback).

Usage (run once per subtask, caches to disk):
  python genre_proxy.py --subtask NoPnx-NP
  python genre_proxy.py --subtask NoPnx-PA --n_clusters 5 --out outputs/genre_proxy
"""
import argparse
import json
from collections import Counter
from pathlib import Path

SUBTASK_DATASETS = {
    "PA":        "MBZUAI/AraSeg-2026-Shared-Task-PA",
    "NoPnx-PA":  "MBZUAI/AraSeg-2026-Shared-Task-NoPnx-PA",
    "NP":        "MBZUAI/AraSeg-2026-Shared-Task-NP",
    "NoPnx-NP":  "MBZUAI/AraSeg-2026-Shared-Task-NoPnx-NP",
}


def build_genre_map(subtask: str, split: str, n_clusters: int = 5) -> dict:
    from datasets import load_dataset
    ds = load_dataset(SUBTASK_DATASETS[subtask], split=split)
    row0 = ds[0]
    id_col = next((k for k in ["doc_id", "id"] if k in row0), None)

    # Native genre column (best case)
    for col in ("genre", "domain", "source", "category", "style"):
        if col in row0:
            print(f"Found native genre column: '{col}'")
            return {str(r[id_col] if id_col else i): str(r[col]) for i, r in enumerate(ds)}

    # Fallback: char n-gram TF-IDF + KMeans on token text
    print(f"No native genre column. Using TF-IDF+KMeans (k={n_clusters}).")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans

    token_col = next(k for k in ["tokens", "words"] if k in row0 and isinstance(row0[k], list))
    texts  = [" ".join(r[token_col]) for r in ds]
    doc_ids = [str(r[id_col] if id_col else i) for i, r in enumerate(ds)]

    vecs = TfidfVectorizer(max_features=5000, analyzer="char_wb", ngram_range=(3, 5)).fit_transform(texts)
    labels = KMeans(n_clusters=min(n_clusters, len(texts)), random_state=42, n_init=10).fit_predict(vecs)
    return dict(zip(doc_ids, [str(lbl) for lbl in labels.tolist()]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subtask", required=True, choices=list(SUBTASK_DATASETS))
    ap.add_argument("--split", default="dev")
    ap.add_argument("--out", default="outputs/genre_proxy")
    ap.add_argument("--n_clusters", type=int, default=5)
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    path = Path(args.out) / f"{args.subtask.lower().replace('-', '_')}_{args.split}.json"

    genre_map = build_genre_map(args.subtask, args.split, args.n_clusters)
    path.write_text(json.dumps(genre_map, ensure_ascii=False))

    print(f"Saved {len(genre_map)} docs → {path}")
    print(f"Genre distribution: {dict(Counter(genre_map.values()).most_common())}")


if __name__ == "__main__":
    main()
