"""
AraSeg 2026 — token-length diagnostic.

Confirms or refutes the truncation hypothesis: each dataset row is a whole
document, and configs use max_length=512 with truncation=True. If documents
tokenize to far more than 512 subwords, the tail of every long document is
discarded and its sentence boundaries are never predicted — which tanks the
document-level macro-F1.

This script tokenizes every document exactly the way training does
(is_split_into_words=True, no truncation) and reports the subword-length
distribution per subtask/split, plus how many docs would be truncated at a
given max_length.

Usage:
    python check_lengths.py                          # XLM-R tokenizer, max_length=512
    python check_lengths.py --model aubmindlab/bert-large-arabertv02
    python check_lengths.py --max-length 512 --subtasks PA NoPnx-NP
"""

import argparse
from typing import List

import numpy as np
from transformers import AutoTokenizer

from data_utils import SUBTASK_DATASETS, _load_split, _detect_columns
from utils import resolve_model_path

SPLITS = ["train", "dev", "test"]


def pct(arr: np.ndarray, q: float) -> int:
    return int(np.percentile(arr, q)) if len(arr) else 0


def analyze_split(subtask: str, split: str, tokenizer, max_length: int):
    try:
        raw = _load_split(subtask, split)
    except Exception as e:
        print(f"  {split:5s}: (unavailable — {type(e).__name__})")
        return

    token_col, label_col, _ = _detect_columns(raw[0])

    sub_lens: List[int] = []   # subword length per doc (no special tokens added cost ignored — small)
    word_lens: List[int] = []  # whitespace-token length per doc
    boundary_total = 0         # gold boundaries (label==1) overall
    boundary_lost = 0          # gold boundaries that fall beyond max_length (truncated away)

    for row in raw:
        words = row[token_col]
        labels = [int(l) for l in row[label_col]]
        word_lens.append(len(words))

        # Tokenize the same way training does, but WITHOUT truncation, so we see
        # the true length and can map word -> subword position.
        enc = tokenizer(words, is_split_into_words=True, add_special_tokens=True)
        word_ids = enc.word_ids()
        n_sub = len(word_ids)
        sub_lens.append(n_sub)

        # A boundary at word w is "lost" if the FIRST subword of word w lands at
        # subword index >= max_length (it would be cut by truncation).
        first_sub_pos = {}
        for pos, wid in enumerate(word_ids):
            if wid is not None and wid not in first_sub_pos:
                first_sub_pos[wid] = pos
        for wid, lab in enumerate(labels):
            if lab == 1:
                boundary_total += 1
                pos = first_sub_pos.get(wid)
                if pos is None or pos >= max_length:
                    boundary_lost += 1

    sub = np.array(sub_lens)
    n_trunc = int((sub > max_length).sum())
    lost_pct = (100.0 * boundary_lost / boundary_total) if boundary_total else 0.0

    print(
        f"  {split:5s}: n={len(sub):4d}  "
        f"words[P50={pct(np.array(word_lens),50):4d} P90={pct(np.array(word_lens),90):4d}]  "
        f"subwords[P50={pct(sub,50):4d} P90={pct(sub,90):4d} P99={pct(sub,99):5d} max={int(sub.max()):5d}]  "
        f"|  >{max_length}: {n_trunc}/{len(sub)} docs ({100.0*n_trunc/len(sub):.0f}%)  "
        f"boundaries lost: {boundary_lost}/{boundary_total} ({lost_pct:.1f}%)"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="xlm-roberta-large",
                    help="Tokenizer to measure against (use the one you'll train with).")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--subtasks", nargs="+", default=list(SUBTASK_DATASETS.keys()))
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()

    model_path = resolve_model_path(args.model)
    print(f"Tokenizer: {model_path}")
    print(f"max_length: {args.max_length}\n")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=args.trust_remote_code)

    for st in args.subtasks:
        print(f"[{st}]")
        for split in SPLITS:
            analyze_split(st, split, tokenizer, args.max_length)
        print()

    print("Reading the output:")
    print("  - 'subwords P90/P99/max' >> max_length  ->  truncation is real.")
    print("  - 'boundaries lost %' is the direct recall ceiling imposed by truncation;")
    print("    e.g. 30% lost means single-window training/inference cannot exceed ~0.70 recall.")
    print("  - If P90 is comfortably under max_length, the truncation hypothesis is refuted.")


if __name__ == "__main__":
    main()
