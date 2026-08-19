"""Document-level evaluation for AraSeg 2026.

The task scores P/R/F1 per document, then macro-averages across documents.

Documents are processed as overlapping windows (see data_utils.py). This module
stitches per-window predictions back into full-document, per-word predictions
before scoring, so the F1 reflects the WHOLE document — not just the first 512
tokens. For words that appear in more than one window (overlap region), the
prediction from the window where the word sits closest to the centre is kept
(edge tokens have the least surrounding context).

It also supports decision-threshold tuning: boundaries are the rare class, so
the F1-optimal probability threshold is usually well below 0.5. We collect
per-word boundary probabilities once, then sweep thresholds cheaply.
"""

from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np
import torch

# The F1 math lives in scoring.py (torch-free) so CPU-only tools can import it
# without pulling torch. Re-exported here — every existing `from metrics import
# _doc_f1` keeps working.
from scoring import _doc_f1, _macro


@torch.no_grad()
def collect_doc_probs(model, dataloader, device: str, subtask: str,
                      doc_gold: Dict[str, List[int]]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Run the model over all windows and reassemble per-document boundary
    probabilities. Returns {doc_id: (prob_per_word, gold_per_word)}.

    Words never covered by any window (e.g. tail of an over-length document that
    even windowing truncates) default to probability 0.0 — correctly counted as
    a missed boundary if gold==1.
    """
    model.eval()
    # doc_id -> word_idx -> (best_prob, distance_to_window_centre)
    store: Dict[str, Dict[int, Tuple[float, float]]] = defaultdict(dict)

    for batch in dataloader:
        logits = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            token_type_ids=batch["token_type_ids"].to(device),
            subtask=subtask,
            syntax_pos_ids=batch["syntax_pos_ids"].to(device) if "syntax_pos_ids" in batch else None,
            syntax_dep_ids=batch["syntax_dep_ids"].to(device) if "syntax_dep_ids" in batch else None,
        )
        prob1 = torch.softmax(logits, dim=-1)[..., 1].cpu().numpy()   # (B, T)
        word_idx = batch["word_idx"].numpy()                          # (B, T)
        attn = batch["attention_mask"].numpy()                        # (B, T)
        doc_ids = batch["doc_ids"]

        for i in range(prob1.shape[0]):
            length = int(attn[i].sum())          # real (unpadded) window length
            centre = length / 2.0
            d_store = store[doc_ids[i]]
            for p in range(length):
                g = int(word_idx[i, p])
                if g < 0:
                    continue
                dist = abs(p - centre)
                prev = d_store.get(g)
                if prev is None or dist < prev[1]:
                    d_store[g] = (float(prob1[i, p]), dist)

    docs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for doc_id, gold in doc_gold.items():
        n = len(gold)
        parr = np.zeros(n, dtype=np.float32)
        for w, (prob, _) in store.get(doc_id, {}).items():
            if w < n:
                parr[w] = prob
        docs[doc_id] = (parr, np.asarray(gold, dtype=np.int64))
    return docs


def macro_f1_at(docs: Dict[str, Tuple[np.ndarray, np.ndarray]], threshold: float) -> Dict[str, float]:
    """Document-macro P/R/F1 at a given probability threshold."""
    scores = [_doc_f1((parr >= threshold).astype(np.int64), gold) for parr, gold in docs.values()]
    return _macro(scores)


@torch.no_grad()
def collect_doc_viterbi(model, dataloader, device: str, subtask: str,
                        doc_gold: Dict[str, List[int]]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Stitch per-word Viterbi predictions into full documents for a CRF model."""
    model.eval()
    store: Dict[str, Dict[int, Tuple[int, float]]] = defaultdict(dict)
    for batch in dataloader:
        paths = model.crf_decode_words(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            token_type_ids=batch["token_type_ids"].to(device),
            subtask=subtask,
            word_idx=batch["word_idx"].to(device),
        )
        word_idx = batch["word_idx"].numpy()
        doc_ids = batch["doc_ids"]
        for i, path in enumerate(paths):
            gids = [int(w) for w in word_idx[i] if w >= 0]
            centre = len(gids) / 2.0
            d_store = store[doc_ids[i]]
            for j, (g, pred) in enumerate(zip(gids, path)):
                dist = abs(j - centre)
                prev = d_store.get(g)
                if prev is None or dist < prev[1]:
                    d_store[g] = (int(pred), dist)
    docs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for doc_id, gold in doc_gold.items():
        n = len(gold)
        parr = np.zeros(n, dtype=np.int64)
        for w, (pred, _) in store.get(doc_id, {}).items():
            if w < n:
                parr[w] = pred
        docs[doc_id] = (parr, np.asarray(gold, dtype=np.int64))
    return docs


def macro_f1_preds(docs: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict[str, float]:
    """Document-macro P/R/F1 from hard per-word predictions."""
    scores = [_doc_f1(pred, gold) for pred, gold in docs.values()]
    return _macro(scores)


def tune_threshold(docs: Dict[str, Tuple[np.ndarray, np.ndarray]],
                   grid: np.ndarray = None) -> Tuple[float, Dict[str, float]]:
    """Sweep thresholds and return (best_threshold, scores_at_best) by macro-F1."""
    if grid is None:
        grid = np.arange(0.05, 0.96, 0.05)
    best_thr, best = 0.5, {"precision": 0.0, "recall": 0.0, "f1": -1.0}
    for thr in grid:
        s = macro_f1_at(docs, float(thr))
        if s["f1"] > best["f1"]:
            best_thr, best = float(thr), s
    return best_thr, best


@torch.no_grad()
def evaluate(model, dataloader, device: str, subtask: str,
             doc_gold: Dict[str, List[int]], threshold: float = 0.5) -> Dict[str, float]:
    """Full-document macro P/R/F1 at a fixed threshold (default argmax-equivalent 0.5)."""
    docs = collect_doc_probs(model, dataloader, device, subtask, doc_gold)
    return macro_f1_at(docs, threshold)
