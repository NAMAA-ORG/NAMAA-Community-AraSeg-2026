"""Out-of-fold (OOF) helpers for closed-legal stacking.

Standard stacking needs member predictions the meta-model can trust — i.e. made
on data the member never trained on. The dev set gives that, but fitting the
stacker on dev is illegal in the closed track (data must be train only). The
legal substitute: K-fold the *train* docs, train each member K times (each on
train-minus-fold), predict the held-out fold. The union of held-out predictions
is an OOF train matrix we can fit the stacker on, using only train data.

Folds are defined *per subtask* from that subtask's own train doc-ids, so
train.py and fit_oof_stack.py agree without assuming doc-ids match across
subtasks. A multitask member (e38) trained holding out fold i of every subtask
still yields honest OOF preds for each head, because for each head it excluded
exactly that head's fold-i docs.
"""
import random
from typing import Dict, List


def train_doc_ids(subtask: str) -> List[str]:
    """Sorted list of a subtask's train doc-ids (deterministic order)."""
    from data_utils import _load_split, _detect_columns  # lazy: pulls torch
    raw = _load_split(subtask, "train")
    _, _, id_col = _detect_columns(raw[0])
    ids = [str(r[id_col]) if id_col else str(i) for i, r in enumerate(raw)]
    return sorted(ids)


def fold_map(subtask: str, n_folds: int = 5, seed: int = 42) -> Dict[str, int]:
    """doc_id -> fold index in [0, n_folds). Seeded shuffle then round-robin so
    folds are balanced in size and stable across processes/runs."""
    ids = train_doc_ids(subtask)
    rng = random.Random(seed)
    rng.shuffle(ids)
    return {doc_id: i % n_folds for i, doc_id in enumerate(ids)}


def holdout_ids(subtask: str, fold: int, n_folds: int = 5, seed: int = 42) -> set:
    """Doc-ids in the given fold (the ones a fold-`fold` model must NOT train on)."""
    fm = fold_map(subtask, n_folds, seed)
    return {d for d, f in fm.items() if f == fold}


def keep_ids(subtask: str, fold: int, n_folds: int = 5, seed: int = 42) -> set:
    """Doc-ids to TRAIN on for fold `fold` (everything except the held-out fold)."""
    fm = fold_map(subtask, n_folds, seed)
    return {d for d, f in fm.items() if f != fold}


if __name__ == "__main__":
    # Implementation note: self-check the fold partition math without needing the real
    # dataset — monkeypatch train_doc_ids with a synthetic universe.
    import sys
    _mod = sys.modules[__name__]
    _mod.train_doc_ids = lambda st: [f"d{i:03d}" for i in range(174)]
    K = 5
    all_ids = set(_mod.train_doc_ids("PA"))
    # every doc lands in exactly one fold; keep + holdout partition the universe
    seen = set()
    sizes = []
    for f in range(K):
        h = holdout_ids("PA", f, K)
        k = keep_ids("PA", f, K)
        assert h & k == set(), "keep and holdout overlap"
        assert h | k == all_ids, "keep+holdout must cover all docs"
        assert seen & h == set(), "a doc appears in two folds"
        seen |= h
        sizes.append(len(h))
    assert seen == all_ids, "folds must cover every doc exactly once"
    assert max(sizes) - min(sizes) <= 1, f"folds unbalanced: {sizes}"
    # determinism
    assert fold_map("PA", K) == fold_map("PA", K), "fold_map not deterministic"
    print(f"OK: {len(all_ids)} docs, {K} folds, sizes={sizes}")
