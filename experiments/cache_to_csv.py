"""Cached member probabilities -> a submission CSV, at a given threshold. CPU-only.

Exists so a single model's predictions can be regenerated for `bootstrap_paired.py`
without a GPU or a retrain: every ensemble member already has its probabilities cached
under outputs/prob_cache/, so a matched control (e.g. `e17` for `e71`) costs nothing.
`train.py` writes the same CSV, but only at training time -- if that file is gone or
was never kept, this is the cheap way back.

    python cache_to_csv.py --subtask NoPnx-PA --split test --member e17 --threshold 0.55
    python cache_to_csv.py --selfcheck

Thresholds are the member's own dev-tuned value (ledger §2.0), NOT a lock threshold.
"""
import argparse
import pickle
from pathlib import Path

import numpy as np

from scoring import _doc_f1, _macro

# member -> (subtask, dev-tuned threshold, published test F1) from ledger §2.0.
# The F1 is what --selfcheck asserts, so a cache/threshold mix-up fails loudly.
KNOWN = {
    "e17": ("NoPnx-PA", 0.55, 0.8147),
    "e19": ("NoPnx-NP", 0.50, 0.8211),
}


def cache_path(subtask, split, member):
    return Path("outputs/prob_cache") / f"{subtask.replace('-', '_')}_{split}_{member}.pkl"


def dump(subtask, split, member, threshold, out=None, verbose=True):
    cache = pickle.load(open(cache_path(subtask, split, member), "rb"))
    out = Path(out or f"outputs/{member}/test_predictions_{subtask.replace('-', '_')}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    scored = []
    with out.open("w", encoding="utf-8") as fh:
        fh.write("Document ID,Prediction\n")
        for did, (prob, gold) in cache.items():
            pred = (np.asarray(prob) >= threshold).astype(np.int64)
            fh.write(f"{did},{''.join(str(int(b)) for b in pred)}\n")
            if gold is not None and len(gold):
                scored.append(_doc_f1(pred, np.asarray(gold, np.int64)))

    m = _macro(scored) if scored else None
    if verbose:
        print(f"[cache2csv] {member} {subtask} {split} @{threshold} -> {out}"
              + (f"  (P {m['precision']:.4f} R {m['recall']:.4f} F1 {m['f1']:.4f})" if m else ""))
    return m


def _selfcheck():
    for member, (subtask, thr, want) in KNOWN.items():
        m = dump(subtask, "test", member, thr, verbose=False)
        got = m["f1"]
        assert abs(got - want) < 0.001, f"{member}: {got:.4f} != ledger {want:.4f}"
        print(f"  {member} {subtask} @{thr}: F1 {got:.4f} == ledger {want:.4f}  OK")
    print("selfcheck OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subtask")
    ap.add_argument("--split", default="test")
    ap.add_argument("--member")
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--out")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        _selfcheck()
    else:
        if a.member in KNOWN and (a.subtask is None or a.threshold is None):
            sub, thr, _ = KNOWN[a.member]          # fill in the known defaults
            a.subtask, a.threshold = a.subtask or sub, a.threshold if a.threshold else thr
        dump(a.subtask, a.split, a.member, a.threshold, a.out)
