"""Paired document bootstrap for AraSeg locks (queue A5).

Two questions this answers, and one it does not:

  1. How uncertain is one system's macro-F1?  -> percentile CI by resampling documents.
  2. Is candidate B really better than lock A? -> resample the SAME documents for both
     and look at the distribution of (F1_B - F1_A). If that interval straddles zero the
     delta is noise; the reported NoPnx rerun spread is ~±1.5 F1, so this is the gate
     before any re-lock.
  3. It CANNOT test our margin over `omar_saqr` — we do not have his per-document
     predictions, only his leaderboard aggregate.

Inputs are submission CSVs ("Document ID,Prediction" with a 0/1 string per document),
so it works on any locked/candidate output without touching a GPU.

Usage:
    python bootstrap_paired.py --subtask NoPnx-PA --split test \
        --a outputs/oof_stack_sat_nopnx_pa/test_predictions_NoPnx_PA.csv \
        --b outputs/decoder_nopnx_pa/test_predictions_NoPnx_PA.csv
"""
import argparse
import csv

import numpy as np

from scoring import _doc_f1


def read_preds(path):
    """{doc_id: np.array of 0/1} from a submission CSV."""
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            doc_id = str(row["Document ID"]).strip()
            out[doc_id] = np.array([int(c) for c in row["Prediction"].strip()], dtype=np.int64)
    return out


def read_gold(subtask, split):
    """{doc_id: np.array of 0/1} from the HF dataset.

    `data_utils` pulls in torch, which the docstring's "without touching a GPU" promise
    does not survive on a laptop -- so fall back to loading the dataset directly, the
    same way oracle_map_back.py does. Keeps the column sniffing when torch IS present.
    """
    try:
        from data_utils import _load_split, _detect_columns
        raw = _load_split(subtask, split)
        _, lab_col, id_col = _detect_columns(raw[0])
    except ImportError:
        from datasets import load_dataset
        raw = load_dataset("MBZUAI/AraSeg-2026-Shared-Task-" + subtask, split=split)
        lab_col, id_col = "labels", "doc_id"
    return {(str(r[id_col]) if id_col else str(i)): np.asarray([int(x) for x in r[lab_col]],
                                                               dtype=np.int64)
            for i, r in enumerate(raw)}


def per_doc_f1(preds, gold):
    """np.array of per-document F1, ordered by sorted doc_id (shared across systems)."""
    doc_ids = sorted(gold)
    missing = [d for d in doc_ids if d not in preds]
    if missing:
        raise KeyError(f"{len(missing)} docs absent from predictions (e.g. {missing[:3]})")
    f1 = []
    for d in doc_ids:
        p, g = preds[d], gold[d]
        if len(p) != len(g):
            raise ValueError(f"doc {d}: {len(p)} preds vs {len(g)} gold labels")
        f1.append(_doc_f1(p, g)["f1"])
    return np.asarray(f1, dtype=np.float64)


def bootstrap(f1_a, f1_b=None, n=10000, alpha=0.05, seed=42):
    """Resample documents with replacement; the SAME index draw scores both systems
    (that pairing is what cancels per-document difficulty out of the delta)."""
    rng = np.random.default_rng(seed)
    n_docs = len(f1_a)
    idx = rng.integers(0, n_docs, size=(n, n_docs))
    boot_a = f1_a[idx].mean(axis=1)
    lo, hi = np.percentile(boot_a, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    res = {"f1_a": float(f1_a.mean()), "ci_a": (float(lo), float(hi))}
    if f1_b is not None:
        boot_d = f1_b[idx].mean(axis=1) - boot_a
        d_lo, d_hi = np.percentile(boot_d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        res.update({
            "f1_b": float(f1_b.mean()),
            "delta": float(f1_b.mean() - f1_a.mean()),
            "ci_delta": (float(d_lo), float(d_hi)),
            "p_b_better": float((boot_d > 0).mean()),
        })
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subtask", required=True, choices=["PA", "NoPnx-PA", "NP", "NoPnx-NP"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--a", required=True, help="Submission CSV of the lock / baseline")
    ap.add_argument("--b", default=None, help="Submission CSV of the candidate (optional)")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    gold = read_gold(args.subtask, args.split)
    f1_a = per_doc_f1(read_preds(args.a), gold)
    f1_b = per_doc_f1(read_preds(args.b), gold) if args.b else None
    r = bootstrap(f1_a, f1_b, args.n, args.alpha, args.seed)

    conf = int(round(100 * (1 - args.alpha)))
    print(f"[bootstrap] {args.subtask}/{args.split}  {len(gold)} docs  {args.n} resamples")
    print(f"  A {args.a}")
    print(f"    macro-F1 {100*r['f1_a']:.2f}  {conf}% CI [{100*r['ci_a'][0]:.2f}, {100*r['ci_a'][1]:.2f}]")
    if f1_b is not None:
        print(f"  B {args.b}")
        print(f"    macro-F1 {100*r['f1_b']:.2f}")
        lo, hi = r["ci_delta"]
        verdict = "SIGNIFICANT" if lo > 0 or hi < 0 else "NOISE (interval spans 0) — do not re-lock"
        print(f"  delta (B-A) {100*r['delta']:+.2f}  {conf}% CI [{100*lo:+.2f}, {100*hi:+.2f}]"
              f"  P(B>A)={r['p_b_better']:.3f}  -> {verdict}")
    print("  note: cannot bootstrap the margin over omar_saqr — his per-doc preds are not public.")


def _selfcheck():
    rng = np.random.default_rng(0)
    # B is a uniform +2 F1 better than A on every doc => delta CI must exclude 0
    a = rng.uniform(0.70, 0.90, 200)
    b = a + 0.02
    r = bootstrap(a, b, n=2000)
    assert r["ci_delta"][0] > 0, r
    assert r["p_b_better"] == 1.0
    # B is a coin-flip jitter around A => delta CI must contain 0
    b2 = a + rng.normal(0, 0.05, 200)
    r2 = bootstrap(a, b2, n=2000)
    assert r2["ci_delta"][0] < 0 < r2["ci_delta"][1], r2
    # the CI must bracket the point estimate
    assert r["ci_a"][0] < a.mean() < r["ci_a"][1]
    print("OK: paired bootstrap separates a real +2 F1 gain from a zero-mean jitter")


if __name__ == "__main__":
    import sys
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()
