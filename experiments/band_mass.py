"""Q9 go/no-go gate: how much F1 is reachable by re-deciding the ambiguous band?

The queue's Q9 proposes a boundary-verifier that reranks tokens whose ensemble
posterior sits in an ambiguous band (default 0.2-0.8). Before training anything,
measure the ceiling: replay the LOCKED decoder, then re-decide only the in-band
tokens with an oracle and rescore.

Three numbers per head:
  lock          - the locked greedy decode (must reproduce decoder_*.json test F1)
  oracle both   - in-band tokens take the gold label (perfect verifier, both directions)
  oracle FP-only- in-band tokens may only be turned OFF (precision-only verifier)

The FP-only row is the one that matters: our deficit against the field is precision
at matched recall, and a verifier that can only delete is the honest model of Q9.

    Implementation note: an oracle ceiling is a NECESSARY gate, not a sufficient one -- see the
    falsified oracle-headroom rule (oracle-k predicted PA +0.4, real -0.19). A low
    ceiling kills Q9; a high ceiling does not promise the gain.

Usage:
    python band_mass.py --subtask NoPnx-PA
    python band_mass.py --subtask NoPnx-NP --band 0.15 0.85
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from fit_decoder import _TOO_SHORT, build_docs
from scoring import _doc_f1, _macro

DECODER_DIR = {"NoPnx-PA": "outputs/decoder_nopnx_pa", "NoPnx-NP": "outputs/decoder_nopnx_np"}


def _load_members(cache_dir, sat_cache_dir, subtask, split, member_ids):
    """Member caches in lock order. sat_ft lives in its own dir, everything else generic."""
    st = subtask.replace("-", "_")
    out = []
    for mid in member_ids:
        d = Path(sat_cache_dir if mid == "sat_ft" else cache_dir)
        fp = d / f"{st}_{split}_{mid}.pkl"
        if not fp.exists():
            raise FileNotFoundError(f"missing member cache {fp}")
        out.append(pickle.load(open(fp, "rb")))
    return out


def replay(static, w, b, thr, override=None):
    """Greedy decode of one document, returning (preds, posteriors).

    `override` is {token_index: 0|1}, forcing that token's decision — the mechanism a
    second opinion (oracle here, cross-encoder in verify_band.py) plugs into. The gap
    features still advance from whatever was actually committed, so the sequence stays
    self-consistent.
    """
    n_static = static.shape[1]
    z_static = static @ w[:n_static] + b
    w_gap, w_short = w[n_static], w[n_static + 1]
    preds = np.zeros(static.shape[0], dtype=np.int64)
    post = np.zeros(static.shape[0], dtype=np.float64)
    since = 0
    for i in range(static.shape[0]):
        z = z_static[i] + w_gap * np.log1p(since) + w_short * (since < _TOO_SHORT)
        p = 1.0 / (1.0 + np.exp(-z))
        post[i] = p
        hit = bool(override[i]) if override is not None and i in override else p >= thr
        preds[i] = int(hit)
        since = 0 if preds[i] else since + 1
    return preds, post


def _score(docs, w, b, thr, band=None, fp_only=False):
    """Macro P/R/F1. With `band`, in-band tokens are re-decided by an oracle.

    Two-pass, exactly as a deployed verifier must work: decode once to find the
    ambiguous positions, then re-decode with those decisions overridden. Deciding
    band membership *during* the overridden decode would let the candidate set shift
    with decisions the verifier has not made yet.
    """
    rows = []
    for s, g in docs.values():
        ov = None
        if band is not None:
            base, post = replay(s, w, b, thr)
            inb = np.flatnonzero((post >= band[0]) & (post <= band[1]))
            ov = {int(i): int(g[i]) for i in inb
                  if not fp_only or (base[i] == 1 and g[i] == 0)}
        rows.append(_doc_f1(replay(s, w, b, thr, override=ov)[0], g))
    return _macro(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subtask", required=True, choices=sorted(DECODER_DIR))
    ap.add_argument("--split", default="test")
    ap.add_argument("--band", type=float, nargs=2, default=(0.2, 0.8))
    ap.add_argument("--cache_dir", default="outputs/prob_cache")
    ap.add_argument("--sat_cache_dir", default="outputs/sat_fullft_caches")
    args = ap.parse_args()

    meta = json.load(open(Path(DECODER_DIR[args.subtask]) / f"decoder_{args.subtask.replace('-', '_')}.json"))
    w = np.array([meta["weights"][n] for n in meta["feature_names"]], dtype=np.float64)
    b, thr = meta["bias"], meta["dev_threshold"]
    docs = build_docs(_load_members(args.cache_dir, args.sat_cache_dir,
                                    args.subtask, args.split, meta["members"]))
    lo, hi = args.band

    lock = _score(docs, w, b, thr)
    ref = meta.get(args.split)
    if ref:  # self-check: the replay must reproduce the locked decoder exactly
        assert abs(lock["f1"] - ref["f1"]) < 1e-6, \
            f"replay F1 {lock['f1']:.6f} != locked {ref['f1']:.6f} -- feature order wrong?"
        print(f"[gate] replay reproduces locked {args.split} F1 {ref['f1']:.4f} OK")

    # band occupancy over the locked decode
    n_tok = n_band = fp_band = fn_band = fp_all = fn_all = 0
    for s, g in docs.values():
        preds, post = replay(s, w, b, thr)
        inb = (post >= lo) & (post <= hi)
        n_tok += len(g)
        n_band += int(inb.sum())
        fp_all += int(((preds == 1) & (g == 0)).sum())
        fn_all += int(((preds == 0) & (g == 1)).sum())
        fp_band += int(((preds == 1) & (g == 0) & inb).sum())
        fn_band += int(((preds == 0) & (g == 1) & inb).sum())

    both = _score(docs, w, b, thr, band=(lo, hi))
    fponly = _score(docs, w, b, thr, band=(lo, hi), fp_only=True)

    print(f"\n{args.subtask} @{thr:.2f}, band [{lo}, {hi}], split={args.split}")
    print(f"  tokens in band : {n_band}/{n_tok} ({100.0 * n_band / n_tok:.2f}%)")
    print(f"  FPs in band    : {fp_band}/{fp_all} ({100.0 * fp_band / max(fp_all, 1):.1f}% of all FPs)")
    print(f"  FNs in band    : {fn_band}/{fn_all} ({100.0 * fn_band / max(fn_all, 1):.1f}% of all FNs)")
    print(f"\n  {'system':<22} {'P':>7} {'R':>7} {'F1':>7} {'dF1':>7}")
    for name, s in (("lock", lock), ("oracle both ways", both), ("oracle FP-only", fponly)):
        d = "" if name == "lock" else f"{100 * (s['f1'] - lock['f1']):+7.2f}"
        print(f"  {name:<22} {100 * s['precision']:7.2f} {100 * s['recall']:7.2f} "
              f"{100 * s['f1']:7.2f} {d:>7}")


if __name__ == "__main__":
    main()
