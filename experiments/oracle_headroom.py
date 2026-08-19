"""Bound the maximum gain any doc-level/structural decoder could get on a head.

Answers, for ~$0 and no training, the question "is it worth training OOF folds for a
decoder on this head?" — by measuring ORACLE ceilings on the cached member probs. If
even a cheating decoder can't beat the global threshold by much, an honest one can't
either, and the fold-training budget is wasted.

Ceilings measured (all use the SAME combined probs the lock uses, so the only thing
that varies is the DECISION RULE):

  global      the lock's rule: one dev-tuned threshold for every word in every doc.
  oracle-thr  a different threshold per document, chosen with knowledge of the gold.
              Upper bound on ANY feature that adapts the threshold per doc — which is
              exactly what doc_density / log_doclen / rel_pos can express.
  oracle-k    keep the top-k highest-prob words per document, k = gold boundary count.
              Upper bound on any pure RANKING rule — what local_rank / local_max express.

Read it like this:
  oracle-thr - global   ~= headroom for doc-level calibration features
  oracle-k   - global   ~= headroom for within-doc ranking features
  both small  ->  the decoder CANNOT help this head. Don't train the folds.

Note these are ceilings, not forecasts: a real decoder gets a fraction of them. On
NoPnx-PA the real decoder banked +0.92, so treat anything under ~+1.0 of ceiling as
"probably not worth 20 GPU-hours".

Usage (login node, .venv-llm, offline-HF env exported):
  python oracle_headroom.py --subtask PA --split test \
      --members e25 e32 e33 --cache_dir outputs/prob_cache \
      --sat_cache_dir outputs/sat_fullft_caches   # optional, adds sat_ft
"""
import argparse
from pathlib import Path

import numpy as np

from scoring import _doc_f1, _macro

EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def combine(member_probs, mode):
    """member_probs: list of 1-D arrays (one per member), same length. -> 1-D array."""
    stacked = np.vstack(member_probs)
    if mode == "prob":
        return stacked.mean(axis=0)
    mean_lo = _logit(stacked).mean(axis=0)          # logit = mean of log-odds
    return 1.0 / (1.0 + np.exp(-mean_lo))


def score(docs, decide):
    """decide(probs, gold) -> 0/1 array. Returns macro P/R/F1 over docs."""
    return _macro([_doc_f1(decide(p, g), g) for p, g in docs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subtask", required=True,
                    choices=["PA", "NoPnx-PA", "NP", "NoPnx-NP"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--members", required=True, nargs="+",
                    help="Experiment ids, e.g. e25 e32 e33")
    ap.add_argument("--cache_dir", default="outputs/prob_cache")
    ap.add_argument("--sat_cache_dir", default=None,
                    help="If given, adds sat_ft as an extra member.")
    ap.add_argument("--combine", choices=["prob", "logit"], default="logit")
    ap.add_argument("--thr_step", type=float, default=0.01)
    args = ap.parse_args()

    from cache_probs import load_cached_member  # torch-free loader

    tags = list(args.members)
    # Cache format is {doc_id: (probs, gold)} — same tuple fit_decoder.build_docs unpacks.
    caches = [load_cached_member(Path(args.cache_dir), args.subtask, args.split, t)
              for t in tags]
    if args.sat_cache_dir:
        caches.append(load_cached_member(Path(args.sat_cache_dir), args.subtask,
                                         args.split, "sat_ft"))
        tags.append("sat_ft")

    doc_ids = sorted(set(caches[0]).intersection(*[set(c) for c in caches[1:]]))
    if not doc_ids:
        raise SystemExit("no overlapping doc ids across caches — check --cache_dir/tags")

    docs = []
    for d in doc_ids:
        probs = combine([np.asarray(c[d][0], dtype=np.float64) for c in caches],
                        args.combine)
        gold = np.asarray(caches[0][d][1], dtype=int)
        if len(probs) != len(gold):
            raise SystemExit(f"length mismatch in doc {d}: {len(probs)} vs {len(gold)}")
        docs.append((probs, gold))

    n_gold = sum(int(g.sum()) for _, g in docs)
    if n_gold == 0:
        raise SystemExit("split has no gold boundaries (blind?) — headroom needs labels")

    print(f"[oracle] {args.subtask}/{args.split}  {len(docs)} docs  "
          f"members={tags}  combine={args.combine}")

    grid = np.arange(0.05, 0.96, args.thr_step)

    # 1. global threshold — the lock's decision rule.
    best_t, best = None, None
    for t in grid:
        s = score(docs, lambda p, g, t=t: (p >= t).astype(int))
        if best is None or s["f1"] > best["f1"]:
            best_t, best = float(t), s
    print(f"  global      thr={best_t:.2f}  P={best['precision']:.4f} "
          f"R={best['recall']:.4f}  F1={best['f1']:.4f}")

    # 2. oracle per-doc threshold — ceiling for doc-level calibration features.
    def per_doc_best(p, g):
        bf, bp = -1.0, (p >= best_t).astype(int)
        for t in grid:
            pred = (p >= t).astype(int)
            f = _doc_f1(pred, g)["f1"]
            if f > bf:
                bf, bp = f, pred
        return bp
    o_thr = score(docs, per_doc_best)
    print(f"  oracle-thr                P={o_thr['precision']:.4f} "
          f"R={o_thr['recall']:.4f}  F1={o_thr['f1']:.4f}   "
          f"(+{100 * (o_thr['f1'] - best['f1']):.2f} over global)")

    # 3. oracle top-k per doc — ceiling for within-doc ranking features.
    o_k = score(docs, _top_k)
    print(f"  oracle-k                  P={o_k['precision']:.4f} "
          f"R={o_k['recall']:.4f}  F1={o_k['f1']:.4f}   "
          f"(+{100 * (o_k['f1'] - best['f1']):.2f} over global)")

    # Calibration, from the two heads where we know the REAL decoder outcome:
    #
    #   head      oracle-thr  oracle-k   real decoder
    #   NoPnx-PA     +3.53      +1.55        +0.92
    #   NP           +1.76      +0.52        -0.13
    #
    # oracle-thr is a BAD predictor — NP had +1.76 of it and the real decoder still
    # lost. Most of that ceiling is unreachable: an honest decoder never knows a doc's
    # gold boundary count, so doc_density can only ever be a prior, not information.
    # oracle-k (ranking headroom) is what tracks reality, and it matches the fitted
    # local_rank weights (+0.70 on NoPnx-PA vs +0.05 on NP). Line through those two
    # points: realized ~= 1.02 * oracle_k - 0.66, i.e. break-even near oracle-k +0.65.
    # Two points is a thin basis — treat as a go/no-go prior, not a forecast.
    thr_gain = 100 * (o_thr["f1"] - best["f1"])
    k_gain = 100 * (o_k["f1"] - best["f1"])
    est = 1.02 * k_gain - 0.66
    print(f"\n  ranking headroom (oracle-k) = +{k_gain:.2f}  <- the one that predicts")
    print(f"  calibration headroom (oracle-thr) = +{thr_gain:.2f}  <- mostly unreachable")
    print(f"  estimated REAL decoder gain: {est:+.2f} F1")
    if k_gain < 0.65:
        print("  -> Below the +0.65 break-even. Do NOT spend GPU-hours on OOF folds here.")
    else:
        print("  -> Above break-even. Train the OOF folds; gate the re-lock on "
              "bootstrap_paired.py.")


def _top_k(p, g):
    k = int(g.sum())
    pred = np.zeros(len(p), dtype=int)
    if k > 0:
        pred[np.argsort(-p)[:k]] = 1
    return pred


def _selfcheck():
    """Two cases, because the two oracles bound DIFFERENT things.

    (a) Per-doc calibration shift, clean within-doc ranking: every doc ranks its own
        boundaries top, but the probability SCALE differs per doc, so no single global
        threshold fits all of them. Both oracles must beat global by a lot — this is
        the situation where a decoder pays (and is what NoPnx-PA looked like).

    (b) One shared scale, same ranking quality everywhere: nothing for a doc-level rule
        to exploit, so the oracle gap must be small. This is the situation where the
        decoder is a waste of GPU (and is what I expect PA to look like).
    """
    grid = np.arange(0.05, 0.96, 0.01)

    def _global(docs):
        return max((score(docs, lambda p, g, t=t: (p >= t).astype(int)) for t in grid),
                   key=lambda s: s["f1"])["f1"]

    rng = np.random.default_rng(0)
    shifted = []
    for i in range(24):
        n, k = 60, 6
        g = np.zeros(n, dtype=int)
        g[rng.choice(n, k, replace=False)] = 1
        base = 0.10 if i % 2 == 0 else 0.55          # per-doc scale shift
        p = np.where(g == 1, base + 0.30, base) + rng.normal(0, 0.01, n)
        shifted.append((np.clip(p, EPS, 1 - EPS), g))
    g_shift, k_shift = _global(shifted), score(shifted, _top_k)["f1"]
    assert k_shift > g_shift + 0.10, f"oracle-k {k_shift:.4f} must clear global {g_shift:.4f}"

    flat = []
    for _ in range(24):
        n, k = 60, 6
        g = np.zeros(n, dtype=int)
        g[rng.choice(n, k, replace=False)] = 1
        p = np.where(g == 1, 0.80, 0.20) + rng.normal(0, 0.01, n)   # one shared scale
        flat.append((np.clip(p, EPS, 1 - EPS), g))
    g_flat, k_flat = _global(flat), score(flat, _top_k)["f1"]
    assert abs(k_flat - g_flat) < 0.02, f"no headroom expected, got {k_flat - g_flat:+.4f}"

    print(f"selfcheck OK:")
    print(f"  per-doc scale shift : global {g_shift:.4f} -> oracle-k {k_shift:.4f} "
          f"(+{100 * (k_shift - g_shift):.1f})  decoder pays here")
    print(f"  one shared scale    : global {g_flat:.4f} -> oracle-k {k_flat:.4f} "
          f"(+{100 * (k_flat - g_flat):.1f})  decoder is a waste here")


if __name__ == "__main__":
    import sys
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()
