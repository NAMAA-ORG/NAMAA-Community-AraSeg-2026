"""P/R/F1 against decision threshold for a locked decoder.

The question this answers: is there free PRECISION available at flat F1? Our operating
point is recall-heavy on every head, and on 2026-08-02 we sit #2 on open NoPnx-NP tied
at 87.0 on F1 while losing to a system with +0.4 precision. If F1 is flat across a band
of thresholds while precision climbs through it, a higher threshold is a free move on a
board we do not lead.

It is NOT free on a board we DO lead: same F1 to two decimals still means a different
system on the blind set, and a head we are winning by 0.4 is not worth that variance.

    Implementation note: reuses band_mass's loaders and replay rather than re-deriving the decode.
    If the replay does not reproduce the locked test F1 it aborts, because a curve
    around the wrong operating point is worse than no curve.

Usage:
    python thr_curve.py --subtask NoPnx-NP --split test
    python thr_curve.py --subtask NoPnx-PA --split dev --width 0.20
"""
import argparse
import json
from pathlib import Path

import numpy as np

from band_mass import DECODER_DIR, _load_members, replay
from fit_decoder import build_docs
from scoring import _doc_f1, _macro


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subtask", required=True, choices=sorted(DECODER_DIR))
    ap.add_argument("--split", default="test")
    ap.add_argument("--width", type=float, default=0.15, help="+/- around the locked threshold")
    ap.add_argument("--step", type=float, default=0.03)
    ap.add_argument("--cache_dir", default="outputs/prob_cache")
    ap.add_argument("--sat_cache_dir", default="outputs/sat_fullft_caches")
    args = ap.parse_args()

    st = args.subtask.replace("-", "_")
    meta = json.load(open(Path(DECODER_DIR[args.subtask]) / f"decoder_{st}.json"))
    w = np.array([meta["weights"][n] for n in meta["feature_names"]], dtype=np.float64)
    b, thr = meta["bias"], meta["dev_threshold"]
    docs = build_docs(_load_members(args.cache_dir, args.sat_cache_dir,
                                    args.subtask, args.split, meta["members"]))

    def score(t):
        return _macro([_doc_f1(replay(s, w, b, t)[0], g) for s, g in docs.values()])

    lock = score(thr)
    ref = meta.get(args.split)
    if ref:  # a curve around the wrong operating point is worse than no curve
        assert abs(lock["f1"] - ref["f1"]) < 1e-6, \
            f"replay F1 {lock['f1']:.6f} != locked {ref['f1']:.6f}"
        print(f"[gate] replay reproduces locked {args.split} F1 {ref['f1']:.4f} OK")

    print(f"\n{args.subtask} @{thr:.2f}, members={meta['members']}, split={args.split}")
    print(f"\n  {'thr':>5} {'P':>7} {'R':>7} {'F1':>7} {'dF1':>7} {'dP':>7}   rounded")
    for t in np.arange(thr - args.width, thr + args.width + 1e-9, args.step):
        t = float(t)
        if not 0.0 < t < 1.0:
            continue
        s = score(t)
        mark = "  <-- LOCK" if abs(t - thr) < 1e-9 else ""
        # `rounded` is what the leaderboard actually prints. Two systems tied there are
        # separated by something we have NOT confirmed -- possibly unrounded F1, possibly
        # precision. Do not spend a submission assuming you know which.
        print(f"  {t:5.2f} {100*s['precision']:7.2f} {100*s['recall']:7.2f} {100*s['f1']:7.2f} "
              f"{100*(s['f1']-lock['f1']):+7.2f} {100*(s['precision']-lock['precision']):+7.2f}   "
              f"{100*s['precision']:.1f}/{100*s['recall']:.1f}/{100*s['f1']:.1f}{mark}")


if __name__ == "__main__":
    main()
