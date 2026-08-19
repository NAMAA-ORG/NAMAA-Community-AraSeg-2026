#!/usr/bin/env python
"""Threshold-curve diagnostic for a prob ensemble.

Computes each member's dev+test probs once (GPU), prob-averages them, then prints
P/R/F1 across a threshold grid for BOTH splits. Use the DEV curve to pick a legal
threshold (standard model selection); the TEST curve is printed for eyeballing the
P/R trade only — never select on it (that's a leak).

Motivating question: NoPnx-PA is precision-heavy / recall-light vs omar_saqr. Is
the locked thr 0.50 a real dev-F1 peak, or a plateau where a lower thr buys recall
for ~free? This shows the curve so we can tell.

    python sweep_thr.py --subtask NoPnx-PA \
        --members configs/e41_qwen35_9b_nopnx_pa.yaml configs/e18_xlmr_single_nopnxpa_s2.yaml
"""
import argparse
import numpy as np
import torch

from train import Config
from ensemble import _member_probs, _combine
from metrics import macro_f1_at


def curve(docs, grid):
    return [(t, macro_f1_at(docs, t)) for t in grid]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subtask", required=True,
                    choices=["PA", "NoPnx-PA", "NP", "NoPnx-NP"])
    ap.add_argument("--members", required=True, nargs="+")
    ap.add_argument("--combine", choices=["prob", "logit", "rank"], default="prob",
                    help="prob=avg probs (piles mass at 0.50); logit=avg log-odds (de-piles)")
    ap.add_argument("--lo", type=float, default=0.30)
    ap.add_argument("--hi", type=float, default=0.60)
    ap.add_argument("--step", type=float, default=0.01)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfgs = [Config.from_yaml(p) for p in args.members]
    ids = [c.experiment_id for c in cfgs]
    print(f"[sweep] subtask={args.subtask} members={ids} combine={args.combine} device={device}")

    dev = _combine([_member_probs(c, args.subtask, "dev",  device) for c in cfgs], args.combine)
    test = _combine([_member_probs(c, args.subtask, "test", device) for c in cfgs], args.combine)

    grid = np.round(np.arange(args.lo, args.hi + 1e-9, args.step), 4)
    dev_c, test_c = curve(dev, grid), curve(test, grid)

    print(f"\n{'thr':>5} | {'devP':>6} {'devR':>6} {'devF1':>7} | {'tstP':>6} {'tstR':>6} {'tstF1':>7}")
    print("-" * 56)
    for (t, d), (_, s) in zip(dev_c, test_c):
        print(f"{t:>5.2f} | {d['precision']*100:>6.2f} {d['recall']*100:>6.2f} "
              f"{d['f1']*100:>7.2f} | {s['precision']*100:>6.2f} {s['recall']*100:>6.2f} "
              f"{s['f1']*100:>7.2f}")

    best_t, best = max(dev_c, key=lambda x: x[1]["f1"])
    print(f"\n[sweep] DEV-optimal thr={best_t:.2f}  devF1={best['f1']*100:.2f}  "
          f"(this is the only legal pick)")
    # Implementation note: no self-check — pure diagnostic reusing already-tested ensemble/metrics fns.


if __name__ == "__main__":
    main()
