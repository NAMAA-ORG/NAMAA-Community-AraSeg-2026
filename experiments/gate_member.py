"""Does a candidate member earn a place in a locked structural decoder?

Fits the decoder twice -- lock members alone, then lock members + candidate --
IDENTICALLY (same OOF train matrix, same L2, same dev threshold sweep), then runs a
paired document bootstrap on the test delta. Re-lock only if the 95% CI excludes zero.
That gate is what correctly rejected the NP decoder at -0.13 and the zero-shot Naqta
column at +0.26, so it is the thing standing between us and a leaderboard-chasing
re-lock on noise.

CPU-only: drives fit_decoder's numpy internals directly, because its own main()
imports train.Config -> torch, which off-cluster boxes don't have.

Candidate probability sources, auto-detected in this order:
  1. outputs/oof/<cand>_fold{0..4}.json      a member we trained (needs OOF folds)
  2. outputs/oof_sat/<Sub>_fold{f}_sat_ft.pkl
  3. outputs/prob_cache/<Sub>_train_<cand>.pkl   a FROZEN external model
For (3) no folds are needed and none exist: a model that never saw AraSeg train cannot
leak through its own train predictions. Anything we fine-tuned must use (1)/(2).

Usage:
    python gate_member.py --subtask NoPnx-PA --candidate e69
    python gate_member.py --subtask NoPnx-NP --candidate naqta --no-folds
"""
import argparse
import itertools
import json
import pickle
from pathlib import Path

import numpy as np

import fit_decoder as FD

LOCKS = {
    "NoPnx-PA": ("outputs/decoder_nopnx_pa/decoder_NoPnx_PA.json", 0.8718),
    # 2026-07-23: e70 (fine-tuned Naqta) gated in at +0.60, CI [+0.19, +1.01]. Bar is the
    # 6-member lock now, so the next candidate has to beat THAT, not the 5-member 0.8589.
    "NoPnx-NP": ("outputs/decoder_nopnx_np/decoder_NoPnx_NP.json", 0.8649),
}
N_FOLDS = 5

# Models with NO AraSeg-train fine-tuning of ours. For these the train-split cache IS
# an honest OOF substitute -- a model that never saw train cannot leak through its own
# train predictions -- and no folds exist or ever will. Anything we fine-tuned must
# supply real folds; see _cand_train.
FROZEN = {"naqta"}


def load_oof(tag, subtask, tag_us):
    """Union of the per-fold files. Leak check: a doc appearing in two folds means a
    fold trained on data it was meant to predict, which silently inflates everything."""
    seen, out = {}, {}
    for f in range(N_FOLDS):
        if tag == "sat_ft":
            fp = Path("outputs/oof_sat") / f"{tag_us}_fold{f}_sat_ft.pkl"
            items = list(pickle.load(open(fp, "rb")).items())
        else:
            fp = Path("outputs/oof") / f"{tag}_fold{f}.json"
            st = json.load(open(fp))["subtasks"][subtask]
            items = [(d, (np.asarray(p, np.float32), np.asarray(st["gold"][d], np.int64)))
                     for d, p in st["probs"].items()]
        for d, v in items:
            if d in seen:
                raise ValueError(f"LEAK: {tag} doc {d} in folds {seen[d]} and {f}")
            seen[d] = f
            out[d] = v
    return out


def load_split(tag, tag_us, split):
    if tag == "sat_ft":
        fp = Path("outputs/sat_fullft_caches") / f"{tag_us}_{split}_sat_ft.pkl"
    else:
        fp = Path("outputs/prob_cache") / f"{tag_us}_{split}_{tag}.pkl"
    return pickle.load(open(fp, "rb"))


def align(mats):
    ids = sorted(set.intersection(*[set(m) for m in mats]))
    return [{d: m[d] for d in ids} for m in mats]


def fit_and_score(tr, dev, te, l2=1.0, thr_step=0.01):
    tr, dev, te = FD.build_docs(align(tr)), FD.build_docs(align(dev)), FD.build_docs(align(te))
    X, y = [], []
    for s, g in tr.values():                       # teacher-forced gap features
        since, gap, short = 0, [], []
        for i in range(len(g)):
            gap.append(np.log1p(since))
            short.append(float(since < FD._TOO_SHORT))
            since = 0 if g[i] == 1 else since + 1
        X.append(np.column_stack([s, gap, short]))
        y.append(g)
    w, b = FD.fit_logistic(np.vstack(X), np.concatenate(y), l2=l2)
    thr, dev_f1 = max(((float(t), FD.score_docs(dev, w, b, float(t))["f1"])
                       for t in np.arange(0.05, 0.96, thr_step)), key=lambda x: x[1])
    return dict(thr=thr, dev_f1=dev_f1, w=w, b=b,
                test=FD._macro([FD._doc_f1(FD.decode(s, w, b, thr), g) for s, g in te.values()]),
                per_doc={d: FD._doc_f1(FD.decode(s, w, b, thr), g)["f1"] for d, (s, g) in te.items()},
                # Same per-doc F1s on dev, so the delta can be bootstrapped there too.
                # Read these as OPTIMISTIC: each system picked its own thr on this very
                # split. The bias hits both arms, so the DELTA is still informative --
                # but a dev CI is not a second independent confirmation of a test CI.
                dev_per_doc={d: FD._doc_f1(FD.decode(s, w, b, thr), g)["f1"]
                             for d, (s, g) in dev.items()})


def best_dev_subset(systems: dict) -> tuple:
    """Pick the non-empty subset with the highest dev_f1. Never let test peek in here --
    that's what turns a gate into leaderboard chasing. Ties favor the smaller subset."""
    return min(systems, key=lambda subset: (-systems[subset]["dev_f1"], len(subset)))


def bootstrap(a, b, n=10000, seed=0):
    """Paired: the SAME resampled docs score both systems, so per-doc difficulty
    cancels out of the delta instead of dominating its variance."""
    ids = sorted(set(a) & set(b))
    A, B = np.array([a[d] for d in ids]), np.array([b[d] for d in ids])
    idx = np.random.default_rng(seed).integers(0, len(ids), size=(n, len(ids)))
    d = B[idx].mean(1) - A[idx].mean(1)
    return ((B.mean() - A.mean()) * 100, np.percentile(d, 2.5) * 100,
            np.percentile(d, 97.5) * 100, float((d > 0).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subtask", required=True, choices=sorted(LOCKS))
    ap.add_argument("--candidate", required=True, nargs="+",
                    help="one or more member tags, e.g. e69 or e69 e71 "
                         "(every non-empty subset is fit; dev picks the winner)")
    ap.add_argument("--no-folds", action="store_true",
                    help="frozen external model: use its train cache directly (no OOF folds)")
    ap.add_argument("--l2", type=float, default=1.0)
    args = ap.parse_args()

    cfgp, lock_f1 = LOCKS[args.subtask]
    tag_us = args.subtask.replace("-", "_")
    mem = json.load(open(cfgp))["members"]
    print(f"{args.subtask}: lock {mem} @ {lock_f1:.4f}   candidates {args.candidate}")

    base = ([load_oof(m, args.subtask, tag_us) for m in mem],
            [load_split(m, tag_us, "dev") for m in mem],
            [load_split(m, tag_us, "test") for m in mem])
    # Per-candidate source, so a frozen external model (naqta) and fine-tuned members
    # (e69/e75/e76) can be swept together. A tag we fine-tuned MUST use its OOF folds;
    # falling back to its train cache would feed in-sample rows to the stacker and bias
    # the fit toward the candidate. So the fallback only fires when no folds exist at
    # all -- which is exactly the frozen-model case that has none by construction.
    def _cand_train(c):
        if not args.no_folds and Path(f"outputs/oof/{c}_fold0.json").exists():
            return load_oof(c, args.subtask, tag_us)
        # "Has no OOF folds" is NOT the same as "never saw AraSeg train". e85 hit this
        # path on 2026-08-01: a fine-tuned seed-2 run whose folds were never produced.
        # Falling back would have fed its in-sample train rows to a stacker whose other
        # members have honest OOF rows, biasing the fit toward the candidate -- the
        # exact failure run_e72_oof.sh was written to avoid. Only models that never
        # trained on AraSeg qualify, and that is a fact about the model, not something
        # the filesystem can tell us. So: allowlist.
        if c not in FROZEN and not args.no_folds:
            raise SystemExit(
                f"\n{c} has no OOF folds and is not a known frozen model.\n"
                f"  If we fine-tuned it, produce folds first (see slurm/run_e72_oof.sh)\n"
                f"  and re-run. If it truly never saw AraSeg train, add it to FROZEN.\n"
                f"  --no-folds forces the train cache, but read the note above first.")
        print(f"  [{c}] frozen model -> train cache (never trained on AraSeg)")
        return load_split(c, tag_us, "train")

    cand_data = {
        c: (_cand_train(c), load_split(c, tag_us, "dev"), load_split(c, tag_us, "test"))
        for c in args.candidate
    }

    baseline = fit_and_score(*base, l2=args.l2)
    print(f"  {'baseline':14} thr {baseline['thr']:.2f}  dev {baseline['dev_f1']:.4f}  test "
          f"P {baseline['test']['precision']:.4f} R {baseline['test']['recall']:.4f} "
          f"F1 {baseline['test']['f1']:.4f}")

    systems = {}
    for r in range(1, len(args.candidate) + 1):
        for subset in itertools.combinations(args.candidate, r):
            extra = [cand_data[c] for c in subset]
            fit = fit_and_score(*[b + [e[i] for e in extra] for i, b in enumerate(base)],
                                 l2=args.l2)
            systems[subset] = fit
            print(f"  {'+' + '+'.join(subset):14} thr {fit['thr']:.2f}  dev {fit['dev_f1']:.4f}  "
                  f"test P {fit['test']['precision']:.4f} R {fit['test']['recall']:.4f} "
                  f"F1 {fit['test']['f1']:.4f}")

    winner = best_dev_subset(systems)
    print(f"\n  dev-selected subset: +{'+'.join(winner)}  (dev {systems[winner]['dev_f1']:.4f})")

    for split, key in (("dev ", "dev_per_doc"), ("test", "per_doc")):
        d, lo, hi, p = bootstrap(baseline[key], systems[winner][key])
        verdict = ("SIGNIFICANT" if lo > 0 else
                   "NOISE (CI spans 0)" if hi > 0 else "SIGNIFICANTLY WORSE")
        print(f"\n  paired bootstrap [{split}]  delta {d:+.2f} F1  "
              f"95% CI [{lo:+.2f}, {hi:+.2f}]  P(B>A)={p:.3f}  --> {verdict}")
    print("  (dev is threshold-tuned per arm -> optimistic; test is the re-lock decision)")

    names = [f"member:{m}" for m in mem] + [f"member:{c}" for c in winner] + FD._STATIC_NAMES + FD._DYN_NAMES
    print("  fitted weights:")
    for n_, v in sorted(zip(names, systems[winner]["w"]), key=lambda x: -abs(x[1])):
        tag = n_.removeprefix("member:")
        print(f"     {n_:24} {v:+.4f}" + ("  <-- CANDIDATE" if tag in winner else ""))


if __name__ == "__main__":
    main()
