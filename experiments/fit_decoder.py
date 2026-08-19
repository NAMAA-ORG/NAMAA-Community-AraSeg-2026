"""Train-only structural decoder over the locked ensemble probabilities (queue B).

The locks score every word independently: a boundary is wherever the combined
probability clears one global threshold. That throws away everything the *sequence*
knows — how far back the last boundary was, whether this word is a local probability
peak, how long the document is. This decoder adds those cues on top of the existing
member probabilities and decodes left-to-right.

Legality (closed track): every learned parameter is fit on OUT-OF-FOLD TRAIN
predictions (`outputs/oof/`, `outputs/oof_sat/`) — the same matrix `fit_oof_stack.py`
fits its stacker on, so no weight ever sees dev or test. Dev is used only to tune the
one decision threshold (ordinary model selection, same as every existing lock).

Shape of the model: an MEMM, not a new network. A logistic model over
[member log-odds | structural features], where one feature — distance since the
*previously decoded* boundary — depends on its own past decisions. Fit with
teacher forcing (gold previous boundary), decode greedily.

    Implementation note: teacher-forced fit + greedy decode is the cheap MEMM. It carries the
    standard exposure-bias ceiling (train sees gold history, inference sees its own).
    Upgrade path if the dev gain is real but small: semi-Markov / Viterbi DP over span
    lengths, same features, ~40 more lines.

Members' dev/test probabilities are read from the generic caches written by
`cache_probs.py` (CPU-only from there on). SaT is just another cached member (`sat_ft`).

Usage:
    python fit_decoder.py --subtask NoPnx-PA \
        --members configs/e41_qwen35_9b_nopnx_pa.yaml configs/e17_xlmr_single_nopnxpa_s1.yaml \
                  configs/e18_xlmr_single_nopnxpa_s2.yaml configs/e11_xlmr_single_nopnxpa_w2.yaml \
                  configs/e38_xlmr_multitask_joint.yaml \
        --oof_dir outputs/oof --sat_oof_dir outputs/oof_sat \
        --cache_dir outputs/prob_cache --sat_cache_dir outputs/sat_fullft_caches \
        --output_dir outputs/decoder_nopnx_pa
"""
import argparse
import json
from pathlib import Path

import numpy as np

from scoring import _doc_f1, _macro

_EPS = 1e-6
# Structural features appended after the M member log-odds. The last two are the
# dynamic ones (recomputed from the decoder's own history during decoding).
_STATIC_NAMES = ["mean_logodds", "local_max", "local_rank", "rel_pos", "log_doclen", "doc_density"]
_DYN_NAMES = ["log_gap", "gap_too_short"]
_TOO_SHORT = 3          # words; a span shorter than this is implausible in AraSeg
_LOCAL_WIN = 3          # +/- words for the local-peak features


def _logodds(p):
    p = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def _static_feats(probs):
    """probs: (N, M) member probabilities for one document -> (N, M + len(_STATIC_NAMES))."""
    lo = _logodds(probs)                      # (N, M)
    n = lo.shape[0]
    mean_lo = lo.mean(axis=1)
    mean_p = probs.mean(axis=1)

    local_max = np.zeros(n)
    local_rank = np.zeros(n)
    for i in range(n):
        a, b = max(0, i - _LOCAL_WIN), min(n, i + _LOCAL_WIN + 1)
        win = mean_p[a:b]
        local_max[i] = float(mean_p[i] >= win.max())
        local_rank[i] = float((win < mean_p[i]).sum()) / max(len(win) - 1, 1)

    rel_pos = np.arange(n) / max(n - 1, 1)
    log_doclen = np.full(n, np.log1p(n))
    doc_density = np.full(n, mean_p.mean())   # boundary-rate prior, from preds only
    return np.column_stack([lo, mean_lo, local_max, local_rank, rel_pos, log_doclen, doc_density])


def _dyn_feats(gaps):
    """gaps: (N,) words since the previous boundary -> (N, 2)."""
    gaps = np.asarray(gaps, dtype=np.float64)
    return np.column_stack([np.log1p(gaps), (gaps < _TOO_SHORT).astype(np.float64)])


def _gold_gaps(gold):
    """Teacher-forced gap feature: distance back to the previous GOLD boundary."""
    gaps, since = np.empty(len(gold)), 0
    for i, g in enumerate(gold):
        gaps[i] = since
        since = 0 if g == 1 else since + 1
    return gaps


def fit_logistic(X, y, l2=1.0, iters=1500, lr=0.5):
    """Plain L2 logistic regression, full-batch GD on standardised features.
    Dependency-free (no sklearn on the offline cluster venv). Returns raw (w, b)."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mu, sd = X.mean(0), X.std(0) + _EPS
    Xs = (X - mu) / sd
    n, m = Xs.shape
    w, b = np.zeros(m), 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(Xs @ w + b)))
        err = p - y
        w -= lr * (Xs.T @ err / n + l2 * w / n)
        b -= lr * err.mean()
    return w / sd, float(b - w @ (mu / sd))


def decode(static, w, b, thr):
    """Greedy left-to-right decode of one document. The static part of the score is
    precomputed; only the two gap features move as boundaries get committed."""
    n_static = static.shape[1]
    z_static = static @ w[:n_static] + b
    w_gap, w_short = w[n_static], w[n_static + 1]
    preds = np.zeros(static.shape[0], dtype=np.int64)
    since = 0
    for i in range(static.shape[0]):
        z = z_static[i] + w_gap * np.log1p(since) + w_short * (since < _TOO_SHORT)
        if 1.0 / (1.0 + np.exp(-z)) >= thr:
            preds[i] = 1
            since = 0
        else:
            since += 1
    return preds


def score_docs(docs, w, b, thr):
    """docs: {doc_id: (static_feats, gold)} -> macro P/R/F1 of the greedy decode."""
    return _macro([_doc_f1(decode(s, w, b, thr), g) for s, g in docs.values()])


def build_docs(members):
    """members: [{doc_id: (prob, gold)}] -> {doc_id: (static_feats, gold)}, member order fixed."""
    out = {}
    for doc_id, (_, gold) in members[0].items():
        probs = np.column_stack([m[doc_id][0] for m in members])
        out[doc_id] = (_static_feats(probs), np.asarray(gold, dtype=np.int64))
    return out


def main():
    from train import Config
    from fit_oof_stack import _load_member_oof, _load_sat_oof, _load_sat_split
    from cache_probs import load_cached_member

    ap = argparse.ArgumentParser()
    ap.add_argument("--subtask", required=True, choices=["PA", "NoPnx-PA", "NP", "NoPnx-NP"])
    ap.add_argument("--members", required=True, nargs="+", help="Member config YAMLs (lock order)")
    ap.add_argument("--oof_dir", default="outputs/oof")
    ap.add_argument("--cache_dir", default="outputs/prob_cache",
                    help="Generic dev/test caches from cache_probs.py")
    ap.add_argument("--sat_oof_dir", default=None, help="Add sat_ft as a member (OOF folds)")
    ap.add_argument("--sat_cache_dir", default=None, help="sat_ft dev/test caches; required with --sat_oof_dir")
    ap.add_argument("--frozen-member", dest="frozen_member", nargs="*", default=[],
                    help="Cached members with no config YAML and no OOF folds -- a frozen "
                         "external model such as `naqta`. Its TRAIN cache is used where "
                         "other members use OOF rows, which is honest here and only here: "
                         "a model that never saw AraSeg train cannot leak through its own "
                         "train predictions. Never pass something we fine-tuned.")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--fold_seed", type=int, default=42)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--thr_step", type=float, default=0.01)
    ap.add_argument("--force-threshold", dest="force_threshold", type=float, default=None,
                    help="Decode at THIS threshold instead of the dev-tuned one. The dev "
                         "sweep still runs and is still reported, so the deviation is "
                         "visible. Only defensible where F1 is flat and you are buying "
                         "something else with the slack -- e.g. precision, on a board we "
                         "do not lead. Never on a head we are winning: same rounded F1 is "
                         "still a different system on blind. See thr_curve.py.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--test-split", dest="test_split", default="test",
                    help="Split to decode for the submission: 'test' (public) or 'blind' "
                         "(Testing Phase). Weights/threshold are still fit on OOF train + dev "
                         "(deterministic, reproduces the locked decoder_*.json). Requires the "
                         "member caches for this split to exist (cache_probs.py --splits <split>).")
    args = ap.parse_args()
    if bool(args.sat_oof_dir) != bool(args.sat_cache_dir):
        ap.error("--sat_oof_dir and --sat_cache_dir must be given together")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfgs = [Config.from_yaml(p) for p in args.members]
    ids = [c.experiment_id for c in cfgs]
    st_us = args.subtask.replace("-", "_")
    print(f"[decoder] subtask={args.subtask} members={ids}{' + sat_ft' if args.sat_oof_dir else ''} (CPU)")

    # ── Fit on OOF TRAIN (closed-legal: no learned weight sees dev/test) ─────
    oof = [_load_member_oof(c.experiment_id, args.subtask, Path(args.oof_dir),
                            args.n_folds, args.fold_seed) for c in cfgs]
    if args.sat_oof_dir:
        oof.append(_load_sat_oof(Path(args.sat_oof_dir), args.subtask, args.n_folds, args.fold_seed))
        ids.append("sat_ft")
    # Frozen members go last, and in the SAME position in split_members() below --
    # the fitted weight vector is positional, so a reordering here silently pairs
    # every member with the wrong coefficient.
    for tag in args.frozen_member:
        oof.append(load_cached_member(args.cache_dir, args.subtask, "train", tag))
        ids.append(tag)

    train_docs = build_docs(oof)
    X = np.vstack([np.hstack([s, _dyn_feats(_gold_gaps(g))]) for s, g in train_docs.values()])
    y = np.concatenate([g for _, g in train_docs.values()])
    w, b = fit_logistic(X, y, l2=args.l2)
    names = [f"logodds_{m}" for m in ids] + _STATIC_NAMES + _DYN_NAMES
    print(f"[decoder] fit on {X.shape[0]} OOF train words, {X.shape[1]} features")
    for nm, wi in zip(names, w):
        print(f"    {nm:>22} {wi:+.4f}")
    print(f"    {'bias':>22} {b:+.4f}")

    # ── Tune the one threshold on dev; report test at that threshold ─────────
    def split_members(split):
        ms = [load_cached_member(args.cache_dir, args.subtask, split, c.experiment_id) for c in cfgs]
        if args.sat_cache_dir:
            ms.append(_load_sat_split(Path(args.sat_cache_dir), args.subtask, split))
        for tag in args.frozen_member:                      # same order as the OOF list
            ms.append(load_cached_member(args.cache_dir, args.subtask, split, tag))
        return ms

    dev_docs = build_docs(split_members("dev"))
    grid = np.arange(0.05, 0.96, args.thr_step)
    dev_scores = [(float(t), score_docs(dev_docs, w, b, float(t))) for t in grid]
    dev_thr, dev_best = max(dev_scores, key=lambda kv: kv[1]["f1"])
    print(f"[decoder] dev tuned@{dev_thr:.2f}: P={dev_best['precision']:.4f} "
          f"R={dev_best['recall']:.4f} F1={dev_best['f1']:.4f}")

    if args.force_threshold is not None:
        forced = float(args.force_threshold)
        at_forced = score_docs(dev_docs, w, b, forced)
        print(f"[decoder] OVERRIDE: decoding @{forced:.2f} instead of the dev-tuned "
              f"{dev_thr:.2f}\n"
              f"[decoder]   dev @{forced:.2f}: P={at_forced['precision']:.4f} "
              f"R={at_forced['recall']:.4f} F1={at_forced['f1']:.4f} "
              f"(dF1 {100*(at_forced['f1']-dev_best['f1']):+.2f}, "
              f"dP {100*(at_forced['precision']-dev_best['precision']):+.2f})")
        dev_thr, dev_best = forced, at_forced

    test_docs = build_docs(split_members(args.test_split))
    test_preds = {d: decode(s, w, b, dev_thr) for d, (s, _) in test_docs.items()}
    all_gold = np.concatenate([g for _, g in test_docs.values()])
    test_score = None
    if int(all_gold.sum()) > 0:
        test_score = _macro([_doc_f1(test_preds[d], g) for d, (_, g) in test_docs.items()])
        print(f"[decoder] TEST @{dev_thr:.2f}: P={test_score['precision']:.4f} "
              f"R={test_score['recall']:.4f} F1={test_score['f1']:.4f}")
    else:
        print("[decoder] test labels hidden — predictions saved only")

    pred_path = out / f"test_predictions_{st_us}.csv"
    with open(pred_path, "w") as f:
        f.write("Document ID,Prediction\n")
        for doc_id, preds in test_preds.items():
            f.write(f"{doc_id},{''.join(map(str, preds))}\n")
    print(f"[decoder] predictions -> {pred_path}")
    print(f"[decoder] gate: run bootstrap_paired.py against the lock CSV before re-locking.")

    with open(out / f"decoder_{st_us}.json", "w") as f:
        json.dump({"subtask": args.subtask, "combine": "oof_decoder", "members": ids,
                   "member_configs": args.members, "with_sat": bool(args.sat_oof_dir),
                   "n_folds": args.n_folds, "fold_seed": args.fold_seed, "l2": args.l2,
                   "feature_names": names,
                   "weights": dict(zip(names, [float(x) for x in w])), "bias": b,
                   "dev_threshold": dev_thr, "dev": dev_best, "test": test_score}, f, indent=2)


def _selfcheck():
    """Synthetic: a member that spikes on every gold boundary but ALSO double-fires one
    word later. No global threshold can remove those duplicates; the gap feature can.
    The decoder must beat the best-threshold baseline on the same probabilities."""
    rng = np.random.default_rng(0)
    docs = {}
    for d in range(40):
        n = 200
        gold = np.zeros(n, dtype=np.int64)
        gold[np.arange(9, n, 10)] = 1                      # a boundary every 10 words
        prob = rng.uniform(0.01, 0.15, n)
        prob[gold == 1] = rng.uniform(0.80, 0.95, gold.sum())
        echo = np.clip(np.flatnonzero(gold) + 1, 0, n - 1)  # spurious echo one word later
        prob[echo] = rng.uniform(0.75, 0.90, len(echo))
        docs[f"d{d}"] = (prob, gold)

    train = build_docs([docs])          # one member
    X = np.vstack([np.hstack([s, _dyn_feats(_gold_gaps(g))]) for s, g in train.values()])
    y = np.concatenate([g for _, g in train.values()])
    w, b = fit_logistic(X, y)

    grid = np.arange(0.05, 0.96, 0.01)
    base = max(_macro([_doc_f1((p >= t).astype(int), g) for p, g in docs.values()])["f1"]
               for t in grid)
    dec = max(score_docs(train, w, b, float(t))["f1"] for t in grid)
    assert dec > base + 0.05, f"decoder {dec:.4f} did not beat thresholding {base:.4f}"
    # the gap features must be what does it: short gaps get pushed down
    assert w[-1] < 0, f"gap_too_short weight should suppress boundaries, got {w[-1]:+.4f}"
    print(f"OK: threshold-only F1={base:.4f} -> decoder F1={dec:.4f} "
          f"(gap_too_short weight {w[-1]:+.3f} kills the echo)")


if __name__ == "__main__":
    import sys
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()
