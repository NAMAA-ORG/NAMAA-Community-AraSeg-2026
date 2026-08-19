"""Closed-legal stacked ensemble: fit the meta-learner on OUT-OF-FOLD TRAIN
predictions instead of dev.

The only difference from `ensemble.py --combine stack` is *where the stacker's
weights come from*. Here they are fit on `fit_stacker(X_oof_train, y_train)`,
where X_oof_train is assembled from the per-fold held-out predictions written by
`train.py --holdout-fold` (union of folds = every train doc, each predicted by a
model that never saw it). Applying those weights to the full-train members' dev
and test predictions is then closed-legal: no learned parameter ever touched
dev/test. The decision threshold is still tuned on dev (one scalar — standard
model selection, same as every prob/logit lock).

Usage:
    python fit_oof_stack.py --subtask NoPnx-PA \
        --members configs/e41_qwen35_9b_nopnx_pa.yaml configs/e17_....yaml ... \
        --oof_dir outputs/oof --n_folds 5 --output_dir outputs/oof_stack_nopnxpa
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from train import Config
from metrics import tune_threshold, macro_f1_at
from combiners import fit_stacker
from ensemble import _member_probs, _pool, _apply_stack_per_doc
from oof import fold_map


def _load_member_oof(exp_id: str, subtask: str, oof_dir: Path, n_folds: int, seed: int):
    """Assemble one member's OOF train preds: {doc_id: (prob, gold)} over all
    train docs, each taken from the fold-file where that doc was held out."""
    fm = fold_map(subtask, n_folds, seed)          # doc_id -> fold
    probs, gold = {}, {}
    for f in range(n_folds):
        fp = oof_dir / f"{exp_id}_fold{f}.json"
        if not fp.exists():
            raise FileNotFoundError(f"missing OOF file {fp} — did fold {f} for {exp_id} finish?")
        d = json.load(open(fp))
        st = d["subtasks"].get(subtask)
        if st is None:
            raise KeyError(f"{fp} has no subtask {subtask} (members trained on wrong head?)")
        for doc_id, parr in st["probs"].items():
            if fm.get(doc_id) != f:
                # doc predicted by a fold that didn't hold it out => leakage
                raise ValueError(f"{exp_id} fold{f}: doc {doc_id} not held out in this fold")
            probs[doc_id] = np.asarray(parr, dtype=np.float32)
            gold[doc_id] = np.asarray(st["gold"][doc_id], dtype=np.int64)
    missing = set(fm) - set(probs)
    if missing:
        raise ValueError(f"{exp_id}: {len(missing)} train docs have no OOF pred (e.g. {list(missing)[:3]})")
    return {d: (probs[d], gold[d]) for d in probs}


def _load_sat_oof(sat_oof_dir: Path, subtask: str, n_folds: int, seed: int):
    """SaT's OOF as a member: union of per-fold {doc_id:(prob,gold)} pkls written by
    sat_finetune.py --holdout-fold. Same leak check as _load_member_oof: each doc must
    come from the fold that held it out. SaT stays in its native cache format (its
    wtpsplit tokenizer never goes through train.py), so no json round-trip."""
    fm = fold_map(subtask, n_folds, seed)
    st_us = subtask.replace("-", "_")
    out = {}
    for f in range(n_folds):
        fp = sat_oof_dir / f"{st_us}_fold{f}_sat_ft.pkl"
        if not fp.exists():
            raise FileNotFoundError(f"missing SaT OOF fold {fp} — did fold {f} finish?")
        for doc_id, (prob, gold) in pickle.load(open(fp, "rb")).items():
            if fm.get(doc_id) != f:
                raise ValueError(f"sat_ft fold{f}: doc {doc_id} not held out in this fold")
            out[doc_id] = (np.asarray(prob, np.float32), np.asarray(gold, np.int64))
    missing = set(fm) - set(out)
    if missing:
        raise ValueError(f"sat_ft: {len(missing)} train docs have no OOF pred (e.g. {list(missing)[:3]})")
    return out


def _load_sat_split(sat_cache_dir: Path, subtask: str, split: str):
    fp = sat_cache_dir / f"{subtask.replace('-', '_')}_{split}_sat_ft.pkl"
    if not fp.exists():
        raise FileNotFoundError(f"missing SaT {split} cache {fp}")
    return pickle.load(open(fp, "rb"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subtask", required=True, choices=["PA", "NoPnx-PA", "NP", "NoPnx-NP"])
    ap.add_argument("--members", required=True, nargs="+", help="Member config YAML paths")
    ap.add_argument("--oof_dir", required=True, help="Dir holding {exp_id}_fold{f}.json")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--fold_seed", type=int, default=42)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--thr_step", type=float, default=0.01)
    ap.add_argument("--stack_l2", type=float, default=1.0)
    ap.add_argument("--threshold_policy", choices=["dev_tuned", "fixed"], default="dev_tuned")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--test-split", dest="test_split", default="test",
                    help="Split to write predictions on: 'test' (public) or 'blind' "
                         "(Testing Phase). Stacker is still fit on OOF train; blind labels "
                         "are hidden so scoring is skipped and only the CSV is written.")
    ap.add_argument("--sat_oof_dir", default=None,
                    help="Add SaT as a member: dir with {subtask}_fold{f}_sat_ft.pkl OOF folds.")
    ap.add_argument("--sat_cache_dir", default=None,
                    help="SaT dev/test caches ({subtask}_{dev,test}_sat_ft.pkl); required with --sat_oof_dir.")
    args = ap.parse_args()
    if bool(args.sat_oof_dir) != bool(args.sat_cache_dir):
        ap.error("--sat_oof_dir and --sat_cache_dir must be given together")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    oof_dir = Path(args.oof_dir)
    cfgs = [Config.from_yaml(p) for p in args.members]
    ids = [c.experiment_id for c in cfgs]
    print(f"[oof-stack] subtask={args.subtask} members={ids} device={device}")

    # ── Fit stacker on OOF train (closed-legal) ──────────────────────────────
    oof_members = [_load_member_oof(c.experiment_id, args.subtask, oof_dir,
                                    args.n_folds, args.fold_seed) for c in cfgs]
    if args.sat_oof_dir:
        oof_members.append(_load_sat_oof(Path(args.sat_oof_dir), args.subtask,
                                         args.n_folds, args.fold_seed))
        ids.append("sat_ft")
    keys0 = set(oof_members[0])
    for mid, m in zip(ids, oof_members):
        if set(m) != keys0:
            raise ValueError(f"member {mid} OOF doc set differs from {ids[0]}")
    X_oof, y = _pool(oof_members)
    w, b = fit_stacker(X_oof, y, l2=args.stack_l2)
    print(f"[oof-stack] fit on {X_oof.shape[0]} OOF train words; "
          f"weights={dict(zip(ids, np.round(w, 4)))} bias={b:.4f}")

    # ── Apply to full-train members' dev/test (recompute from checkpoints) ────
    dev_members = [_member_probs(c, args.subtask, "dev", device) for c in cfgs]
    if args.sat_oof_dir:
        dev_members.append(_load_sat_split(Path(args.sat_cache_dir), args.subtask, "dev"))
    dev_docs = _apply_stack_per_doc(dev_members, w, b)
    grid = np.arange(0.05, 0.96, args.thr_step)
    dev_thr, dev_best = tune_threshold(dev_docs, grid)
    write_thr = dev_thr if args.threshold_policy == "dev_tuned" else args.threshold
    print(f"[oof-stack] dev tuned@{dev_thr:.2f}: P={dev_best['precision']:.4f} "
          f"R={dev_best['recall']:.4f} F1={dev_best['f1']:.4f}")

    test_members = [_member_probs(c, args.subtask, args.test_split, device) for c in cfgs]
    if args.sat_oof_dir:
        test_members.append(_load_sat_split(Path(args.sat_cache_dir), args.subtask, args.test_split))
    test_docs = _apply_stack_per_doc(test_members, w, b)
    all_gold = np.concatenate([g for _, g in test_docs.values()])
    test_result = None
    if int(all_gold.sum()) > 0:
        s = macro_f1_at(test_docs, write_thr)
        test_result = s
        print(f"[oof-stack] TEST @{write_thr:.2f}: P={s['precision']:.4f} "
              f"R={s['recall']:.4f} F1={s['f1']:.4f}")
    else:
        print("[oof-stack] test labels hidden — predictions saved only")

    # ── mlflow: track the stack (submission-level, not just per-model) ───────
    # Optional — off-cluster (RunPod) has no mlflow (numpy-2.x deps break the
    # pinned stack). Skip logging on a miss but STILL write the CSV below.
    try:
        import mlflow
    except ModuleNotFoundError:
        mlflow = None
    if mlflow is not None:
        mlflow.set_experiment("AraSeg-Per-Dataset")
        with mlflow.start_run(run_name=f"oof-stack-{args.subtask.replace('-', '_')}{'-sat' if args.sat_oof_dir else ''}"):
            mlflow.log_params({"dataset": args.subtask, "combine": "oof_stack",
                               "members": ",".join(ids), "n_folds": args.n_folds,
                               "stack_l2": args.stack_l2, "with_sat": bool(args.sat_oof_dir)})
            mlflow.log_metrics({"dev_f1": dev_best["f1"], "dev_threshold": float(dev_thr),
                                "write_threshold": float(write_thr), "stack_bias": float(b),
                                **{f"stack_w_{mid}": float(wi) for mid, wi in zip(ids, w)}})
            if test_result is not None:
                mlflow.log_metrics({"test_f1": test_result["f1"], "test_precision": test_result["precision"],
                                    "test_recall": test_result["recall"]})

    # ── Write submission CSV + summary ───────────────────────────────────────
    st_us = args.subtask.replace("-", "_")
    pred_path = out / f"test_predictions_{st_us}.csv"
    with open(pred_path, "w") as f:
        f.write("Document ID,Prediction\n")
        for doc_id, (parr, _) in test_docs.items():
            preds = (parr >= write_thr).astype(int)
            f.write(f"{doc_id},{''.join(map(str, preds))}\n")
    print(f"[oof-stack] predictions → {pred_path}")

    with open(out / f"ensemble_{st_us}.json", "w") as f:
        json.dump({
            "subtask": args.subtask, "combine": "oof_stack",
            "members": ids, "member_configs": args.members,
            "n_folds": args.n_folds, "fold_seed": args.fold_seed, "stack_l2": args.stack_l2,
            "stack_weights": dict(zip(ids, [float(x) for x in w])), "stack_bias": float(b),
            "threshold_policy": args.threshold_policy,
            "dev_threshold": float(dev_thr), "write_threshold": float(write_thr),
            "dev": dev_best, "test": test_result,
        }, f, indent=2)


if __name__ == "__main__":
    main()
