"""Equal-weight test-time ensembling over existing OOF fold checkpoints.

train.py's --holdout-fold path trains N independent checkpoints, one per fold, each
seeing a different (N-1)/N slice of train. Those checkpoints already exist once OOF
folds are generated for a candidate -- averaging their dev/test predictions is a free
ensemble member: same architecture and recipe, N different train subsets, no new
GPU-hours beyond what OOF folding already spent.

Usage:
    python fold_bag.py --config configs/e71_naqta_restore_nopnxpa.yaml \
      --fold-output-pattern outputs/e71_fold{fold} --n-folds 5 \
      --subtask NoPnx-PA --output-dir outputs/candidate_foldbag_e71
"""
import argparse
import dataclasses
import json
import pickle
from pathlib import Path

import numpy as np

from cache_probs import cache_path
from train import Config
from ensemble import _member_probs
from metrics import tune_threshold, macro_f1_at


def _fold_checkpoint(output_dir: str, subtask: str) -> Path:
    return Path(output_dir) / f"best_{subtask.replace('-', '_')}.pt"


def _fold_probs(cfg: Config, fold_output_dir: str, subtask: str, split: str, device: str):
    """One fold's {doc_id: (prob, gold)} on `split`, in original subtask token space."""
    fold_cfg = dataclasses.replace(cfg, output_dir=fold_output_dir)
    if cfg.naqta_restore:
        from naqta_restore import build_restored_docs
        from cache_probs import map_restored_docs
        from restore_then_segment import _docs as _original_docs
        records, owners = build_restored_docs(subtask, split, cfg.naqta_restore_min_p,
                                               comma=cfg.naqta_comma)
        restored = _member_probs(fold_cfg, subtask, split, device, records=records)
        original_gold = {d: g for d, (_, g) in _original_docs(subtask, split).items()}
        return map_restored_docs(restored, owners, original_gold)
    return _member_probs(fold_cfg, subtask, split, device)


def average_folds(docs_per_fold: list) -> dict:
    """Equal-weight mean of per-word probabilities across folds. Every fold must
    cover the same document IDs with the same per-document length -- a missing
    fold silently biases the average toward whichever folds are present, so this
    fails loudly instead."""
    doc_ids = set(docs_per_fold[0])
    for i, docs in enumerate(docs_per_fold[1:], 1):
        assert set(docs) == doc_ids, f"fold {i} covers a different document set"
    out = {}
    for doc_id in doc_ids:
        golds = [docs[doc_id][1] for docs in docs_per_fold]
        for g in golds[1:]:
            assert len(g) == len(golds[0]), f"{doc_id}: fold gold length mismatch"
        probs = np.mean([docs[doc_id][0] for docs in docs_per_fold], axis=0)
        out[doc_id] = (probs, golds[0])
    return out


def write_cache(cache_dir, subtask, split, tag, docs):
    path = cache_path(Path(cache_dir), subtask, split, tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(docs, fh)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="the fold family's training config YAML")
    ap.add_argument("--fold-output-pattern", required=True,
                    help="e.g. outputs/e71_fold{fold}")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--subtask", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache-dir")
    ap.add_argument("--cache-tag")
    args = ap.parse_args()
    if bool(args.cache_dir) != bool(args.cache_tag):
        ap.error("--cache-dir and --cache-tag must be provided together")

    cfg = Config.from_yaml(args.config)
    fold_dirs = [args.fold_output_pattern.format(fold=f) for f in range(args.n_folds)]
    for d in fold_dirs:
        ckpt = _fold_checkpoint(d, args.subtask)
        if not ckpt.exists():
            raise FileNotFoundError(f"fold checkpoint missing: {ckpt}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dev_bag = average_folds([_fold_probs(cfg, d, args.subtask, "dev", args.device)
                              for d in fold_dirs])
    thr, dev_scores = tune_threshold(dev_bag)
    print(f"[fold_bag] {cfg.experiment_id} {args.subtask}: dev thr={thr:.2f} "
          f"F1={dev_scores['f1']:.4f}")

    test_bag = average_folds([_fold_probs(cfg, d, args.subtask, "test", args.device)
                               for d in fold_dirs])
    test_scores = macro_f1_at(test_bag, thr)
    print(f"[fold_bag] {cfg.experiment_id} {args.subtask}: test "
          f"P {test_scores['precision']:.4f} R {test_scores['recall']:.4f} "
          f"F1 {test_scores['f1']:.4f}")
    if args.cache_dir:
        for split, bag in (("dev", dev_bag), ("test", test_bag)):
            print(f"[fold_bag] cache -> "
                  f"{write_cache(args.cache_dir, args.subtask, split, args.cache_tag, bag)}")

    tag = args.subtask.replace("-", "_")
    for split, bag in (("dev", dev_bag), ("test", test_bag)):
        csv_path = out / f"{split}_predictions_{tag}.csv"
        with csv_path.open("w", encoding="utf-8") as fh:
            fh.write("Document ID,Prediction\n")
            for doc_id, (parr, _) in bag.items():
                bits = (parr >= thr).astype(int)
                fh.write(f"{doc_id},{''.join(str(int(b)) for b in bits)}\n")

    summary = {
        "experiment_id": cfg.experiment_id, "subtask": args.subtask,
        "fold_output_dirs": fold_dirs, "n_folds": args.n_folds,
        "dev_threshold": thr, "dev": dev_scores, "test": test_scores,
    }
    (out / "fold_bag_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[fold_bag] wrote predictions and summary -> {out}")


if __name__ == "__main__":
    main()
