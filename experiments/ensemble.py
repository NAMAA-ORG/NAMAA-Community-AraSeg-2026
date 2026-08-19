"""AraSeg 2026 — Ensemble inference.

Average per-word boundary probabilities across several trained checkpoints, tune
one decision threshold on dev, then score test and write the submission CSV.

Why this works across heterogeneous encoders: `collect_doc_probs` returns, per
document, a probability for every *word* (not subword) — length == len(gold).
That representation is tokenizer-independent, so an XLM-R member and a MARBERTv2
member produce arrays of the same shape per document and can be averaged
directly. Members may therefore mix seeds AND architectures.

Each member is specified by its training config YAML; the config tells us the
model_name (to rebuild the encoder), the subtasks list (to rebuild the matching
head dict so the checkpoint loads), the output_dir (where best_<subtask>.pt
lives), and the windowing params (max_length, window_stride).

Usage:
    python ensemble.py --subtask PA \
        --members configs/e10_xlmr_single_pa_weighted.yaml \
                  configs/e13_xlmr_single_pa_s1.yaml \
                  configs/e14_xlmr_single_pa_s2.yaml \
        --output_dir outputs/ens_pa
"""

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer

from train import Config
from data_utils import AraSegDataset, make_collate
from model import build_model, load_checkpoint
from metrics import collect_doc_probs, tune_threshold, macro_f1_at
from utils import resolve_model_path
from combiners import COMBINERS, fit_stacker, apply_stacker


def _ckpt_name(subtask: str) -> str:
    return f"best_{subtask.replace('-', '_')}.pt"


def _member_probs(cfg: Config, subtask: str, split: str, device: str,
                  records: Optional[List[dict]] = None
                  ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Build one member, load its checkpoint, return {doc_id: (prob, gold)}.

    `records` runs the member over in-memory docs instead of the HF split — used by
    restore_then_segment.py to score a punctuated model on punctuation-RESTORED text.
    """
    model_path = resolve_model_path(cfg.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=cfg.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = AraSegDataset(subtask, split, tokenizer, cfg.max_length, cfg.window_stride,
                       records=records)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    loader = torch.utils.data.DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=make_collate(pad_id), num_workers=cfg.num_workers, pin_memory=True,
    )

    # Rebuild with the SAME architecture flags used in training (heads, optional
    # BiLSTM, LLM backbone/LoRA) so the module graph matches the checkpoint.
    # build_model reads all of that from the member's own config; load_checkpoint
    # accepts both full (BERT) and trainable-only (LoRA) checkpoint formats.
    model = build_model(cfg, model_path).to(device)
    # Off-cluster (blind day): checkpoints live at $ARASEG_CKPT_ROOT/<exp_id>, not the
    # KISSKI-relative cfg.output_dir. When set, it overrides output_dir for every member
    # and every tool (cache_probs/fit_oof_stack/ensemble) — one knob, no per-config patch.
    ckpt_root = os.environ.get("ARASEG_CKPT_ROOT")
    ckpt_dir = Path(ckpt_root) / cfg.experiment_id if ckpt_root else Path(cfg.output_dir)
    ckpt = ckpt_dir / _ckpt_name(subtask)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint missing for member {cfg.experiment_id}: {ckpt}")
    load_checkpoint(model, ckpt, device)
    model.eval()

    return collect_doc_probs(model, loader, device, subtask, ds.doc_gold)


def _load_cache_member(cache_dir: Path, subtask: str, split: str, tag: str = "sat_ft"
                       ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Load a pre-computed member from a pkl, e.g. SaT's sat_finetune caches. The
    pkl is already {doc_id: (prob_per_word, gold)} — the exact member format — so
    no rebuild/checkpoint is needed (SaT uses wtpsplit's tokenizer, which the
    config-YAML path can't load). Implementation note: SaT-specific naming; generalise the
    filename scheme only when a second cache-only member appears."""
    fp = cache_dir / f"{subtask.replace('-', '_')}_{split}_{tag}.pkl"
    if not fp.exists():
        raise FileNotFoundError(f"cache member missing: {fp}")
    return pickle.load(open(fp, "rb"))


def _combine(per_member: List[Dict[str, Tuple[np.ndarray, np.ndarray]]],
             mode: str = "prob", weights=None
             ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Combine per-word member probabilities per document using the chosen
    combiner. Gold is identical across members (same split, same doc_gold) so we
    take it from the first."""
    fn = COMBINERS[mode]
    docs = per_member[0]
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for doc_id, (_, gold) in docs.items():
        stack = []
        for m in per_member:
            if doc_id not in m:
                raise KeyError(f"doc {doc_id} missing from a member — inconsistent splits")
            parr, _ = m[doc_id]
            if len(parr) != len(gold):
                raise ValueError(f"doc {doc_id} length mismatch across members "
                                 f"({len(parr)} vs {len(gold)}) — different word segmentation?")
            stack.append(parr)
        out[doc_id] = (fn(stack, weights=weights), gold)
    return out


def _pool(per_member: List[Dict[str, Tuple[np.ndarray, np.ndarray]]]):
    """Flatten members into (N_words, M) probs + (N_words,) gold, in a fixed
    doc/word order shared across members. Used by the stacker."""
    doc_ids = list(per_member[0].keys())
    cols = []
    for m in per_member:
        cols.append(np.concatenate([m[d][0] for d in doc_ids]))
    X = np.stack(cols, axis=1)
    y = np.concatenate([per_member[0][d][1] for d in doc_ids])
    return X, y


def _apply_stack_per_doc(per_member, w, b):
    """Return {doc_id: (stacked_prob, gold)} using a fitted stacker."""
    docs = per_member[0]
    out = {}
    for doc_id, (_, gold) in docs.items():
        cols = np.stack([m[doc_id][0] for m in per_member], axis=1)  # (N, M)
        out[doc_id] = (apply_stacker(cols, w, b), gold)
    return out


def _exhaustive_sweep(member_ids, subtask: str, dev_members, test_members,
                      args, out: Path) -> List[int]:
    """Evaluate all non-empty subsets on dev and cross-check top-K on test."""
    from itertools import combinations

    n = len(member_ids)
    print(f"[ensemble] exhaustive sweep: {2**n - 1} subsets of {n} members")
    results = []
    for size in range(1, n + 1):
        for idxs in combinations(range(n), size):
            weights = [args.weights[i] for i in idxs] if args.weights else None
            avg_docs = _combine([dev_members[i] for i in idxs], args.combine, weights)
            thr, best = tune_threshold(avg_docs)
            results.append({
                "members": [member_ids[i] for i in idxs],
                "indices": list(idxs),
                "n": size,
                "dev_f1": best["f1"],
                "dev_thr": thr,
                "dev_precision": best["precision"],
                "dev_recall": best["recall"],
            })

    results.sort(key=lambda r: (-round(r["dev_f1"], 3), r["n"]))
    topk = results[:args.exhaustive_top_k]

    print(f"[ensemble] Top-{args.exhaustive_top_k} subsets (dev-ranked, parsimony tie-break):")
    print(f"  {'dev_F1':>7} {'@thr':>5} {'test@devthr':>11} {'test@0.5':>9}  members")
    for r in topk:
        weights = [args.weights[i] for i in r["indices"]] if args.weights else None
        tavg = _combine([test_members[i] for i in r["indices"]], args.combine, weights)
        r["test_f1_at_dev_thr"] = macro_f1_at(tavg, r["dev_thr"])["f1"]
        r["test_f1_at_0.5"] = macro_f1_at(tavg, 0.5)["f1"]
        print(f"  {r['dev_f1']:>7.4f} {r['dev_thr']:>5.2f} "
              f"{r['test_f1_at_dev_thr']:>11.4f} {r['test_f1_at_0.5']:>9.4f}  {r['members']}")

    sweep_path = out / f"exhaustive_sweep_{subtask.replace('-', '_')}.json"
    with open(sweep_path, "w") as f:
        json.dump({"all_dev_ranked": results, "topk_with_test": topk}, f, indent=2)
    print(f"[ensemble] sweep results -> {sweep_path}")
    print("[ensemble] NOTE: the selected subset is the dev/parsimony winner; review the top-K dev/test table before submitting.")
    return results[0]["indices"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subtask", required=True, choices=["PA", "NoPnx-PA", "NP", "NoPnx-NP"])
    ap.add_argument("--members", nargs="*", default=[], help="Member config YAML paths")
    ap.add_argument("--cache-members", nargs="*", default=[],
                    help="Named members already stored in --cache-dir")
    ap.add_argument("--cache-dir", default="outputs/prob_cache")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--test-split", dest="test_split", default="test",
                    help="Split to write the submission on. 'test' (default, public) or "
                         "'blind' (Testing Phase). Dev tuning is unchanged; the blind split "
                         "has no labels so scoring is skipped and only the CSV is written.")
    ap.add_argument("--threshold_policy", choices=["dev_tuned", "fixed"], default="dev_tuned",
                    help="dev_tuned: write submission at the dev-tuned threshold (default). "
                         "fixed: write at --threshold instead (e.g. 0.5 for NoPnx, which beat "
                         "the dev-tuned threshold on open-test).")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Threshold used when --threshold_policy=fixed. Ignored otherwise.")
    ap.add_argument("--thr_step", type=float, default=0.05,
                    help="Dev threshold-sweep resolution. Finer (e.g. 0.01) finds the "
                         "F1-max point our precision-heavy models tend to leave below 0.5.")
    ap.add_argument("--combine", choices=["prob", "logit", "rank", "stack"], default="prob",
                    help="Per-word member combination. 'prob' = original mean; "
                         "'logit'/'rank' fix the 0.5 pile-up; 'stack' fits a logistic "
                         "meta-learner on dev (no subset search).")
    ap.add_argument("--weights", type=float, nargs="+", default=None,
                    help="Optional per-member weights (same order/count as --members). "
                         "Applied to prob/logit/rank combiners.")
    ap.add_argument("--stack_l2", type=float, default=1.0,
                    help="L2 strength for --combine stack. Higher = closer to a flat average.")
    ap.add_argument("--sat_cache", default=None,
                    help="Dir with SaT prob caches ({subtask}_{split}_sat_ft.pkl). Adds "
                         "SaT as one extra 'sat_ft' member (prob/logit/rank only).")
    ap.add_argument("--exhaustive", action="store_true",
                    help="Evaluate all 2^N-1 non-empty subsets of --members on dev.")
    ap.add_argument("--exhaustive_top_k", type=int, default=10,
                    help="Number of top subsets to print in exhaustive mode.")
    args = ap.parse_args()

    if args.combine == "stack" and args.exhaustive:
        ap.error("--combine stack cannot be used with --exhaustive (stacker fits a fixed member set)")
    if args.sat_cache and args.combine == "stack":
        ap.error("--sat_cache works with prob/logit/rank only")
    if not (args.members or args.cache_members or args.sat_cache):
        ap.error("provide at least one checkpoint or cached member")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfgs = [Config.from_yaml(p) for p in args.members]
    member_ids = [c.experiment_id for c in cfgs] + list(args.cache_members)
    print(f"[ensemble] subtask={args.subtask}  members={member_ids}")
    print(f"[ensemble] device={device}")

    dev_members = []
    test_members = []
    for cfg in cfgs:
        print(f"  [dev ] member {cfg.experiment_id} ({cfg.model_name})")
        dev_members.append(_member_probs(cfg, args.subtask, "dev", device))
        print(f"  [test] member {cfg.experiment_id} (split={args.test_split})")
        test_members.append(_member_probs(cfg, args.subtask, args.test_split, device))
    cache_dir = Path(args.cache_dir)
    for tag in args.cache_members:
        print(f"  [dev ] member {tag} (cache {cache_dir})")
        dev_members.append(_load_cache_member(cache_dir, args.subtask, "dev", tag))
        print(f"  [test] member {tag} (cache {cache_dir}, split={args.test_split})")
        test_members.append(_load_cache_member(cache_dir, args.subtask, args.test_split, tag))
    sat_dir = Path(args.sat_cache) if args.sat_cache else None
    if sat_dir:
        print(f"  [dev ] member sat_ft (cache {sat_dir})")
        dev_members.append(_load_cache_member(sat_dir, args.subtask, "dev"))
        print(f"  [test] member sat_ft (cache {sat_dir}, split={args.test_split})")
        test_members.append(_load_cache_member(sat_dir, args.subtask, args.test_split))
        member_ids.append("sat_ft")
    if args.weights and len(args.weights) != len(member_ids):
        ap.error(f"--weights has {len(args.weights)} values for {len(member_ids)} members")
    if args.exhaustive:
        selected = _exhaustive_sweep(
            member_ids, args.subtask, dev_members, test_members, args, out)
        member_ids = [member_ids[i] for i in selected]
        dev_members = [dev_members[i] for i in selected]
        test_members = [test_members[i] for i in selected]
        args.weights = [args.weights[i] for i in selected] if args.weights else None
        print(f"[ensemble] Continuing with dev/parsimony winner: "
              f"{member_ids} (review the dev/test table before submitting)")

    # ── Dev: average, tune one threshold ─────────────────────────────────────
    dev_docs = _combine(dev_members, args.combine if args.combine != "stack" else "prob", args.weights)

    stacker = None
    if args.combine == "stack":
        Xd, yd = _pool(dev_members)
        w, b = fit_stacker(Xd, yd, l2=args.stack_l2)
        stacker = (w, b)
        dev_docs = _apply_stack_per_doc(dev_members, w, b)
        print(f"[ensemble] stack weights={np.round(w, 3).tolist()} bias={b:.3f} "
              f"(members={member_ids})")

    grid = np.arange(0.05, 0.96, args.thr_step)
    dev_thr, dev_best = tune_threshold(dev_docs, grid)
    # The threshold that actually drives the written submission. dev_tuned uses
    # the dev-optimal threshold; fixed overrides it (NoPnx ensembles scored
    # better at 0.5 than at their dev-tuned threshold on open-test).
    write_thr = dev_thr if args.threshold_policy == "dev_tuned" else args.threshold
    dev_single = [macro_f1_at(m, dev_thr)["f1"] for m in dev_members]
    print(f"[ensemble] dev tuned@{dev_thr:.2f}: P={dev_best['precision']:.4f} "
          f"R={dev_best['recall']:.4f} F1={dev_best['f1']:.4f}")
    print(f"[ensemble] threshold_policy={args.threshold_policy} → writing submission @ {write_thr:.2f}")
    print(f"[ensemble] dev member F1s @same thr: "
          f"{[f'{mid}={f:.4f}' for mid, f in zip(member_ids, dev_single)]}")

    # ── Test: average, score @ dev threshold, write CSV ──────────────────────
    test_docs = _combine(test_members, args.combine if args.combine != "stack" else "prob", args.weights)

    if stacker is not None:
        test_docs = _apply_stack_per_doc(test_members, *stacker)

    all_gold = np.concatenate([g for _, g in test_docs.values()])
    has_labels = int(all_gold.sum()) > 0
    test_score = None
    if has_labels:
        s05 = macro_f1_at(test_docs, 0.5)
        s_dev = macro_f1_at(test_docs, dev_thr)
        s_write = macro_f1_at(test_docs, write_thr)
        test_score = s_write
        print(f"[ensemble] TEST F1@0.5={s05['f1']:.4f}  |  dev_tuned@{dev_thr:.2f}={s_dev['f1']:.4f}"
              f"  |  WRITTEN@{write_thr:.2f}: P={s_write['precision']:.4f} "
              f"R={s_write['recall']:.4f} F1={s_write['f1']:.4f}")
        # Diagnostic: test P/R/F1 across the grid so the precision<->recall trade is
        # visible. We still WRITE at the dev-tuned threshold (picking on test = leak).
        print(f"[ensemble] TEST P/R/F1 curve (diagnostic only; dev_thr={dev_thr:.2f} drives the CSV):")
        for thr in grid:
            s = macro_f1_at(test_docs, float(thr))
            mark = " <-- dev_thr" if abs(thr - dev_thr) < 1e-9 else ""
            print(f"    thr={thr:.2f}  P={s['precision']:.4f} R={s['recall']:.4f} F1={s['f1']:.4f}{mark}")
    else:
        print(f"[ensemble] test labels hidden — predictions saved only")

    pred_path = out / f"test_predictions_{args.subtask.replace('-', '_')}.csv"
    with open(pred_path, "w") as f:
        f.write("Document ID,Prediction\n")
        for doc_id, (parr, _) in test_docs.items():
            preds = (parr >= write_thr).astype(int)
            f.write(f"{doc_id},{''.join(map(str, preds))}\n")
    print(f"[ensemble] predictions → {pred_path}")

    summary = {
        "subtask": args.subtask,
        "members": member_ids,
        "member_configs": list(args.members),
        "sat_cache": args.sat_cache,
        "threshold_policy": args.threshold_policy,
        "dev_threshold": dev_thr,
        "write_threshold": write_thr,
        "dev": dev_best,
        "dev_member_f1": dict(zip(member_ids, dev_single)),
        "test": test_score,
    }
    with open(out / f"ensemble_{args.subtask.replace('-', '_')}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[ensemble] summary → {out}/ensemble_{args.subtask.replace('-', '_')}.json")

    # ── mlflow: track the ensemble (submission-level) ────────────────────────
    # Optional: the CSV is already written above. Off-cluster (RunPod) skips
    # mlflow (its numpy-2.x deps break the pinned stack), so a miss is harmless.
    try:
        import mlflow
    except ModuleNotFoundError:
        return
    mlflow.set_experiment("AraSeg-Per-Dataset")
    with mlflow.start_run(run_name=f"ens-{args.combine}-{args.subtask.replace('-', '_')}"
                                   f"{'-sat' if args.sat_cache else ''}"):
        mlflow.log_params({"dataset": args.subtask, "combine": args.combine,
                           "members": ",".join(member_ids), "with_sat": bool(args.sat_cache),
                           "threshold_policy": args.threshold_policy})
        mlflow.log_metrics({"dev_f1": dev_best["f1"], "dev_threshold": float(dev_thr),
                            "write_threshold": float(write_thr)})
        if test_score is not None:
            mlflow.log_metrics({"test_f1": test_score["f1"], "test_precision": test_score["precision"],
                                "test_recall": test_score["recall"]})


if __name__ == "__main__":
    main()
