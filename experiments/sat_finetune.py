"""Fine-tune SaT (full or LoRA) on AraSeg train, with the two research-backed tweaks
that target our actual gap (NoPnx over-segmentation). Companion to sat_segment.py
(zero-shot) — reuses its alignment, metrics, prior decoder, and ensemble-ready cache
format, so outputs drop straight into ensemble.py.

Two levers, both cheap, both aimed at the NoPnx gap:

  (#3) PUNCT-CORRUPTION AUGMENTATION  (--punct-corrupt, NoPnx subtasks only)
     SaT+SM's own second stage trains on corrupted text (space/punct removal). For
     Arabic the relevant corruption IS NoPnx = punctuation deletion. We reuse the
     drop_p continuum from punct_aug.py (Track 2's proven lever: aux/dropout beat
     frozen syntax +2.11/+1.34 on NoPnx). Train on PA/NP corrupted at drop_p, EVAL
     on the official NoPnx split (drop_p=1.0).

  (#1) LENGTH-PRIOR DECODE  (reported alongside the plain-threshold baseline)
     AraSeg boundary density is regular (~47/doc), so a count/length prior fixes the
     over-segmentation a flat threshold can't. We reuse sat_segment.fit_count_model
     (a train-fit length->count regression + top-k decode) as the robust prior, and
     also try SaT's native Viterbi decode (--viterbi, best-effort) for comparison.

CLOSED-TRACK LEGALITY: full-FT / LoRA trained ONLY on AraSeg train = legal. The
length prior is fit on TRAIN only (standard model selection). Legal.

Requires (KISSKI .venv-llm): wtpsplit, transformers, datasets, mlflow, torch (+ peft
for --lora). sat-12l-sm and the 4 MBZUAI datasets must be HF-cached (offline nodes).

DATA-SCALE VARIANT (--train-splits / --val-split none): pool several labelled splits
into the training set. Default is the official `train` (174 docs/subtask) with `dev`
held out; `--train-splits train dev --val-split none --eval-splits test` trains on
396 docs and scores the untouched 262-doc `test`. With no held-out set there is no
early stopping and no best-checkpoint selection: the fixed 15-epoch schedule IS the
recipe (the val runs never early-stopped anyway), and the threshold is tuned in-sample
on the training pool. Still closed-track legal — all four splits are AraSeg's own.

Examples:
  python sat_finetune.py --selfcheck                          # no GPU/wtpsplit needed
  python sat_finetune.py --subtask PA                          # full-FT, one subtask
  python sat_finetune.py --punct-corrupt --drop-p 1.0          # all 4; NoPnx uses aug
  python sat_finetune.py --lora                                # LoRA instead of full-FT
  python sat_finetune.py --train-splits train dev --val-split none \
      --eval-splits test --tag sat_ft_traindev                 # 2.3x data, test-scored
"""

import argparse
import pickle
from pathlib import Path

import numpy as np

# reuse — do not reinvent: alignment, metrics, prior decode, cache format
from sat_segment import (
    SUBTASK_DATASETS, load_docs, sat_probs_for_docs,
    macro_f1_at, tune_threshold, fit_count_model, macro_f1_count, count_preds,
)
from punct_aug import build_aug_docs, _NOPNX_OF
from scoring import _doc_f1 as _doc_prf, _macro       # P/R/F1 per doc, no torch import

MODEL = "sat-12l-sm"          # the Supervised-Mixture variant (strongest base)
MAX_LEN = 512
FT_LR = 3e-5                  # full-FT LR for XLM-R (NOT the 2e-4 LoRA uses)
LORA_LR = 2e-4
# PA/NP corrupted at drop_p -> which NoPnx split we eval on
_PA_OF = {v: k for k, v in _NOPNX_OF.items()}   # NoPnx-PA->PA, NoPnx-NP->NP


# -------------------------------------------------- subword labels (as in notebook)
def _boundary_char_ends(tokens, labels):
    ends, pos = set(), 0
    for tok, lab in zip(tokens, labels):
        pos += len(tok)
        if lab == 1:
            ends.add(pos)
        pos += 1
    return ends


def _examples_from_records(records, tokenizer):
    """[{tokens,labels}] -> subword-labelled <=512-token chunks (boundary = subword
    whose last char ends a segment, recovered from the fast tokenizer offsets)."""
    cls, sep = tokenizer.cls_token_id, tokenizer.sep_token_id
    win = MAX_LEN - 2
    out = []
    for r in records:
        text = " ".join(r["tokens"])
        bce = _boundary_char_ends(r["tokens"], r["labels"])
        enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        labs = [1 if (e > s and e in bce) else 0 for (s, e) in offs]
        for i in range(0, len(ids), win):
            out.append({"input_ids": [cls] + ids[i:i + win] + [sep],
                        "labels": [-100] + labs[i:i + win] + [-100]})
    return out


def _train_records(subtask, punct_corrupt, drop_p, seed, keep=None, splits=("train",)):
    """Training docs as [{tokens,labels}] pooled over `splits`. NoPnx + --punct-corrupt
    => drop_p continuum from its PA/NP parent; otherwise the official splits. `keep`
    (set of doc_ids) restricts to a fold's train side for OOF — ids are shared across
    all 4 subtasks."""
    recs = []
    for sp in splits:
        if punct_corrupt and subtask in _PA_OF:
            parent = _PA_OF[subtask]
            print(f"  train[{sp}]: punct-corrupt from {parent} @ drop_p={drop_p}")
            recs += [{"tokens": d["tokens"], "labels": d["labels"]}
                     for d in build_aug_docs(parent, sp, drop_p=drop_p, seed=seed)
                     if keep is None or d["doc_id"] in keep]
        else:
            recs += [{"tokens": t, "labels": g.tolist()}
                     for did, t, g, _ in load_docs(subtask, sp)
                     if keep is None or did in keep]
    return recs


# --------------------------------------------------------------------- training
def _unwrap(m):
    """wtpsplit versions differ: SaT.model is sometimes the raw HF module, sometimes
    a PyTorchWrapper around it (KISSKI). HF Trainer/PEFT need the raw nn.Module."""
    import torch.nn as nn
    inner = getattr(m, "model", m)          # PyTorchWrapper stores it at .model
    return inner if isinstance(inner, nn.Module) else m


def _make_model(use_lora):
    from wtpsplit import SaT
    sat = SaT(MODEL)
    tokenizer, backbone = sat.tokenizer, _unwrap(sat.model).float()
    if not use_lora:
        n = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        print(f"  full fine-tune: {n:,} trainable params")
        return tokenizer, backbone
    from peft import LoraConfig, TaskType, get_peft_model
    cfg = LoraConfig(task_type=TaskType.TOKEN_CLS, r=16, lora_alpha=32, lora_dropout=0.2,
                     target_modules=["query", "key", "value"], modules_to_save=["classifier"])
    model = get_peft_model(backbone, cfg)
    model.print_trainable_parameters()
    return tokenizer, model


def _train(subtask, tokenizer, model, records, out_dir, use_lora, seed, val_split="dev"):
    import torch, torch.nn.functional as F
    from datasets import Dataset
    from transformers import (TrainingArguments, Trainer, EarlyStoppingCallback,
                              DataCollatorForTokenClassification)

    class BCETrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs["labels"]
            out = model(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"))
            logits = out.logits[..., 0]                      # SaT's newline channel
            mask = labels != -100
            loss = F.binary_cross_entropy_with_logits(logits, labels.clamp(min=0).float(),
                                                       reduction="none")
            loss = (loss * mask).sum() / mask.sum().clamp(min=1)
            return (loss, out) if return_outputs else loss

    def metrics(ep):
        logits, labels = ep
        pred = (1 / (1 + np.exp(-np.clip(logits[..., 0], -30, 30))) > 0.5).astype(int)
        m = labels != -100
        y, p = labels[m], pred[m]
        tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
        fn = int(((p == 0) & (y == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        return {"token_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}

    # val_split=None: no held-out set at all -> fixed 15 epochs, no early stopping, no
    # best-checkpoint selection. Safe here because the 15-epoch runs never early-stopped
    # (token_f1 still creeping at epoch 15, job 14722407); the schedule IS the recipe.
    dev_ds = None
    if val_split:
        dev = [{"tokens": t, "labels": g.tolist()} for _, t, g, _ in load_docs(subtask, val_split)]
        dev_ds = Dataset.from_list(_examples_from_records(dev, tokenizer))
    train_ds = Dataset.from_list(_examples_from_records(records, tokenizer))
    print(f"  chunks: train={len(train_ds)} val={len(dev_ds) if dev_ds else 0}")

    args = TrainingArguments(
        output_dir=out_dir, num_train_epochs=15,
        per_device_train_batch_size=32, per_device_eval_batch_size=64,
        learning_rate=LORA_LR if use_lora else FT_LR, warmup_ratio=0.1, weight_decay=0.05,
        eval_strategy="epoch" if dev_ds else "no",
        save_strategy="epoch" if dev_ds else "no", save_total_limit=1,   # no ckpt pileup
        load_best_model_at_end=bool(dev_ds),
        metric_for_best_model="token_f1", greater_is_better=True,
        logging_steps=10, report_to="mlflow", fp16=torch.cuda.is_available(),
        remove_unused_columns=False, label_names=["labels"], seed=seed)
    trainer = BCETrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=dev_ds,
                         data_collator=DataCollatorForTokenClassification(tokenizer, label_pad_token_id=-100),
                         compute_metrics=metrics,
                         callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] if dev_ds else None)
    trainer.train()
    return model


# -------------------------------------------------- inference from locked ckpt
def _find_sat_ckpt(root, subtask):
    """Locked full-FT sat_ft checkpoint: <root>/<subtask>/checkpoint-*/model.safetensors
    (HF Trainer layout; save_total_limit=1 leaves only the best)."""
    root = Path(root)
    cands = sorted(root.glob(f"{subtask}/checkpoint-*/model.safetensors")) \
        or sorted(root.glob(f"{subtask}/**/model.safetensors"))
    if not cands:
        raise FileNotFoundError(f"no sat_ft checkpoint under {root}/{subtask} "
                                f"(expected {subtask}/checkpoint-*/model.safetensors)")
    return cands[-1]


def _infer_from_checkpoint(args):
    """Blind day / any-split: load the FROZEN sat_ft checkpoint (no retraining) and
    write ensemble-ready caches for --splits. Retraining would give different probs
    than the locked decoder/stack weights expect, so we reload the exact backbone."""
    from safetensors.torch import load_file
    subtasks = [args.subtask] if args.subtask else list(SUBTASK_DATASETS)
    d = Path(args.cache_dir); d.mkdir(parents=True, exist_ok=True)
    for st in subtasks:
        print(f"\n=== sat_ft infer : {st} ===")
        _, backbone = _make_model(use_lora=False)          # same arch that was trained
        ckpt = _find_sat_ckpt(args.eval_checkpoint, st)
        sd = load_file(str(ckpt))
        model_keys = set(backbone.state_dict())
        overlap = model_keys & set(sd)
        if len(overlap) < 0.5 * len(model_keys):           # wrong file/arch → don't ship garbage
            raise RuntimeError(f"{ckpt}: only {len(overlap)}/{len(model_keys)} keys match the "
                               f"backbone — wrong checkpoint or architecture, refusing to cache.")
        missing, unexpected = backbone.load_state_dict(sd, strict=False)
        print(f"  loaded {ckpt}  (matched={len(overlap)}/{len(model_keys)} "
              f"missing={len(missing)} unexpected={len(unexpected)})")
        sat = _wrap_sat(backbone, use_lora=False)
        for sp in args.splits:
            cache = sat_probs_for_docs(sat, load_docs(st, sp))
            fn = d / f"{st.replace('-', '_')}_{sp}_sat_ft.pkl"
            pickle.dump(cache, open(fn, "wb"))
            print(f"  cache -> {fn}  ({len(cache)} docs)")


# --------------------------------------------------------------------- eval
def _wrap_sat(model, use_lora):
    import torch
    from wtpsplit import SaT
    from wtpsplit.extract import PyTorchWrapper
    backbone = model.merge_and_unload() if use_lora else model
    if torch.cuda.is_available():
        backbone = backbone.half().to("cuda")
    backbone.eval()
    sat = SaT(MODEL)
    sat.model = PyTorchWrapper(backbone)
    return sat


def _evaluate(subtask, sat, out_cache, try_viterbi, train_splits=("train",),
              val_split="dev", eval_splits=("dev", "test"), tag="sat_ft"):
    """Score plain-threshold (baseline) + length-prior (count-calib) on each of
    `eval_splits` and save ensemble-ready caches. The decision threshold is tuned on
    `val_split`; with val_split=None there is no held-out set, so it is tuned IN-SAMPLE
    on the training pool — reported as such. Returns a metrics dict for mlflow."""
    train_cache = {}
    for sp in train_splits:
        train_cache.update(sat_probs_for_docs(sat, load_docs(subtask, sp)))
    predict = fit_count_model(train_cache, None)                  # length prior, train-fit

    if val_split:
        thr, tune_f1 = tune_threshold(sat_probs_for_docs(sat, load_docs(subtask, val_split)))
        print(f"  threshold {thr:.2f} tuned on {val_split} (F1={tune_f1:.4f})")
    else:
        thr, tune_f1 = tune_threshold(train_cache)
        print(f"  threshold {thr:.2f} tuned IN-SAMPLE on {'+'.join(train_splits)} "
              f"(F1={tune_f1:.4f}) — no validation split")
    res = {"best_threshold": thr, "tune_f1_threshold": tune_f1}

    d = Path(out_cache) if out_cache else None
    if d:
        d.mkdir(parents=True, exist_ok=True)
    for sp in eval_splits:
        cache = sat_probs_for_docs(sat, load_docs(subtask, sp))
        # P and R alongside F1: a results table with F1 alone hides whether a gain is
        # the model firing more or firing better, which is the whole question here.
        m = _macro([_doc_prf((pa >= thr).astype(np.int64), g) for pa, g in cache.values()])
        f1_cnt = macro_f1_count(cache, predict, None)
        res[f"{sp}_f1_threshold"], res[f"{sp}_f1_prior"] = m["f1"], f1_cnt
        res[f"{sp}_precision"], res[f"{sp}_recall"] = m["precision"], m["recall"]
        print(f"  [{sp}] threshold@{thr:.2f} P={m['precision']:.4f} R={m['recall']:.4f} "
              f"F1={m['f1']:.4f}   length-prior F1={f1_cnt:.4f}")
        if d:
            fn = d / f"{subtask.replace('-', '_')}_{sp}_{tag}.pkl"
            pickle.dump(cache, open(fn, "wb"))
            print(f"       cache -> {fn}")

    if try_viterbi:
        try:
            res["test_f1_viterbi"] = _viterbi_f1(subtask, sat)
            print(f"  [native viterbi] test={res['test_f1_viterbi']:.4f}")
        except Exception as e:                                       # best-effort only
            print(f"  [native viterbi] skipped: {e}")
    return res


def _viterbi_f1(subtask, sat):
    """SaT's native Viterbi+length-prior decode, mapped back to token boundaries."""
    from sat_segment import _offsets, _doc_f1
    fs = []
    for _, toks, gold, _ in load_docs(subtask, "test"):
        text = " ".join(toks)
        sents = sat.split(text, do_paragraph_segmentation=False)
        cuts = set()                       # char index where a sentence ends
        pos = 0
        for s in sents:
            pos += len(s)
            cuts.add(pos - len(s) + len(s.rstrip()))     # end of trimmed sentence
        spans, _ = _offsets(toks)
        pred = np.array([1 if (e in cuts or e - 1 in cuts) else 0 for (_, e) in spans], np.int64)
        pred[-1] = 1
        fs.append(_doc_f1(pred, gold))
    return float(np.mean(fs))


# --------------------------------------------------------------------- OOF fold
def run_oof_fold(args):
    """Train SaT on the fold's TRAIN side, predict only the held-out fold's docs,
    write a {doc_id:(prob,gold)} pkl. Union of folds = a leak-free OOF train matrix
    fit_oof_stack.py can stack on (closed-legal). Members' folds must share the same
    fold_map as train.py; oof.py defines it per-subtask, so seeds/folds must match."""
    import mlflow
    from oof import keep_ids, holdout_ids
    st = args.subtask
    if not st:
        raise SystemExit("--holdout-fold requires --subtask")
    keep = keep_ids(st, args.holdout_fold, args.n_folds, args.fold_seed)
    hold = holdout_ids(st, args.holdout_fold, args.n_folds, args.fold_seed)
    print(f"=== SaT OOF {st} fold {args.holdout_fold}/{args.n_folds}: "
          f"train={len(keep)} holdout={len(hold)} ===")
    kind = "lora" if args.lora else "full"
    mlflow.set_experiment("AraSeg-Per-Dataset")
    with mlflow.start_run(run_name=f"{kind}-sat-12l-sm-oof-{st.replace('-', '_')}-f{args.holdout_fold}"):
        mlflow.log_params({"dataset": st, "model": MODEL, "finetune": kind, "mode": "oof_fold",
                           "holdout_fold": args.holdout_fold, "n_folds": args.n_folds,
                           "fold_seed": args.fold_seed, "punct_corrupt": args.punct_corrupt,
                           "drop_p": args.drop_p if args.punct_corrupt else None})
        recs = _train_records(st, args.punct_corrupt, args.drop_p, args.seed, keep=keep)
        tok, model = _make_model(args.lora)
        model = _train(st, tok, model, recs,
                       str(Path(args.out_dir) / f"{st.replace('-', '_')}_oof{args.holdout_fold}"),
                       args.lora, args.seed)
        sat = _wrap_sat(model, args.lora)
        hold_docs = [d for d in load_docs(st, "train") if d[0] in hold]
        cache = sat_probs_for_docs(sat, hold_docs)
        if set(cache) != hold:
            raise SystemExit(f"OOF fold predicted {len(cache)} docs but holdout has {len(hold)}")
        out = Path(args.oof_out); out.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(cache, open(out, "wb"))
        # held-out F1 at 0.5 — a logged diagnostic, not used for any selection
        mlflow.log_metric("holdout_f1_at_0.5", macro_f1_at(cache, 0.5))
        print(f"  OOF fold cache -> {out}  ({len(cache)} docs)")


# --------------------------------------------------------------------- selfcheck
def selfcheck():
    """No GPU / wtpsplit / network. Checks subword labelling round-trips and that
    punct-corrupt records are well-formed."""
    class _Tok:                                             # tiny char tokenizer stub
        cls_token_id, sep_token_id = 101, 102
        def __call__(self, text, return_offsets_mapping=False, add_special_tokens=False):
            offs, pos = [], 0
            for ch in text:
                offs.append((pos, pos + 1)); pos += 1
            return {"input_ids": [ord(c) for c in text], "offset_mapping": offs}

    recs = [{"tokens": ["ab", "c", "de"], "labels": [1, 0, 1]}]
    ex = _examples_from_records(recs, _Tok())
    assert len(ex) == 1 and ex[0]["labels"][0] == -100 and ex[0]["labels"][-1] == -100
    # 'ab'(end@2) is boundary, 'c'(end@4) not, 'de'(end@7) boundary
    body = ex[0]["labels"][1:-1]
    assert body[1] == 1 and body[4] == 0 and body[6] == 1, body
    print("[selfcheck] subword labelling OK:", body)

    for pa_st in ("PA", "NP"):
        recs = _train_records(_NOPNX_OF[pa_st], punct_corrupt=True, drop_p=1.0, seed=42)
        assert all(set(r["labels"]) <= {0, 1} and len(r["tokens"]) == len(r["labels"]) for r in recs)
        print(f"[selfcheck] punct-corrupt {_NOPNX_OF[pa_st]} @drop_p=1.0: {len(recs)} docs OK")

    # multi-split pooling: the pool must be exactly the concatenation, no dedup/loss
    n_tr = len(_train_records("PA", False, 1.0, 42, splits=["train"]))
    n_dv = len(_train_records("PA", False, 1.0, 42, splits=["dev"]))
    n_both = len(_train_records("PA", False, 1.0, 42, splits=["train", "dev"]))
    assert n_both == n_tr + n_dv, f"pooling lost docs: {n_tr}+{n_dv} != {n_both}"
    print(f"[selfcheck] pooled train+dev = {n_tr}+{n_dv} = {n_both} docs OK")
    print("[selfcheck] OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subtask", choices=list(SUBTASK_DATASETS), help="default: all 4")
    ap.add_argument("--lora", action="store_true", help="LoRA instead of full fine-tune")
    ap.add_argument("--punct-corrupt", action="store_true", help="drop_p aug for NoPnx (#3)")
    ap.add_argument("--drop-p", type=float, default=1.0)
    ap.add_argument("--viterbi", action="store_true", help="also try SaT native Viterbi decode")
    ap.add_argument("--out-dir", default="outputs/sat_ft", help="checkpoints (auto-pruned)")
    ap.add_argument("--cache-dir", default="outputs/sat_fullft_caches", help="prob caches for ensemble.py")
    ap.add_argument("--train-splits", nargs="+", default=["train"],
                    help="labelled splits pooled as training data (e.g. train dev test)")
    ap.add_argument("--val-split", default="dev",
                    help="held-out split for early stopping + threshold tuning; "
                         "'none' = no validation split (fixed 15 epochs, in-sample threshold)")
    ap.add_argument("--eval-splits", nargs="+", default=["dev", "test"],
                    help="labelled splits to score after training")
    ap.add_argument("--tag", default="sat_ft",
                    help="cache filename suffix; change it so a new run cannot overwrite "
                         "the locked sat_ft caches")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--holdout-fold", type=int, default=None,
                    help="OOF mode: train on all folds except this one, predict it, write --oof-out.")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--fold-seed", type=int, default=42)
    ap.add_argument("--oof-out", default=None, help="OOF fold cache pkl (with --holdout-fold)")
    ap.add_argument("--eval-checkpoint", default=None,
                    help="Inference-only: load the FROZEN sat_ft checkpoint from this root "
                         "(<root>/<subtask>/checkpoint-*/model.safetensors) and cache --splits. "
                         "No training. Used on blind day (e.g. --eval-checkpoint "
                         "/workspace/AraSeg-backup/sat_ft --splits blind).")
    ap.add_argument("--splits", nargs="+", default=["blind"],
                    help="Splits to cache in --eval-checkpoint mode (default: blind).")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.val_split.lower() in ("none", ""):
        args.val_split = None
    overlap = set(args.train_splits) & set(args.eval_splits)
    if overlap:                       # a number on data you trained on is not a result
        print(f"*** WARNING: {sorted(overlap)} is in BOTH --train-splits and "
              f"--eval-splits — those F1s are in-sample, not generalisation. ***")

    if args.selfcheck:
        selfcheck(); return
    if args.eval_checkpoint:
        _infer_from_checkpoint(args); return
    if args.holdout_fold is not None:
        run_oof_fold(args); return

    import mlflow
    mlflow.set_experiment("AraSeg-Per-Dataset")
    kind = "lora" if args.lora else "full"
    prefix = f"{kind}-sat-12l-sm"
    subtasks = [args.subtask] if args.subtask else list(SUBTASK_DATASETS)

    for st in subtasks:
        print(f"\n=== {prefix} : {st} ===")
        if mlflow.active_run():
            mlflow.end_run()
        with mlflow.start_run(run_name=f"{prefix}-{st.replace('-', '_')}"):
            mlflow.log_params({"dataset": st, "model": MODEL, "finetune": kind,
                               "learning_rate": LORA_LR if args.lora else FT_LR,
                               "punct_corrupt": args.punct_corrupt,
                               "drop_p": args.drop_p if args.punct_corrupt else None,
                               "train_splits": "+".join(args.train_splits),
                               "val_split": args.val_split or "none"})
            recs = _train_records(st, args.punct_corrupt, args.drop_p, args.seed,
                                  splits=args.train_splits)
            print(f"  training docs: {len(recs)} from {'+'.join(args.train_splits)}")
            tok, model = _make_model(args.lora)
            model = _train(st, tok, model, recs, str(Path(args.out_dir) / st), args.lora,
                           args.seed, val_split=args.val_split)
            sat = _wrap_sat(model, args.lora)
            res = _evaluate(st, sat, args.cache_dir, args.viterbi,
                            train_splits=args.train_splits, val_split=args.val_split,
                            eval_splits=args.eval_splits, tag=args.tag)
            mlflow.log_metrics(res)


if __name__ == "__main__":
    main()
