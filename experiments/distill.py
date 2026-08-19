"""AraSeg 2026 - punctuation-invariant distillation.

Teacher trained on punctuated docs produces per-word boundary probabilities.
Student trains on the same docs without punctuation using CE + KL supervision.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from data_utils import AraSegDataset, make_collate
from data_utils import _load_split, _detect_columns
from distill_alignment import align_teacher_to_student
from model import build_model, save_checkpoint, load_checkpoint
from metrics import collect_doc_probs, tune_threshold, macro_f1_at
from train import Config, set_seed, make_loader, build_criterion, _build_llrd_optimizer
from utils import resolve_model_path


@dataclass
class DistillConfig(Config):
    distill_teacher_config: str = ""
    distill_teacher_subtask: str = ""
    distill_soft_probs: str = ""
    distill_alpha: float = 1.0
    distill_temperature: float = 2.0

    @classmethod
    def from_yaml(cls, path: str) -> "DistillConfig":
        with open(path) as f:
            return cls(**yaml.safe_load(f))


def _collect_teacher_probs(teacher_cfg: Config, teacher_subtask: str,
                           device: str) -> Dict[str, np.ndarray]:
    """Run teacher on its training split and return {doc_id: probs_per_word}."""
    model_path = resolve_model_path(teacher_cfg.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              trust_remote_code=teacher_cfg.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = AraSegDataset(teacher_subtask, "train", tokenizer,
                       teacher_cfg.max_length, teacher_cfg.window_stride)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    loader = make_loader(ds, teacher_cfg.batch_size, False, teacher_cfg.num_workers,
                         make_collate(pad_id))

    teacher = build_model(teacher_cfg, model_path).to(device)
    ckpt_name = f"best_{teacher_subtask.replace('-', '_')}.pt"
    ckpt_path = Path(teacher_cfg.output_dir) / ckpt_name
    load_checkpoint(teacher, ckpt_path, device)
    teacher.eval()

    docs = collect_doc_probs(teacher, loader, device, teacher_subtask, ds.doc_gold)
    return {doc_id: parr for doc_id, (parr, _) in docs.items()}


def _doc_tokens(subtask: str, split: str) -> Dict[str, list]:
    raw = _load_split(subtask, split)
    token_col, _, id_col = _detect_columns(raw[0])
    return {str(row[id_col]) if id_col else str(i): list(row[token_col])
            for i, row in enumerate(raw)}


def _distill_step(model, batch, criterion, teacher_probs: Dict[str, np.ndarray],
                  device: str, subtask: str, alpha: float, temperature: float) -> torch.Tensor:
    """CE loss on hard labels plus KL divergence against teacher soft probs."""
    labels = batch["labels"].to(device)
    logits = model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        token_type_ids=batch["token_type_ids"].to(device),
        subtask=subtask,
    )
    ce_loss = criterion(logits.view(-1, 2), labels.view(-1))
    if alpha == 0.0:
        return ce_loss

    B, T = logits.shape[:2]
    teacher_soft = torch.zeros(B, T, device=device)
    word_idx = batch["word_idx"]
    for i, doc_id in enumerate(batch["doc_ids"]):
        t_probs = teacher_probs.get(doc_id)
        if t_probs is None:
            continue
        n_words = len(t_probs)
        for t in range(T):
            w = int(word_idx[i, t].item())
            if 0 <= w < n_words:
                teacher_soft[i, t] = float(t_probs[w])

    valid = labels.view(-1) != -100
    if valid.sum() == 0:
        return ce_loss

    t_soft_flat = teacher_soft.view(-1)[valid]
    t_dist = torch.stack([1.0 - t_soft_flat, t_soft_flat], dim=-1).clamp(min=1e-8)
    t_dist_T = torch.softmax(t_dist.log() / temperature, dim=-1)
    s_log_T = F.log_softmax(logits.view(-1, 2)[valid] / temperature, dim=-1)
    kl = F.kl_div(s_log_T, t_dist_T, reduction="batchmean")
    return ce_loss + alpha * (temperature ** 2) * kl


def run_teacher(cfg: DistillConfig, device: str) -> None:
    teacher_cfg = Config.from_yaml(cfg.distill_teacher_config)
    print(f"[teacher] {cfg.distill_teacher_subtask}: inference from {teacher_cfg.experiment_id}")
    probs = _collect_teacher_probs(teacher_cfg, cfg.distill_teacher_subtask, device)
    tokens = _doc_tokens(cfg.distill_teacher_subtask, "train")
    doc_ids = list(probs.keys())
    arrs = np.empty(len(doc_ids), dtype=object)
    token_arrs = np.empty(len(doc_ids), dtype=object)
    for i, d in enumerate(doc_ids):
        arrs[i] = np.asarray(probs[d], dtype=np.float32)
        token_arrs[i] = np.asarray(tokens[d], dtype=object)
    Path(cfg.distill_soft_probs).parent.mkdir(parents=True, exist_ok=True)
    np.savez(cfg.distill_soft_probs,
             doc_ids=np.array(doc_ids, dtype=object), probs=arrs, tokens=token_arrs,
             teacher_subtask=np.array(cfg.distill_teacher_subtask, dtype=object))
    print(f"[teacher] saved soft probs for {len(doc_ids)} docs -> {cfg.distill_soft_probs}")


def _load_teacher_probs(path: str) -> Tuple[Dict[str, np.ndarray], Dict[str, list]]:
    z = np.load(path, allow_pickle=True)
    probs = {str(d): np.asarray(p, dtype=np.float32) for d, p in zip(z["doc_ids"], z["probs"])}
    tokens = {str(d): [str(t) for t in ts] for d, ts in zip(z["doc_ids"], z["tokens"])}
    return probs, tokens


def _project_teacher_probs(
    teacher_probs: Dict[str, np.ndarray],
    teacher_tokens: Dict[str, list],
    student_tokens: Dict[str, list],
) -> Dict[str, np.ndarray]:
    projected = {}
    errors = []
    for doc_id, stoks in student_tokens.items():
        if doc_id not in teacher_probs or doc_id not in teacher_tokens:
            continue
        try:
            projected[doc_id] = np.asarray(
                align_teacher_to_student(teacher_tokens[doc_id], teacher_probs[doc_id], stoks),
                dtype=np.float32,
            )
        except ValueError as exc:
            errors.append((doc_id, str(exc)))
    if errors:
        sample = "; ".join(f"{doc}: {err}" for doc, err in errors[:3])
        raise AssertionError(f"Teacher/student token alignment failed on {len(errors)} docs, e.g. {sample}")
    return projected


def run_student(cfg: DistillConfig, device: str) -> None:
    set_seed(cfg.seed)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_path = resolve_model_path(cfg.model_name)
    print(f"[{cfg.experiment_id}] distillation: "
          f"teacher_subtask={cfg.distill_teacher_subtask} -> student_subtask={cfg.subtasks[0]}")
    if not Path(cfg.distill_soft_probs).exists():
        raise FileNotFoundError(
            f"Teacher probs not found at {cfg.distill_soft_probs}. Run teacher mode first.")
    raw_teacher_probs, teacher_tokens = _load_teacher_probs(cfg.distill_soft_probs)
    print(f"[{cfg.experiment_id}] loaded teacher probs for {len(raw_teacher_probs)} documents.")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=cfg.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    subtask = cfg.subtasks[0]
    train_set = AraSegDataset(subtask, "train", tokenizer, cfg.max_length, cfg.window_stride)
    dev_set = AraSegDataset(subtask, "dev", tokenizer, cfg.max_length, cfg.window_stride)
    student_tokens = _doc_tokens(subtask, "train")
    teacher_probs = _project_teacher_probs(raw_teacher_probs, teacher_tokens, student_tokens)

    student_ids = set(train_set.doc_gold.keys())
    covered = student_ids & set(teacher_probs.keys())
    frac = len(covered) / max(len(student_ids), 1)
    if frac < 0.95:
        raise AssertionError(
            f"Teacher probs cover only {frac:.0%} of student docs "
            f"({len(covered)}/{len(student_ids)}). Re-check alignment.")
    bad = [(d, len(teacher_probs[d]), len(train_set.doc_gold[d]))
           for d in covered if len(teacher_probs[d]) != len(train_set.doc_gold[d])]
    if bad:
        raise AssertionError(
            f"Projected teacher/student length mismatch on {len(bad)} docs, e.g. {bad[:3]}. Re-check alignment.")
    print(f"[{cfg.experiment_id}] alignment OK: teacher covers {frac:.0%} of student docs after punctuation projection.")

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    collate_fn = make_collate(pad_id)
    train_loader = make_loader(train_set, cfg.batch_size, True, cfg.num_workers, collate_fn)
    dev_loader = make_loader(dev_set, cfg.batch_size, False, cfg.num_workers, collate_fn)

    model = build_model(cfg, model_path).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[{cfg.experiment_id}] params: trainable {n_train:,} / total {n_total:,} "
          f"({100 * n_train / max(n_total, 1):.2f}%)")

    criterion = build_criterion(cfg, device)
    if cfg.llrd_decay < 1.0:
        optimizer = _build_llrd_optimizer(model, cfg)
    else:
        optimizer = AdamW((p for p in model.parameters() if p.requires_grad),
                          lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * len(train_loader)
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_f1 = 0.0
    best_thr = 0.5
    history: Dict = {}
    for epoch in range(1, cfg.epochs + 1):
        print(f"\n[{cfg.experiment_id}] === Epoch {epoch}/{cfg.epochs} ===")
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            loss = _distill_step(model, batch, criterion, teacher_probs, device,
                                 subtask, cfg.distill_alpha, cfg.distill_temperature)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        docs = collect_doc_probs(model, dev_loader, device, subtask, dev_set.doc_gold)
        s05 = macro_f1_at(docs, 0.5)
        thr, s_best = tune_threshold(docs)
        print(f"  dev [{subtask}]  F1@0.5={s05['f1']:.4f}  |  "
              f"tuned@{thr:.2f}: P={s_best['precision']:.4f} R={s_best['recall']:.4f} F1={s_best['f1']:.4f}")
        if s_best["f1"] > best_f1:
            best_f1 = s_best["f1"]
            best_thr = thr
            save_checkpoint(model, out / f"best_{subtask.replace('-', '_')}.pt")
        history[f"epoch_{epoch}"] = {"train_loss": avg_loss,
                                      "dev": {subtask: {"thr0.5": s05, "tuned": s_best,
                                                         "best_threshold": thr}}}

    ckpt_path = out / f"best_{subtask.replace('-', '_')}.pt"
    load_checkpoint(model, ckpt_path, device)
    test_set = AraSegDataset(subtask, "test", tokenizer, cfg.max_length, cfg.window_stride)
    test_loader = make_loader(test_set, cfg.batch_size, False, cfg.num_workers, collate_fn)
    docs = collect_doc_probs(model, test_loader, device, subtask, test_set.doc_gold)
    s_thr = macro_f1_at(docs, best_thr)
    print(f"  [{subtask}]  tuned@{best_thr:.2f}: P={s_thr['precision']:.4f} "
          f"R={s_thr['recall']:.4f} F1={s_thr['f1']:.4f}")

    pred_path = out / f"test_predictions_{subtask.replace('-', '_')}.csv"
    with open(pred_path, "w") as f:
        f.write("Document ID,Prediction\n")
        for doc_id, (parr, _) in docs.items():
            preds = (parr >= best_thr).astype(int)
            f.write(f"{doc_id},{''.join(map(str, preds))}\n")

    summary = {
        "experiment_id": cfg.experiment_id,
        "model_name": cfg.model_name,
        "task_mode": cfg.task_mode,
        "subtasks": cfg.subtasks,
        "distill_teacher": cfg.distill_teacher_config,
        "distill_alpha": cfg.distill_alpha,
        "distill_temperature": cfg.distill_temperature,
        "best_dev_f1": {subtask: best_f1},
        "best_threshold": {subtask: best_thr},
        "best_test_f1": {subtask: s_thr["f1"]},
        "history": history,
    }
    with open(out / "results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved -> {out}/results.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["teacher", "student"], required=True)
    args = parser.parse_args()

    cfg = DistillConfig.from_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.mode == "teacher":
        run_teacher(cfg, device)
    else:
        run_student(cfg, device)


if __name__ == "__main__":
    main()
