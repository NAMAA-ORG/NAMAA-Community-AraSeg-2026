"""
AraSeg 2026 — Training script.

Supports single-task (one subtask, one head) and multi-task (shared encoder,
per-subtask heads trained by randomly rotating through subtask batches).

Opt-in training features (all default to OFF / current behavior):
  - head_type: "linear" (default) | "bilstm"          (see model.py)
  - loss_type: "ce" (default) | "focal"               (focal_gamma)
  - fgm:       adversarial training on word embeddings (fgm_epsilon)
  - swa:       stochastic weight averaging from swa_start_epoch

Usage:
    python train.py --config configs/e1_arabert_single_pa.yaml
"""

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

try:
    import mlflow
except ModuleNotFoundError:
    # Inference/ensemble path never logs — off-cluster (RunPod) skips the heavy
    # mlflow tree (it drags in numpy 2.x and breaks the pinned stack). Training
    # still requires a real mlflow; this stub only keeps the import from failing.
    class _NoMlflow:
        def __getattr__(self, _):
            return lambda *a, **k: None
    mlflow = _NoMlflow()
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.optim import AdamW
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from data_utils import AraSegDataset, make_collate
from model import (build_model, save_checkpoint, load_checkpoint,
                   load_encoder_weights, _get_num_layers)
from metrics import (collect_doc_probs, tune_threshold, macro_f1_at,
                     collect_doc_viterbi, macro_f1_preds)
from utils import resolve_model_path


@dataclass
class Config:
    experiment_id: str
    model_name: str
    task_mode: str          # "single" | "multi"
    subtasks: List[str]
    epochs: int
    batch_size: int
    lr: float
    max_length: int
    output_dir: str
    seed: int = 42
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    num_workers: int = 4
    trust_remote_code: bool = False
    pos_weight: float = 1.0       # weight for boundary class (>1 boosts recall)
    window_stride: int = 128      # token overlap between consecutive document windows
    # ── opt-in features (defaults preserve original behavior) ──
    head_type: str = "linear"     # "linear" | "bilstm"
    lstm_hidden: int = 256
    lstm_layers: int = 1
    loss_type: str = "ce"         # "ce" | "focal"
    focal_gamma: float = 2.0
    fgm: bool = False             # adversarial training on embeddings
    fgm_epsilon: float = 1.0
    swa: bool = False             # stochastic weight averaging
    swa_start_epoch: int = 0      # start averaging from this epoch (1-indexed); 0 => last 25%
    # ── LLM backbones (e31+): all default OFF so e1–e30 behave identically ──
    torch_dtype: str = "float32"          # "bfloat16" for LLM backbones
    bidirectional: bool = False           # defeat the causal mask (LLM2Vec-style)
    lora_r: int = 0                       # 0 = no LoRA (full fine-tune)
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None   # None = q/k/v/o/gate/up/down proj
    gradient_checkpointing: bool = False
    # ── R-Drop, LLRD, and sequential pretraining (e34+) ──────────────────────
    rdrop_alpha: float = 0.0
    llrd_decay: float = 1.0
    pretrain_from: Optional[str] = None
    # ── QLoRA on small cards (e67+, Kaggle T4): 4-bit base, fp16 compute ─────
    load_in_4bit: bool = False    # NF4-quantize the frozen base (see model.py)
    grad_accum: int = 1           # optimizer step every N batches (effective batch = batch_size*N)
    fp16_scaler: bool = False     # GradScaler for fp16 compute (T4 has no bf16)
    # ── Phase B: recall loss, syntax features, worst-genre selection ──────────
    recall_beta: float = 2.0          # β for F-beta Dice recall loss (β>1 → favour recall)
    use_syntax: bool = False          # concat frozen POS/dep embeddings to encoder output
    syntax_cache_dir: str = ""        # directory with extract_syntax.py outputs
    n_pos_tags: int = 0               # vocab size (from vocab.json); 0 = syntax disabled
    n_dep_rels: int = 0
    syntax_dim: int = 32              # embedding dim for POS/dep tables
    use_worst_genre: bool = False     # checkpoint selection by worst-genre dev F1
    genre_cache_dir: str = ""         # directory with genre_proxy.py outputs
    # ── Track 2: punctuation-dropout augmentation + aux punctuation head ───────
    use_punct_aug: bool = False       # build NoPnx train data from PA via punct_aug.py
    punct_aug_source: str = ""        # PA subtask to corrupt ("PA" -> NoPnx-PA, "NP" -> NoPnx-NP)
    punct_drop_p: float = 1.0         # 1.0 = exact NoPnx; <1.0 = partial-punctuation continuum
    n_aux_classes: int = 0            # 0 = no aux head; 3 = NONE/OTHER/SENT punct-restore head
    aux_punct_weight: float = 0.5     # weight of the aux punctuation-restoration loss
    # ── Naqta-restore (Kareem's recipe): train on punctuation-RESTORED NoPnx text ─
    naqta_restore: bool = False       # insert Naqta-predicted marks into train+dev input
    naqta_restore_min_p: float = 0.5  # insert a mark only above this class probability
    naqta_comma: str = "،"            # class-2 mark override; "،" reproduces e71/e72
    # ── Paired-view consistency (Lane C Task 4) ─────────────────────────────────
    paired_view_source: str = ""      # e.g. "PA" -- project punctuation onto NoPnx train
    paired_consistency_alpha: float = 0.0  # weight of the symmetric per-word KL term

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            return cls(**yaml.safe_load(f))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(ds, batch_size: int, shuffle: bool, num_workers: int, collate_fn) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )


# ── Loss ─────────────────────────────────────────────────────────────────────
def build_criterion(cfg: Config, device: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Return a callable(logits_2d, target_1d) -> scalar loss.

    class_weight[1] > 1 upweights boundary tokens → higher recall.
    "ce" uses the original nn.CrossEntropyLoss for exact reproducibility.
    "focal" adds the (1-p_t)^gamma down-weighting of easy (non-boundary) tokens.
    """
    class_weight = torch.tensor([1.0, cfg.pos_weight], device=device)

    if cfg.loss_type == "ce":
        return nn.CrossEntropyLoss(ignore_index=-100, weight=class_weight)

    if cfg.loss_type == "focal":
        gamma = cfg.focal_gamma

        def focal(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            valid = target != -100
            if valid.sum() == 0:
                return logits.sum() * 0.0
            logits, target = logits[valid], target[valid]
            logp = F.log_softmax(logits, dim=-1)
            ce = F.nll_loss(logp, target, weight=class_weight, reduction="none")  # weighted -log p_t
            pt = logp.gather(1, target.unsqueeze(1)).squeeze(1).exp()             # p_t
            return (((1.0 - pt) ** gamma) * ce).mean()

        return focal

    if cfg.loss_type == "recall":
        # F-beta Dice loss: differentiable, directly maximises F-beta over boundary class.
        # β=2 → recall weighted 4× more than precision. Smooth=1 prevents div-by-zero.
        # Implementation note: batch-level Dice, not per-doc — stable enough on the boundary-rare AraSeg data
        beta2 = cfg.recall_beta ** 2

        def recall_dice(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            valid = target != -100
            if not valid.any():
                return logits.sum() * 0.0
            p = torch.softmax(logits[valid], dim=-1)[:, 1]   # P(boundary)
            t = target[valid].float()
            tp = (p * t).sum()
            fn = ((1.0 - p) * t).sum()
            fp = (p * (1.0 - t)).sum()
            return 1.0 - (1.0 + beta2) * tp / ((1.0 + beta2) * tp + beta2 * fn + fp + 1.0)

        return recall_dice

    raise ValueError(f"Unknown loss_type={cfg.loss_type!r} (expected 'ce', 'focal', or 'recall')")


def _build_llrd_optimizer(model: nn.Module, cfg: Config) -> AdamW:
    """Layer-wise LR decay for LoRA fine-tuning."""
    import re

    enc = model.encoder
    inner = getattr(getattr(enc, "base_model", enc), "model", enc)
    hf_config = getattr(inner, "config", None) or getattr(enc, "config", None)
    if hf_config is None:
        raise AttributeError("Could not locate an HF config under model.encoder for LLRD")
    num_layers = _get_num_layers(hf_config)

    groups: dict = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        m = re.search(r"\.layers\.(\d+)\.", name)
        depth = int(m.group(1)) if m else num_layers
        groups.setdefault(depth, []).append(param)

    param_groups = [
        {"params": ps, "lr": cfg.lr * (cfg.llrd_decay ** (num_layers - d))}
        for d, ps in groups.items()
    ]
    print(f"[llrd] num_layers={num_layers}  param_groups={len(param_groups)}")
    return AdamW(param_groups, weight_decay=cfg.weight_decay)


# ── FGM adversarial training ────────────────────────────────────────────────
class FGM:
    """Fast Gradient Method: perturb word-embedding weights along their gradient
    to create a worst-case input, then take an extra step on the perturbed loss.
    Standard, reliable token-task regularizer (~+0.3-0.5 F1)."""

    def __init__(self, model: nn.Module, epsilon: float = 1.0, substr: str = "word_embeddings"):
        self.model = model
        self.epsilon = epsilon
        self.substr = substr
        self.backup: Dict[str, torch.Tensor] = {}

    def attack(self):
        for name, p in self.model.named_parameters():
            if p.requires_grad and self.substr in name and p.grad is not None:
                self.backup[name] = p.data.clone()
                norm = torch.norm(p.grad)
                if norm != 0 and not torch.isnan(norm):
                    p.data.add_(self.epsilon * p.grad / norm)

    def restore(self):
        for name, p in self.model.named_parameters():
            if name in self.backup:
                p.data = self.backup[name]
        self.backup = {}


def _rdrop_kl(logits1: torch.Tensor, logits2: torch.Tensor,
              labels_flat: torch.Tensor) -> torch.Tensor:
    """Symmetric KL divergence at valid non-padding token positions."""
    valid = labels_flat != -100
    if valid.sum() == 0:
        return logits1.sum() * 0.0
    l1, l2 = logits1[valid], logits2[valid]
    p1 = F.softmax(l1, dim=-1)
    p2 = F.softmax(l2, dim=-1)
    kl12 = F.kl_div(F.log_softmax(l1, dim=-1), p2, reduction="batchmean")
    kl21 = F.kl_div(F.log_softmax(l2, dim=-1), p1, reduction="batchmean")
    return (kl12 + kl21) / 2.0


def _gather_first_subword(logits: torch.Tensor, word_idx: torch.Tensor):
    """Dense left-contiguous (batch, max_words, C) pack of first-subword logits,
    plus a mask for which slots are real words. Same packing AraSegModel._gather_words
    uses for CRF, kept standalone here since paired_word_kl runs on two independently-
    tokenized views with different sequence lengths."""
    B, T, C = logits.shape
    first = word_idx >= 0
    counts = first.sum(dim=1)
    Wmax = max(int(counts.max().item()), 1) if B else 1
    out = logits.new_zeros(B, Wmax, C)
    mask = torch.zeros(B, Wmax, dtype=torch.bool, device=logits.device)
    for i in range(B):
        idx = first[i].nonzero(as_tuple=True)[0]
        n = int(idx.numel())
        if n == 0:
            continue
        out[i, :n] = logits[i, idx]
        mask[i, :n] = True
    return out, mask


def paired_word_kl(logits_a: torch.Tensor, word_idx_a: torch.Tensor,
                   logits_b: torch.Tensor, word_idx_b: torch.Tensor) -> torch.Tensor:
    """Symmetric per-word KL between two views of the same words. The views tokenize
    differently (e.g. NoPnx text vs its punctuation-projected pair), so positions
    can't be compared directly -- gather each view's first-subword logits into
    word-index order first (build_paired_chunks guarantees equal word counts and
    matching global word indices per chunk), then compare word-for-word."""
    wa, ma = _gather_first_subword(logits_a, word_idx_a)
    wb, mb = _gather_first_subword(logits_b, word_idx_b)
    n = min(wa.size(1), wb.size(1))
    mask = ma[:, :n] & mb[:, :n]
    if not mask.any():
        return logits_a.sum() * 0.0
    la = F.log_softmax(wa[:, :n][mask], dim=-1)
    lb = F.log_softmax(wb[:, :n][mask], dim=-1)
    kl_ab = F.kl_div(la, lb.exp(), reduction="batchmean")
    kl_ba = F.kl_div(lb, la.exp(), reduction="batchmean")
    return (kl_ab + kl_ba) / 2.0


def _paired_view_loss(model, batch, criterion, device, subtask, alpha) -> torch.Tensor:
    """CE trains on the NoPnx view's official labels (the paired view carries no
    labels of its own); a symmetric per-word KL pulls both views' predictions
    together. See punct_aug.build_paired_chunks and paired_word_kl."""
    labels = batch["labels"].to(device)
    logits = model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        token_type_ids=batch["token_type_ids"].to(device),
        subtask=subtask,
    )
    ce = criterion(logits.view(-1, 2), labels.view(-1))
    paired_logits = model(
        input_ids=batch["paired_input_ids"].to(device),
        attention_mask=batch["paired_attention_mask"].to(device),
        token_type_ids=batch["paired_token_type_ids"].to(device),
        subtask=subtask,
    )
    kl = paired_word_kl(logits, batch["word_idx"].to(device),
                        paired_logits, batch["paired_word_idx"].to(device))
    return ce + alpha * kl


def _step_loss(model, batch, criterion, device, subtask) -> torch.Tensor:
    labels = batch["labels"].to(device)
    syn_pos = batch["syntax_pos_ids"].to(device) if "syntax_pos_ids" in batch else None
    syn_dep = batch["syntax_dep_ids"].to(device) if "syntax_dep_ids" in batch else None
    if getattr(model, "head_type", "linear") == "crf":
        return model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            token_type_ids=batch["token_type_ids"].to(device),
            subtask=subtask,
            labels=labels,
            word_idx=batch["word_idx"].to(device),
            syntax_pos_ids=syn_pos,
            syntax_dep_ids=syn_dep,
        )
    want_aux = getattr(model, "aux_head", None) is not None and "aux_labels" in batch
    out = model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        token_type_ids=batch["token_type_ids"].to(device),
        subtask=subtask,
        syntax_pos_ids=syn_pos,
        syntax_dep_ids=syn_dep,
        return_aux=want_aux,
    )
    if want_aux:
        logits, aux_logits = out
        loss = criterion(logits.view(-1, 2), labels.view(-1))
        aux = batch["aux_labels"].to(device)
        aux_loss = F.cross_entropy(aux_logits.reshape(-1, aux_logits.size(-1)),
                                   aux.view(-1), ignore_index=-100)
        return loss + getattr(model, "aux_weight", 0.5) * aux_loss
    return criterion(out.view(-1, 2), labels.view(-1))


def _optim_step(model, batch, criterion, optimizer, scheduler, device,
                subtask, grad_clip, fgm, rdrop_alpha=0.0, paired_consistency_alpha=0.0,
                scaler=None, accum=1, do_step=True):
    """One micro-batch: accumulate grads; step/zero only when do_step.

    Defaults (scaler=None, accum=1, do_step=True) reproduce the original
    per-batch behavior exactly, so existing callers are unaffected.
    """
    if "paired_input_ids" in batch:
        loss = _paired_view_loss(model, batch, criterion, device, subtask,
                                 paired_consistency_alpha)
    elif rdrop_alpha > 0.0:
        labels_flat = batch["labels"].view(-1).to(device)
        syn_pos = batch["syntax_pos_ids"].to(device) if "syntax_pos_ids" in batch else None
        syn_dep = batch["syntax_dep_ids"].to(device) if "syntax_dep_ids" in batch else None
        logits1 = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            token_type_ids=batch["token_type_ids"].to(device),
            subtask=subtask,
            syntax_pos_ids=syn_pos,
            syntax_dep_ids=syn_dep,
        )
        logits2 = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            token_type_ids=batch["token_type_ids"].to(device),
            subtask=subtask,
            syntax_pos_ids=syn_pos,
            syntax_dep_ids=syn_dep,
        )
        ce = (criterion(logits1.view(-1, 2), labels_flat) +
              criterion(logits2.view(-1, 2), labels_flat)) / 2.0
        kl = _rdrop_kl(logits1.view(-1, 2), logits2.view(-1, 2), labels_flat)
        loss = ce + rdrop_alpha * kl
    else:
        loss = _step_loss(model, batch, criterion, device, subtask)
    scaled = loss / accum
    if scaler is not None:
        scaler.scale(scaled).backward()
    else:
        scaled.backward()
    if fgm is not None:                 # never combined with scaler (checked in main)
        fgm.attack()                                            # perturb embeddings
        loss_adv = _step_loss(model, batch, criterion, device, subtask) / accum
        loss_adv.backward()                                     # accumulate adv grads
        fgm.restore()                                           # undo perturbation
    if do_step:
        if scaler is not None:
            scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return loss.item()


def train_single(model, loader, optimizer, scheduler, criterion, device,
                 subtask, grad_clip, fgm, rdrop_alpha=0.0, paired_consistency_alpha=0.0,
                 scaler=None, accum=1, limit_batches=0) -> float:
    model.train()
    optimizer.zero_grad()
    n_batches = len(loader) if limit_batches <= 0 else min(limit_batches, len(loader))
    total = 0.0
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        do_step = ((i + 1) % accum == 0) or (i + 1 == n_batches)
        total += _optim_step(model, batch, criterion, optimizer, scheduler,
                             device, subtask, grad_clip, fgm, rdrop_alpha,
                             paired_consistency_alpha,
                             scaler=scaler, accum=accum, do_step=do_step)
    return total / max(n_batches, 1)


def train_multi(model, loaders, optimizer, scheduler, criterion, device,
                subtasks, steps, grad_clip, fgm, rdrop_alpha=0.0,
                scaler=None, accum=1, limit_batches=0) -> float:
    model.train()
    optimizer.zero_grad()
    if limit_batches > 0:
        steps = min(limit_batches, steps)
    iterators = {st: iter(loaders[st]) for st in subtasks}
    total = 0.0
    for i in range(steps):
        subtask = random.choice(subtasks)
        try:
            batch = next(iterators[subtask])
        except StopIteration:
            iterators[subtask] = iter(loaders[subtask])
            batch = next(iterators[subtask])
        do_step = ((i + 1) % accum == 0) or (i + 1 == steps)
        total += _optim_step(model, batch, criterion, optimizer, scheduler,
                             device, subtask, grad_clip, fgm, rdrop_alpha,
                             scaler=scaler, accum=accum, do_step=do_step)
    return total / steps


def _eval_subtask(model, loader, device, subtask, doc_gold, return_docs=False):
    """Full-doc dev eval. CRF uses Viterbi; other heads tune a threshold."""
    if getattr(model, "head_type", "linear") == "crf":
        docs = collect_doc_viterbi(model, loader, device, subtask, doc_gold)
        s = macro_f1_preds(docs)
        result = (s, -1.0, s)
        return (*result, docs) if return_docs else result
    docs = collect_doc_probs(model, loader, device, subtask, doc_gold)
    s05 = macro_f1_at(docs, 0.5)
    thr, s_best = tune_threshold(docs)
    result = (s05, thr, s_best)
    return (*result, docs) if return_docs else result


# ── Epoch-level resume (12 h Kaggle sessions chained into one run; e67+) ────
def _trainable_sd(model: nn.Module) -> dict:
    keep = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v for k, v in model.state_dict().items() if k in keep}


def _save_train_state(path: Path, epoch, model, optimizer, scheduler, scaler,
                      best_f1, best_thr, history) -> None:
    """Atomic save — a session killed mid-write must not corrupt the state."""
    state = {
        "epoch": epoch,
        "model": _trainable_sd(model),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best_f1": best_f1, "best_thr": best_thr, "history": history,
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }
    tmp = Path(str(path) + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def _load_train_state(path: Path, model, optimizer, scheduler, scaler, device) -> dict:
    state = torch.load(path, map_location=device, weights_only=False)
    _, unexpected = model.load_state_dict(state["model"], strict=False)
    if unexpected:
        raise RuntimeError(f"resume: unexpected keys in {path}: {list(unexpected)[:5]} ...")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and state.get("scaler"):
        scaler.load_state_dict(state["scaler"])
    rng = state["rng"]
    torch.set_rng_state(rng["torch"].cpu())
    if torch.cuda.is_available() and len(rng["cuda"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all([t.cpu() for t in rng["cuda"]])
    np.random.set_state(rng["numpy"])
    random.setstate(rng["python"])
    return state


def _worst_genre_f1(docs, thr, genre_map):
    """Compute the worst per-genre macro-F1 at the given threshold.

    Returns (worst_f1, {genre: f1}). Unknown doc IDs are grouped as 'unknown'.
    thr < 0 means hard predictions (CRF Viterbi) — uses macro_f1_preds.
    """
    from collections import defaultdict
    genre_docs = defaultdict(dict)
    for doc_id, (parr, gold) in docs.items():
        genre_docs[str(genre_map.get(doc_id, "unknown"))][doc_id] = (parr, gold)
    if thr < 0:  # CRF: parr already contains hard 0/1 predictions
        genre_f1 = {g: macro_f1_preds(gdocs)["f1"] for g, gdocs in genre_docs.items()}
    else:
        genre_f1 = {g: macro_f1_at(gdocs, thr)["f1"] for g, gdocs in genre_docs.items()}
    worst = min(genre_f1.values()) if genre_f1 else 0.0
    return worst, genre_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--with-artifacts", action="store_true",
                        help="also log best_*.pt checkpoints to MLflow as run artifacts (large)")
    # OOF stacking (closed-legal): train on train-minus-fold, emit held-out preds.
    parser.add_argument("--holdout-fold", type=int, default=-1,
                        help="OOF: hold this fold out of training (-1 = normal full-train run)")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=42)
    parser.add_argument("--oof-out", default="",
                        help="OOF: write held-out-fold per-word probs (JSON) here")
    parser.add_argument("--output-dir", default="",
                        help="override cfg.output_dir (per-fold dirs so checkpoints don't clobber)")
    # Session-chaining (Kaggle 12 h sessions; e67+). State is only written when
    # one of these is set, so KISSKI/local runs are byte-identical to before.
    parser.add_argument("--resume", action="store_true",
                        help="resume from <output_dir>/train_state.pt if it exists")
    parser.add_argument("--max-hours", type=float, default=0.0,
                        help="after any epoch that exceeds this wall-clock budget, save state and exit 0")
    parser.add_argument("--limit-batches", type=int, default=0,
                        help="smoke tests only: cap train batches per epoch (dev/test eval untouched)")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if cfg.lora_r > 0 and (cfg.swa or cfg.fgm):
        # SWA would clone the multi-GB frozen base; FGM's "word_embeddings" name
        # doesn't exist in LLM trunks (they use "embed_tokens") and would no-op.
        raise ValueError("swa/fgm are not supported with LoRA backbones — disable them in the config")
    if cfg.rdrop_alpha > 0.0 and cfg.lora_r == 0:
        raise ValueError("rdrop_alpha > 0 is designed for LoRA backbones - set lora_r > 0 or rdrop_alpha = 0")
    if cfg.head_type == "crf" and cfg.lora_r > 0:
        raise ValueError("head_type='crf' is not supported with LoRA backbones - use head_type='linear' for LLM configs")
    if cfg.fp16_scaler and cfg.fgm:
        raise ValueError("fp16_scaler + fgm is not supported (adversarial grads would be unscaled)")
    if cfg.paired_view_source and (cfg.naqta_restore or cfg.use_punct_aug or cfg.use_syntax
                                    or cfg.head_type == "crf" or cfg.task_mode == "multi"):
        raise ValueError("paired_view_source is not compatible with naqta_restore/"
                         "use_punct_aug/use_syntax/head_type=crf/task_mode=multi")
    if cfg.swa and (args.resume or args.max_hours > 0):
        raise ValueError("swa is not supported with --resume/--max-hours (averaged weights aren't in train_state.pt)")
    set_seed(cfg.seed)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("araseg")
    mlflow.start_run(run_name=cfg.experiment_id)
    mlflow.set_tag("experiment_id", cfg.experiment_id)
    mlflow.log_params({k: v for k, v in asdict(cfg).items() if v is not None})

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = resolve_model_path(cfg.model_name)
    syntax_flag = f"syntax({cfg.n_pos_tags}pos/{cfg.n_dep_rels}dep)" if cfg.use_syntax else "no-syntax"
    genre_flag = "worst-genre" if cfg.use_worst_genre else "mean-f1"
    print(f"[{cfg.experiment_id}] device={device}  mode={cfg.task_mode}  head={cfg.head_type}  "
          f"loss={cfg.loss_type}  fgm={cfg.fgm}  swa={cfg.swa}  "
          f"{syntax_flag}  ckpt-sel={genre_flag}")
    print(f"[{cfg.experiment_id}] model={model_path}")

    print(f"[{cfg.experiment_id}] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=cfg.trust_remote_code)
    if tokenizer.pad_token is None:           # LLaMA-style tokenizers ship without a pad token
        tokenizer.pad_token = tokenizer.eos_token

    # ── Syntax feature cache (Phase B) ──────────────────────────────────────
    syntax_caches_train: dict = {}
    syntax_caches_dev: dict = {}
    syntax_caches_test: dict = {}
    pos2id: dict = {}
    dep2id: dict = {}

    if cfg.use_syntax and cfg.syntax_cache_dir:
        import pickle
        cache_dir = Path(cfg.syntax_cache_dir)
        for st in cfg.subtasks:
            key = st.lower().replace("-", "_")
            vocab_path = cache_dir / f"{key}_vocab.json"
            if not vocab_path.exists():
                print(f"  [syntax] vocab not found for {st} at {vocab_path} — skipping syntax features")
                continue
            vocab = json.loads(vocab_path.read_text())
            # ID 0=PAD, 1=UNK, 2+ = real tags (matching data_utils._SYN_PAD/_SYN_UNK)
            pos2id = {t: i + 2 for i, t in enumerate(vocab["pos_vocab"])}
            dep2id = {t: i + 2 for i, t in enumerate(vocab["dep_vocab"])}
            pos2id["X"]    = 1   # explicit UNK
            dep2id["dep"]  = 1

            splits_stores = [("train", syntax_caches_train), ("dev", syntax_caches_dev),
                             ("test", syntax_caches_test)]
            for split_name, store in splits_stores:
                pth = cache_dir / f"{key}_{split_name}.pkl"
                if pth.exists():
                    store[st] = pickle.load(open(pth, "rb"))["cache"]
                    print(f"  [syntax] loaded {split_name} cache for {st} ({len(store[st])} docs)")
                else:
                    print(f"  [syntax] no {split_name} cache for {st} at {pth}")

    # ── Genre map for worst-genre checkpoint selection (Phase B) ─────────────
    genre_maps: dict = {}
    if cfg.use_worst_genre and cfg.genre_cache_dir:
        import json as _json
        for st in cfg.subtasks:
            key = st.lower().replace("-", "_")
            gpath = Path(cfg.genre_cache_dir) / f"{key}_dev.json"
            if gpath.exists():
                genre_maps[st] = _json.loads(gpath.read_text())
                print(f"  [genre] loaded {len(genre_maps[st])} doc genres for {st} from {gpath}")
            else:
                print(f"  [genre] genre map not found for {st} at {gpath} — using mean-F1 selection")

    # ── Track 2: punctuation-dropout augmented train records (Phase D) ─────────
    aug_records: dict = {}
    if cfg.use_punct_aug:
        from punct_aug import build_aug_docs, _NOPNX_OF
        target = _NOPNX_OF[cfg.punct_aug_source]
        aug_records[target] = build_aug_docs(cfg.punct_aug_source, "train",
                                             cfg.punct_drop_p, cfg.seed)
        print(f"[{cfg.experiment_id}] punct-aug: {len(aug_records[target])} {target} train docs "
              f"from {cfg.punct_aug_source} @ drop_p={cfg.punct_drop_p} "
              f"(aux={cfg.n_aux_classes} classes, w={cfg.aux_punct_weight})")

    # ── Naqta-restore: punctuation put back into train+dev input (Kareem) ──────
    # test is restored + scored separately (restore_then_segment.py --members) so the
    # boundary can map back to NoPnx space; here we only need train + dev-selection.
    restore_records: dict = {}
    restore_owners: dict = {}
    if cfg.naqta_restore:
        assert not cfg.use_punct_aug, "naqta_restore and punct_aug are different recipes"
        from naqta_restore import build_restored_docs
        for st in cfg.subtasks:
            restore_records[st] = {}
            restore_owners[st] = {}
            for split in ("train", "dev"):
                recs, owners = build_restored_docs(st, split, cfg.naqta_restore_min_p,
                                                    comma=cfg.naqta_comma)
                restore_records[st][split] = recs
                restore_owners[st][split] = owners
            print(f"[{cfg.experiment_id}] naqta-restore {st}: "
                  f"train {len(restore_records[st]['train'])} / dev "
                  f"{len(restore_records[st]['dev'])} docs @ min_p={cfg.naqta_restore_min_p}")

    # ── Paired-view consistency (Lane C Task 4): NoPnx train input paired with a
    # punctuation-projected view of the same words (equal word count per chunk), so
    # the two views can be trained together with a per-word KL term. Dev/test stay
    # plain official NoPnx-* -- the config guard above already rejects combining this
    # with naqta_restore/use_punct_aug/use_syntax/CRF/multitask.
    paired_records: dict = {}
    if cfg.paired_view_source:
        from punct_aug import build_paired_chunks, _NOPNX_OF
        target = _NOPNX_OF[cfg.paired_view_source]
        paired_records[target] = build_paired_chunks(cfg.paired_view_source, "train")
        print(f"[{cfg.experiment_id}] paired-view {target}: "
              f"{len(paired_records[target])} train chunks from {cfg.paired_view_source} "
              f"@ alpha={cfg.paired_consistency_alpha}")

    # OOF: for a held-out fold, train on every train doc EXCEPT that fold's docs
    # (per-subtask fold map — see oof.py). aug records aren't fold-filtered, so
    # OOF stacking of a punct-aug member isn't supported (none in our stack pools).
    oof_keep = {}
    if args.holdout_fold >= 0:
        from oof import keep_ids
        assert not cfg.use_punct_aug, "OOF fold-filtering not implemented for punct-aug members"
        oof_keep = {st: keep_ids(st, args.holdout_fold, args.n_folds, args.fold_seed)
                    for st in cfg.subtasks}
        for st in cfg.subtasks:
            print(f"[{cfg.experiment_id}] OOF fold {args.holdout_fold}/{args.n_folds}: "
                  f"[{st}] train on {len(oof_keep[st])} docs (holding out the rest)")

    print(f"[{cfg.experiment_id}] Loading datasets...")
    train_sets = {
        st: AraSegDataset(st, "train", tokenizer, cfg.max_length, cfg.window_stride,
                          syntax_cache=syntax_caches_train.get(st),
                          pos2id=pos2id or None, dep2id=dep2id or None,
                          records=(restore_records.get(st, {}).get("train")
                                   or aug_records.get(st) or paired_records.get(st)),
                          keep_doc_ids=oof_keep.get(st))
        for st in cfg.subtasks
    }
    dev_sets = {
        st: AraSegDataset(st, "dev",   tokenizer, cfg.max_length, cfg.window_stride,
                          syntax_cache=syntax_caches_dev.get(st),
                          pos2id=pos2id or None, dep2id=dep2id or None,
                          records=restore_records.get(st, {}).get("dev"))
        for st in cfg.subtasks
    }
    for st in cfg.subtasks:
        print(f"  [{st:10s}] train: {len(train_sets[st].doc_gold)} docs -> {len(train_sets[st])} windows  |  "
              f"dev: {len(dev_sets[st].doc_gold)} docs -> {len(dev_sets[st])} windows")

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    collate_fn = make_collate(pad_id)
    train_loaders = {st: make_loader(ds, cfg.batch_size, True,  cfg.num_workers, collate_fn) for st, ds in train_sets.items()}
    dev_loaders   = {st: make_loader(ds, cfg.batch_size, False, cfg.num_workers, collate_fn) for st, ds in dev_sets.items()}

    print(f"[{cfg.experiment_id}] Building model...")
    model = build_model(cfg, model_path).to(device)
    model.aux_weight = cfg.aux_punct_weight     # read by _step_loss when aux head present
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[{cfg.experiment_id}] params: trainable {n_train:,} / total {n_total:,} "
          f"({100 * n_train / max(n_total, 1):.2f}%)")

    if cfg.pretrain_from:
        print(f"[{cfg.experiment_id}] Loading pretrained ENCODER from {cfg.pretrain_from}")
        load_encoder_weights(model, cfg.pretrain_from, device)
        print(f"[{cfg.experiment_id}] Pretrained encoder loaded (task head trained fresh).")

    criterion = build_criterion(cfg, device)
    if cfg.llrd_decay < 1.0:
        optimizer = _build_llrd_optimizer(model, cfg)
        print(f"[{cfg.experiment_id}] LLRD optimizer: decay={cfg.llrd_decay}, "
              f"{len(optimizer.param_groups)} param groups")
    else:
        optimizer = AdamW((p for p in model.parameters() if p.requires_grad),
                          lr=cfg.lr, weight_decay=cfg.weight_decay)
    fgm = FGM(model, cfg.fgm_epsilon) if cfg.fgm else None

    steps_per_epoch = (
        len(train_loaders[cfg.subtasks[0]])
        if cfg.task_mode == "single"
        else max(len(l) for l in train_loaders.values())
    )
    # scheduler steps once per OPTIMIZER step, which is 1/accum batches
    accum = max(cfg.grad_accum, 1)
    total_steps  = cfg.epochs * (-(-steps_per_epoch // accum))
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = torch.amp.GradScaler("cuda") if cfg.fp16_scaler else None
    if scaler is not None or accum > 1:
        print(f"[{cfg.experiment_id}] fp16_scaler={cfg.fp16_scaler}  grad_accum={accum} "
              f"(effective batch {cfg.batch_size * accum})")

    # SWA: average weights over the tail of training. swa_start_epoch=0 => last 25%.
    swa_model = AveragedModel(model) if cfg.swa else None
    swa_start = cfg.swa_start_epoch if cfg.swa_start_epoch > 0 else int(cfg.epochs * 0.75) + 1

    best_f1: Dict[str, float] = {st: 0.0 for st in cfg.subtasks}
    best_thr: Dict[str, float] = {st: 0.5 for st in cfg.subtasks}
    history: Dict = {}

    t0 = time.time()
    chaining = args.resume or args.max_hours > 0
    state_path = out / "train_state.pt"
    start_epoch = 1
    if args.resume and state_path.exists():
        state = _load_train_state(state_path, model, optimizer, scheduler, scaler, device)
        best_f1, best_thr, history = state["best_f1"], state["best_thr"], state["history"]
        start_epoch = state["epoch"] + 1
        print(f"[{cfg.experiment_id}] resumed {state_path} -> continuing at epoch {start_epoch}/{cfg.epochs}")
    elif args.resume:
        print(f"[{cfg.experiment_id}] --resume: no {state_path} yet, starting fresh")

    for epoch in range(start_epoch, cfg.epochs + 1):
        print(f"\n[{cfg.experiment_id}] === Epoch {epoch}/{cfg.epochs} ===")

        if cfg.task_mode == "single":
            loss = train_single(model, train_loaders[cfg.subtasks[0]], optimizer, scheduler,
                                criterion, device, cfg.subtasks[0], cfg.grad_clip, fgm,
                                rdrop_alpha=cfg.rdrop_alpha,
                                paired_consistency_alpha=cfg.paired_consistency_alpha,
                                scaler=scaler, accum=accum,
                                limit_batches=args.limit_batches)
        else:
            loss = train_multi(model, train_loaders, optimizer, scheduler,
                               criterion, device, cfg.subtasks, steps_per_epoch, cfg.grad_clip,
                               fgm, rdrop_alpha=cfg.rdrop_alpha, scaler=scaler, accum=accum,
                               limit_batches=args.limit_batches)

        print(f"  train_loss={loss:.4f}")

        if swa_model is not None and epoch >= swa_start:
            swa_model.update_parameters(model)
            print(f"  [swa] averaged weights (epoch {epoch} >= {swa_start})")

        epoch_scores: Dict = {}
        for st in cfg.subtasks:
            # Full-document, stitched predictions; checkpoint on the dev-tuned F1.
            need_docs = cfg.use_worst_genre and st in genre_maps
            result = _eval_subtask(model, dev_loaders[st], device, st, dev_sets[st].doc_gold,
                                   return_docs=need_docs)
            if need_docs:
                s05, thr, s_best, docs = result
                worst_f1, genre_f1 = _worst_genre_f1(docs, thr, genre_maps[st])
                selection_f1 = worst_f1
                genre_str = "  genres: " + " ".join(f"{g}={v:.3f}" for g, v in sorted(genre_f1.items()))
            else:
                s05, thr, s_best = result
                selection_f1 = s_best["f1"]
                genre_str = ""
            epoch_scores[st] = {"thr0.5": s05, "tuned": s_best, "best_threshold": thr}
            print(f"  dev [{st:10s}]  F1@0.5={s05['f1']:.4f}  |  "
                  f"tuned@{thr:.2f}: P={s_best['precision']:.4f} R={s_best['recall']:.4f} "
                  f"F1={s_best['f1']:.4f}{genre_str}")
            if selection_f1 > best_f1[st]:
                best_f1[st] = selection_f1
                best_thr[st] = thr
                save_checkpoint(model, out / f"best_{st.replace('-','_')}.pt")

        history[f"epoch_{epoch}"] = {"train_loss": loss, "dev": epoch_scores}
        mlflow.log_metrics(
            {"train_loss": loss, **{f"dev_f1_{st}": s["tuned"]["f1"] for st, s in epoch_scores.items()}},
            step=epoch,
        )

        if chaining:
            _save_train_state(state_path, epoch, model, optimizer, scheduler, scaler,
                              best_f1, best_thr, history)
            elapsed_h = (time.time() - t0) / 3600.0
            if 0 < args.max_hours < elapsed_h and epoch < cfg.epochs:
                print(f"\n[{cfg.experiment_id}] BUDGET STOP after epoch {epoch} "
                      f"({elapsed_h:.2f}h > {args.max_hours}h) — state saved; "
                      f"rerun with --resume to continue.")
                mlflow.end_run()
                return

    # SWA finalize: evaluate the averaged model; if it beats the per-epoch best
    # on a subtask, it replaces that checkpoint (downstream test/ensemble code is
    # untouched — it always reads best_<subtask>.pt).
    if swa_model is not None:
        print(f"\n[{cfg.experiment_id}] === SWA finalize ===")
        swa_core = swa_model.module
        for st in cfg.subtasks:
            need_docs = cfg.use_worst_genre and st in genre_maps
            result = _eval_subtask(swa_core, dev_loaders[st], device, st, dev_sets[st].doc_gold,
                                   return_docs=need_docs)
            if need_docs:
                s05, thr, s_best, docs = result
                sel_f1, _ = _worst_genre_f1(docs, thr, genre_maps[st])
            else:
                s05, thr, s_best = result
                sel_f1 = s_best["f1"]
            tag = "kept" if sel_f1 > best_f1[st] else "worse than per-epoch best"
            print(f"  swa dev [{st:10s}] tuned@{thr:.2f} F1={s_best['f1']:.4f}  ({tag})")
            if sel_f1 > best_f1[st]:
                best_f1[st] = sel_f1
                best_thr[st] = thr
                save_checkpoint(swa_core, out / f"best_{st.replace('-','_')}.pt")

    # ── Test-set evaluation ──────────────────────────────────────────────────
    # Load best checkpoint per subtask, run on test split with dev-tuned threshold,
    # report F1 (if labels are non-trivial) and write submission CSV.
    print(f"\n[{cfg.experiment_id}] === Test Set Evaluation ===")
    test_f1: Dict[str, float] = {}
    test_results: Dict = {}

    for st in cfg.subtasks:
        ckpt_path = out / f"best_{st.replace('-','_')}.pt"
        load_checkpoint(model, ckpt_path, device)

        try:
            test_set = AraSegDataset(st, "test", tokenizer, cfg.max_length, cfg.window_stride,
                                     syntax_cache=syntax_caches_test.get(st),
                                     pos2id=pos2id or None, dep2id=dep2id or None)
        except Exception as e:
            print(f"  [{st:10s}] test set unavailable: {e}")
            continue

        test_loader = make_loader(test_set, cfg.batch_size, False, cfg.num_workers, collate_fn)
        is_crf = getattr(model, "head_type", "linear") == "crf"
        thr = best_thr[st]
        if is_crf:
            docs = collect_doc_viterbi(model, test_loader, device, st, test_set.doc_gold)
        else:
            docs = collect_doc_probs(model, test_loader, device, st, test_set.doc_gold)

        all_gold = np.concatenate([g for _, g in docs.values()])
        has_labels = int(all_gold.sum()) > 0

        if has_labels:
            if is_crf:
                s_thr = macro_f1_preds(docs)
                print(f"  [{st:10s}]  Viterbi: P={s_thr['precision']:.4f} "
                      f"R={s_thr['recall']:.4f} F1={s_thr['f1']:.4f}")
                test_results[st] = {"viterbi": s_thr}
            else:
                s05 = macro_f1_at(docs, 0.5)
                s_thr = macro_f1_at(docs, thr)
                print(f"  [{st:10s}]  F1@0.5={s05['f1']:.4f}  |  "
                      f"tuned@{thr:.2f}: P={s_thr['precision']:.4f} R={s_thr['recall']:.4f} F1={s_thr['f1']:.4f}")
                test_results[st] = {"thr0.5": s05, "tuned": s_thr, "threshold": thr}
            test_f1[st] = s_thr["f1"]
        else:
            print(f"  [{st:10s}]  test labels hidden — predictions saved only")
            test_results[st] = {"threshold": thr}

        # Submission CSV — official format: "Document ID","Prediction"
        pred_path = out / f"test_predictions_{st.replace('-','_')}.csv"
        with open(pred_path, "w") as f:
            f.write("Document ID,Prediction\n")
            for doc_id, (parr, _) in docs.items():
                preds = parr.astype(int) if is_crf else (parr >= thr).astype(int)
                f.write(f"{doc_id},{''.join(map(str, preds))}\n")
        print(f"  [{st:10s}]  predictions → {pred_path}")

    # ── OOF held-out predictions ─────────────────────────────────────────────
    # Predict on the fold this model did NOT train on, per subtask. The union of
    # these across all folds is the closed-legal train matrix fit_oof_stack.py
    # feeds to the stacker (in place of dev). CRF heads emit emissions, not
    # softmax, so they're excluded from stacking upstream — skip them here too.
    if args.holdout_fold >= 0 and args.oof_out:
        assert getattr(model, "head_type", "linear") != "crf", \
            "OOF stacking needs per-word softmax; CRF heads are excluded from the pools"
        from oof import holdout_ids
        oof_dump = {"fold": args.holdout_fold, "n_folds": args.n_folds,
                    "seed": args.fold_seed, "subtasks": {}}
        for st in cfg.subtasks:
            ho = holdout_ids(st, args.holdout_fold, args.n_folds, args.fold_seed)
            load_checkpoint(model, out / f"best_{st.replace('-','_')}.pt", device)
            if cfg.naqta_restore:
                # The model trained on restored text; mirror that at OOF time, then
                # map restored-space probabilities back onto original NoPnx tokens
                # so the OOF JSON stays in the schema fit_oof_stack.py expects.
                from cache_probs import map_restored_docs
                from restore_then_segment import _docs as _original_docs
                ho_set = AraSegDataset(st, "train", tokenizer, cfg.max_length, cfg.window_stride,
                                       syntax_cache=syntax_caches_train.get(st),
                                       pos2id=pos2id or None, dep2id=dep2id or None,
                                       records=restore_records[st]["train"], keep_doc_ids=ho)
                ho_loader = make_loader(ho_set, cfg.batch_size, False, cfg.num_workers, collate_fn)
                restored_docs = collect_doc_probs(model, ho_loader, device, st, ho_set.doc_gold)
                original_gold = {d: g for d, (_, g) in _original_docs(st, "train").items() if d in ho}
                owners = {d: o for d, o in restore_owners[st]["train"].items() if d in ho}
                docs = map_restored_docs(restored_docs, owners, original_gold)
            else:
                ho_set = AraSegDataset(st, "train", tokenizer, cfg.max_length, cfg.window_stride,
                                       syntax_cache=syntax_caches_train.get(st),
                                       pos2id=pos2id or None, dep2id=dep2id or None,
                                       keep_doc_ids=ho)
                ho_loader = make_loader(ho_set, cfg.batch_size, False, cfg.num_workers, collate_fn)
                docs = collect_doc_probs(model, ho_loader, device, st, ho_set.doc_gold)
            oof_dump["subtasks"][st] = {
                "probs": {d: [round(float(p), 6) for p in parr] for d, (parr, _) in docs.items()},
                "gold":  {d: [int(g) for g in gold] for d, (_, gold) in docs.items()},
            }
            print(f"  [{st:10s}]  OOF fold {args.holdout_fold}: {len(docs)} held-out docs")
        Path(args.oof_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.oof_out, "w") as f:
            json.dump(oof_dump, f)
        print(f"[{cfg.experiment_id}] OOF preds → {args.oof_out}")

    summary = {
        "experiment_id": cfg.experiment_id,
        "model_name": cfg.model_name,
        "task_mode": cfg.task_mode,
        "subtasks": cfg.subtasks,
        "features": {
            "head_type": cfg.head_type, "loss_type": cfg.loss_type,
            "fgm": cfg.fgm, "swa": cfg.swa,
            "rdrop_alpha": cfg.rdrop_alpha, "llrd_decay": cfg.llrd_decay,
            "pretrain_from": cfg.pretrain_from,
        },
        "best_dev_f1": best_f1,
        "best_threshold": best_thr,
        "best_test_f1": test_f1,
        "test": test_results,
        "history": history,
    }
    with open(out / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[{cfg.experiment_id}] Best dev F1 (tuned threshold):")
    for st, f1 in best_f1.items():
        print(f"  {st:12s}: {f1:.4f}  @thr={best_thr[st]:.2f}")
    if test_f1:
        print(f"[{cfg.experiment_id}] Best test F1 (dev threshold):")
        for st, f1 in test_f1.items():
            print(f"  {st:12s}: {f1:.4f}")
    print(f"Saved → {out}/results.json")

    mlflow.log_metrics({f"best_dev_f1_{st}": f1 for st, f1 in best_f1.items()})
    mlflow.log_metrics({f"best_threshold_{st}": thr for st, thr in best_thr.items()})
    mlflow.log_metrics({f"best_test_f1_{st}": f1 for st, f1 in test_f1.items()})
    if args.with_artifacts:
        for ckpt in out.glob("best_*.pt"):
            mlflow.log_artifact(str(ckpt))
    mlflow.end_run()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        if mlflow.active_run():
            mlflow.end_run(status="FAILED")
        raise
