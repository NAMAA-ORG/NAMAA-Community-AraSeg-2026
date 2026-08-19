"""Smoke test for LLM backbones (run BEFORE queueing e31+ jobs).

Verifies, with a small cached LLM (default Qwen/Qwen2.5-0.5B, ~1 GB):
  1. tokenizer: fast, supports is_split_into_words + overflow + word_ids on Arabic
  2. causal control: with bidirectional=False, position 0 logits IGNORE later tokens
  3. bidirectionality: with bidirectional=True, position 0 logits REACT to later tokens
  4. LoRA: trainable param ratio < 5%
  5. checkpoint: save_checkpoint writes adapters+heads only (small file), roundtrips
  6. one optimizer step with gradient checkpointing runs and yields a finite loss

    python smoke_test_llm.py                                  # CPU ok (fp32)
    python smoke_test_llm.py --model_name Qwen/Qwen2.5-0.5B
"""

import argparse
import tempfile
from pathlib import Path

import torch
from torch.optim import AdamW
from transformers import AutoTokenizer

from model import AraSegModel, build_model, save_checkpoint, load_checkpoint
from train import Config, build_criterion, _optim_step
from utils import resolve_model_path

ARABIC_WORDS = "ذهب الولد إلى المدرسة صباحا ثم عاد إلى البيت مساء وقرأ كتابه".split()


def _probe_batch(device, seqlen=12, lo=100, hi=1000):
    torch.manual_seed(0)
    ids = torch.randint(lo, hi, (1, seqlen)).to(device)
    mask = torch.ones_like(ids)
    return ids, mask


def reacts_to_right_context(model, device) -> bool:
    """True iff position 0's logits change when the LAST token changes."""
    model.eval()
    ids, mask = _probe_batch(device)
    ids2 = ids.clone()
    ids2[0, -1] += 1
    with torch.no_grad():
        h1 = model(input_ids=ids, attention_mask=mask, subtask="PA")
        h2 = model(input_ids=ids2, attention_mask=mask, subtask="PA")
    return not torch.allclose(h1[0, 0], h2[0, 0], atol=1e-5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = "bfloat16" if device == "cuda" else "float32"
    mpath = resolve_model_path(args.model_name)
    print(f"[smoke-llm] device={device}  dtype={dtype}  model={args.model_name}")

    # 1 ── tokenizer: windowing + word alignment on Arabic
    tok = AutoTokenizer.from_pretrained(mpath)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    assert tok.is_fast, "need a fast tokenizer (word_ids/overflow)"
    enc = tok(ARABIC_WORDS, is_split_into_words=True, max_length=8, truncation=True,
              stride=2, return_overflowing_tokens=True, padding=False)
    assert len(enc["input_ids"]) > 1, "expected multiple overflow windows"
    wids = enc.word_ids(batch_index=0)
    assert any(w is not None for w in wids), "word_ids() empty"
    print(f"[smoke-llm] 1/6 tokenizer OK ({len(enc['input_ids'])} windows, pad={tok.pad_token!r})")

    # 2 ── causal control: base behavior must NOT see right context
    causal = AraSegModel(mpath, ["PA"], torch_dtype=dtype, bidirectional=False).to(device)
    assert not reacts_to_right_context(causal, device), \
        "causal model reacted to right context — mask handling is broken"
    print("[smoke-llm] 2/6 causal control OK (no right-context leak)")
    del causal

    # 3+4 ── bidirectional + LoRA model
    model = AraSegModel(mpath, ["PA"], torch_dtype=dtype,
                        bidirectional=True, lora_r=8).to(device)
    assert reacts_to_right_context(model, device), \
        "bidirectional=True had no effect — 4D mask not reaching attention"
    print("[smoke-llm] 3/6 bidirectionality probe OK")

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    assert n_train / n_total < 0.05, f"base not frozen: {n_train/n_total:.1%} trainable"
    print(f"[smoke-llm] 4/6 LoRA OK ({n_train:,}/{n_total:,} = {n_train/n_total:.2%} trainable)")

    # 5 ── slim checkpoint roundtrip
    with tempfile.TemporaryDirectory() as td:
        ckpt = Path(td) / "ckpt.pt"
        save_checkpoint(model, ckpt)
        size_mb = ckpt.stat().st_size / 1e6
        full_mb = sum(v.numel() * v.element_size() for v in model.state_dict().values()) / 1e6
        assert size_mb < 0.5 * full_mb, f"checkpoint not slim: {size_mb:.0f}MB vs model {full_mb:.0f}MB"
        ids, mask = _probe_batch(device)
        model.eval()
        with torch.no_grad():
            before = model(input_ids=ids, attention_mask=mask, subtask="PA")
        load_checkpoint(model, ckpt, device)
        with torch.no_grad():
            after = model(input_ids=ids, attention_mask=mask, subtask="PA")
        assert torch.allclose(before, after, atol=1e-5), "roundtrip changed outputs"
    print(f"[smoke-llm] 5/6 slim checkpoint roundtrip OK ({size_mb:.0f} MB)")
    del model

    # 6 ── one training step via the factory, with gradient checkpointing
    cfg = Config(experiment_id="smoke-llm", model_name=mpath, task_mode="single",
                 subtasks=["PA"], epochs=1, batch_size=2, lr=1e-4, max_length=32,
                 output_dir="outputs/smoke-llm", pos_weight=3.0,
                 torch_dtype=dtype, bidirectional=True, lora_r=8,
                 gradient_checkpointing=True)
    model = build_model(cfg, mpath).to(device)
    model.train()
    torch.manual_seed(0)
    bs, sl = 2, 16
    batch = {
        "input_ids": torch.randint(100, 1000, (bs, sl)),
        "attention_mask": torch.ones(bs, sl, dtype=torch.long),
        "token_type_ids": torch.zeros(bs, sl, dtype=torch.long),
        "labels": torch.randint(0, 2, (bs, sl)),
    }
    batch["attention_mask"][0, sl // 2:] = 0      # ragged lengths
    batch["labels"][batch["attention_mask"] == 0] = -100
    crit = build_criterion(cfg, device)
    opt = AdamW((p for p in model.parameters() if p.requires_grad), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
    loss = _optim_step(model, batch, crit, opt, sched, device, "PA", 1.0, None)
    assert loss == loss and loss < float("inf"), f"bad loss: {loss}"
    print(f"[smoke-llm] 6/6 train step OK (loss={loss:.4f}, grad ckpt on)")

    print("[smoke-llm] ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
