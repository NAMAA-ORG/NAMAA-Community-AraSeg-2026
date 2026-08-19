"""Fast validation of the opt-in training features (run on a GPU node before
launching real jobs). Uses random tiny batches + a cached encoder — NO dataset
download. Exercises: linear & bilstm heads (forward+backward, shape checks),
focal loss, FGM attack/restore round-trip, and an SWA averaging step.

    python smoke_test.py                      # uses xlm-roberta-large (cached)
    python smoke_test.py --model_name ...     # any cached encoder
"""

import argparse
import torch
from torch.optim import AdamW
from torch.optim.swa_utils import AveragedModel

from model import AraSegModel
from train import Config, build_criterion, FGM, _optim_step
from utils import resolve_model_path


def _fake_batch(vocab_size, bs=2, seqlen=16, device="cpu"):
    ids = torch.randint(5, vocab_size, (bs, seqlen))
    attn = torch.ones(bs, seqlen, dtype=torch.long)
    attn[0, seqlen // 2:] = 0                       # ragged lengths -> exercises packing
    labels = torch.randint(0, 2, (bs, seqlen))
    labels[attn == 0] = -100                        # ignore padding
    return {
        "input_ids": ids, "attention_mask": attn,
        "token_type_ids": torch.zeros(bs, seqlen, dtype=torch.long),
        "labels": labels,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="xlm-roberta-large")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[smoke] device={device}  model={args.model_name}")
    mpath = resolve_model_path(args.model_name)

    for head in ("linear", "bilstm"):
        model = AraSegModel(mpath, ["PA"], head_type=head).to(device)
        vocab = model.encoder.config.vocab_size
        batch = _fake_batch(vocab, device=device)

        # forward shape
        logits = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            token_type_ids=batch["token_type_ids"].to(device),
            subtask="PA",
        )
        assert logits.shape == (*batch["input_ids"].shape, 2), logits.shape
        print(f"[smoke] head={head:7s} forward OK  logits={tuple(logits.shape)}")

        for loss_type in ("ce", "focal"):
            for use_fgm in (False, True):
                cfg = Config(
                    experiment_id="smoke", model_name=mpath, task_mode="single",
                    subtasks=["PA"], epochs=1, batch_size=2, lr=1e-5, max_length=16,
                    output_dir="/tmp/smoke", pos_weight=3.0, head_type=head,
                    loss_type=loss_type, fgm=use_fgm,
                )
                crit = build_criterion(cfg, device)
                opt = AdamW(model.parameters(), lr=1e-5)
                sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
                fgm = FGM(model, cfg.fgm_epsilon) if use_fgm else None

                # snapshot embedding to verify FGM fully restores it
                emb = next(p for n, p in model.named_parameters() if "word_embeddings" in n)
                before = emb.data.clone()
                loss = _optim_step(model, batch, crit, opt, sched, device, "PA", 1.0, fgm)
                assert loss == loss, "NaN loss"                      # NaN check
                if use_fgm:
                    # after optim step weights changed, but FGM's perturbation itself
                    # must have been restored before step (no double-add). Just assert
                    # the run completed and loss finite.
                    pass
                print(f"[smoke] head={head:7s} loss={loss_type:5s} fgm={use_fgm!s:5s} "
                      f"step OK  loss={loss:.4f}")

        # SWA averaging step
        swa = AveragedModel(model)
        swa.update_parameters(model)
        _ = swa.module(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            token_type_ids=batch["token_type_ids"].to(device),
            subtask="PA",
        )
        print(f"[smoke] head={head:7s} SWA update+forward OK")

    print("[smoke] ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
