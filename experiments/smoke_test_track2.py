"""Smoke test for Track 2 (punctuation-dropout aug + aux punctuation head).

Runs on any machine with torch + a tiny HF model (CPU is fine). Verifies the
whole path the configs e60-e63 exercise: aug records -> dataset aux tensors ->
model aux head -> _step_loss aux branch, plus the no-aux fallback.

    python smoke_test_track2.py
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from punct_aug import build_aug_docs, N_AUX_CLASSES
from data_utils import AraSegDataset, make_collate
from model import AraSegModel
import train as T

M = "hf-internal-testing/tiny-random-bert"


def main():
    tok = AutoTokenizer.from_pretrained(M)
    assert tok.is_fast, "need a fast tokenizer"

    recs = build_aug_docs("PA", "train", drop_p=0.5, seed=1)[:6]
    assert all("aux_labels" in r for r in recs), "aug records missing aux_labels"

    ds = AraSegDataset("NoPnx-PA", "train", tok, max_length=64, window_stride=16, records=recs)
    ex = ds[0]
    assert "aux_labels" in ex and ex["aux_labels"].shape == ex["labels"].shape
    print("[1] dataset emits aux_labels aligned to labels OK")

    col = make_collate(tok.pad_token_id or 0)
    b = next(iter(DataLoader(ds, batch_size=3, collate_fn=col)))
    assert b["aux_labels"].shape == b["labels"].shape
    print("[2] collate pads aux_labels OK", tuple(b["aux_labels"].shape))

    m = AraSegModel(M, ["NoPnx-PA"], n_aux_classes=N_AUX_CLASSES)
    m.aux_weight = 0.5
    logits, aux = m(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                    token_type_ids=b["token_type_ids"], subtask="NoPnx-PA", return_aux=True)
    assert aux.shape[-1] == N_AUX_CLASSES and logits.shape[-1] == 2
    print("[3] forward return_aux OK: logits", tuple(logits.shape), "aux", tuple(aux.shape))

    plain = m(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
              token_type_ids=b["token_type_ids"], subtask="NoPnx-PA")
    assert torch.is_tensor(plain), "return_aux=False must return a plain tensor"
    print("[4] plain forward (return_aux=False) OK")

    crit = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss = T._step_loss(m, b, crit, "cpu", "NoPnx-PA")
    assert torch.isfinite(loss), "aux step loss not finite"
    print("[5] _step_loss aux branch OK:", float(loss))

    # no-aux fallback: records without aux_labels, model without aux head
    recs2 = [{"doc_id": r["doc_id"], "tokens": r["tokens"], "labels": r["labels"]} for r in recs]
    ds2 = AraSegDataset("NoPnx-PA", "train", tok, max_length=64, window_stride=16, records=recs2)
    assert "aux_labels" not in ds2[0]
    m2 = AraSegModel(M, ["NoPnx-PA"])
    b2 = next(iter(DataLoader(ds2, batch_size=3, collate_fn=col)))
    loss2 = T._step_loss(m2, b2, crit, "cpu", "NoPnx-PA")
    assert torch.isfinite(loss2)
    print("[6] no-aux fallback OK:", float(loss2))

    print("ALL TRACK-2 SMOKE CHECKS PASS")


if __name__ == "__main__":
    main()
