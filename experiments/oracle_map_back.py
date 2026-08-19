"""Restore-then-segment ORACLE, reproducible on a laptop. No GPU, no torch.

Answers "what is the missing punctuation worth?" by mapping a PUNCTUATED lock's
already-written test predictions onto the NoPnx positions. Because NoPnx is the
punctuated text with punctuation tokens deleted (a strict subsequence), the map is
exact -- so this measures the cue, not an alignment penalty.

`restore_then_segment.py --mode gold` computes the same thing by RE-RUNNING the PA
lock's three members on GPU. That path is wired for PA only (NP's lock is an OOF
stack whose weights would also have to be borrowed), which left the NP->NoPnx-NP
oracle number in the ledger and the paper with no command behind it. This script
closes that: it reads the locked prediction CSVs instead of re-running models, so
both heads work and neither needs a GPU.

    python oracle_map_back.py              # both heads + the perfect-restore ceilings

Expected (test split): NoPnx-PA 0.9442 vs lock 0.8718, NoPnx-NP 0.9299 vs 0.8589.
"""
import argparse
from pathlib import Path

import numpy as np
from datasets import load_dataset

from punct_aug import _removed_flags
from restore_then_segment import map_back, restore_gold
from scoring import _doc_f1, _macro

# punctuated head, its locked test-prediction CSV, the NoPnx lock it is compared against
ORACLES = {
    "NoPnx-PA": ("PA", "outputs/phaseC_v2_pa_logit/test_predictions_PA.csv", 0.8718),
    "NoPnx-NP": ("NP", "outputs/oof_stack_sat_np/test_predictions_NP.csv", 0.8589),
}
_REPO = "MBZUAI/AraSeg-2026-Shared-Task-"


def _docs(subtask, split):
    """{doc_id: (tokens, labels)}. Loads the HF dataset directly -- data_utils pulls in
    torch, which the CPU-only tools deliberately avoid (see scoring.py)."""
    ds = load_dataset(_REPO + subtask, split=split)
    return {str(r["doc_id"]): (list(r["tokens"]), list(r["labels"])) for r in ds}


def _predictions(csv_path):
    """{doc_id: array} from a submission-format CSV (`Document ID,Prediction`, the
    prediction being one 0/1 character per token)."""
    out = {}
    for line in Path(csv_path).read_text(encoding="utf-8").splitlines()[1:]:
        if not line.strip():
            continue
        did, bits = line.split(",", 1)
        out[did.strip()] = np.array(list(bits.strip()), dtype=np.int64)
    return out


def oracle(nopnx_subtask, split="test", verbose=True):
    punctuated, csv_path, lock_f1 = ORACLES[nopnx_subtask]
    nopnx, pa = _docs(nopnx_subtask, split), _docs(punctuated, split)
    pred = _predictions(csv_path)

    mapped, ceiling = [], []
    for did, (ntok, ngold) in nopnx.items():
        ptok, pgold = pa[did]
        _, owner = restore_gold(ntok, ptok)
        assert len(pred[did]) == len(ptok), f"{did}: csv {len(pred[did])} vs {len(ptok)} tokens"
        ngold = np.asarray(ngold, np.int64)
        mapped.append(_doc_f1(map_back(pred[did], owner, len(ntok)), ngold))
        # ceiling: map the punctuated GOLD instead of the lock -- how lossless is the map?
        ceiling.append(_doc_f1(map_back(np.asarray(pgold, np.int64), owner, len(ntok)), ngold))

    m, c = _macro(mapped), _macro(ceiling)
    if verbose:
        print(f"[oracle] {punctuated} -> {nopnx_subtask} ({split}, {len(nopnx)} docs)")
        print(f"  perfect-restore ceiling   F1 {c['f1']:.4f}")
        print(f"  {punctuated} lock mapped over    P {m['precision']:.4f} R {m['recall']:.4f} "
              f"F1 {m['f1']:.4f}")
        print(f"  {nopnx_subtask} lock              F1 {lock_f1:.4f}"
              f"   -> headroom {100 * (m['f1'] - lock_f1):+.2f} F1\n")
    return m["f1"], c["f1"]


def _selfcheck():
    """The alignment is the whole result: an off-by-one silently deflates every number.
    Both ceilings must be ~1.0 (the map itself loses nothing) and NoPnx-PA must
    reproduce the 0.9442 that restore_then_segment.py --mode gold gets on GPU."""
    f1_pa, ceil_pa = oracle("NoPnx-PA", verbose=False)
    _, ceil_np = oracle("NoPnx-NP", verbose=False)
    assert ceil_pa > 0.99 and ceil_np > 0.99, f"map is lossy: {ceil_pa:.4f} / {ceil_np:.4f}"
    assert abs(f1_pa - 0.9442) < 0.001, f"NoPnx-PA oracle {f1_pa:.4f} != 0.9442 (GPU path)"
    print("selfcheck OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subtask", choices=sorted(ORACLES), default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        _selfcheck()
    else:
        for s in ([a.subtask] if a.subtask else sorted(ORACLES)):
            oracle(s, a.split)
