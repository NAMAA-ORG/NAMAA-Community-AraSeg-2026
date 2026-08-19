"""Naqta punctuation-restoration probabilities as an ensemble member / decoder feature.

`MostafaMaroof/Naqta` is XLM-R-large fine-tuned to predict the punctuation mark that
follows each word of unpunctuated Arabic (8 classes, macro-F1 0.896, MIT). The NoPnx
subtasks strip exactly that cue, and the restore-then-segment oracle says the whole
NoPnx deficit is the missing punctuation (see the paper's RQ3 section). So Naqta's
P(sentence-terminal mark after word i) is a natural extra column for `fit_decoder.py`.

Emits the SAME {doc_id: (prob_per_word, gold_per_word)} pickle every other member uses,
so ensemble.py / fit_decoder.py / bootstrap_paired.py read it with no special-casing:

    outputs/prob_cache/<Subtask>_<split>_naqta.pkl        P(terminal) - drop-in member
    outputs/prob_cache/<Subtask>_<split>_naqta_full.pkl   all 8 class probs (fp16)

The `_full` file costs one extra dump and saves a second GPU pass when we want to ask
"does the comma channel help too?" - the class subset is then a CPU-side decision.

LEGALITY (decided 2026-07-23): usable in CLOSED. Naqta is a public pretrained model, and
the organizer ruling allows finetuning a pretrained model as long as OUR finetuning data
is AraSeg train -- which it is. The earlier "open track only" note here treated Naqta's
own external pretraining as disqualifying; that test would equally disqualify `sat_ft`
(wtpsplit, external data), which is a weighted member of three locks already. Consistent
line: external data in a public upstream checkpoint is fine, external data in OUR
finetuning is not.

Usage (GPU only for this step; the decoder fit that consumes it is CPU-only):
    python naqta_cache.py --subtasks NoPnx-PA NoPnx-NP --splits train dev test
    python naqta_cache.py --selfcheck        # alignment + signal-direction gate
"""
import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from data_utils import AraSegDataset, make_collate

MODEL_ID = "MostafaMaroof/Naqta"
# id2label: 0=O 1=. 2=, 3=? 4=! 5=: 6=; 7=-   (2/3/6 are the Arabic forms)
TERMINAL_IDS = (1, 3, 4)   # . ? !  -- the marks that end a sentence/paragraph


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based ROC AUC. No sklearn dependency, ties handled by average rank."""
    pos, neg = int(labels.sum()), int((labels == 0).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks within tie groups so a constant score scores 0.5, not 1.0
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


@torch.no_grad()
def naqta_doc_probs(model, loader, device, doc_gold):
    """Windows -> per-document (n_words, 8) class probabilities.

    Mirrors metrics.collect_doc_probs: where windows overlap, keep the prediction
    from the window in which the word sits closest to the centre (most context on
    both sides). Words no window covered stay at the 'O' class.
    """
    model.eval()
    store = defaultdict(dict)                    # doc_id -> word_idx -> (probs, dist)
    for batch in loader:
        logits = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        ).logits
        probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()   # (B, T, 8)
        word_idx = batch["word_idx"].numpy()
        attn = batch["attention_mask"].numpy()
        for i in range(probs.shape[0]):
            length = int(attn[i].sum())
            centre = length / 2.0
            d_store = store[batch["doc_ids"][i]]
            for p in range(length):
                g = int(word_idx[i, p])
                if g < 0:
                    continue
                dist = abs(p - centre)
                prev = d_store.get(g)
                if prev is None or dist < prev[1]:
                    d_store[g] = (probs[i, p], dist)

    out = {}
    for doc_id, gold in doc_gold.items():
        n = len(gold)
        arr = np.zeros((n, probs.shape[-1]), dtype=np.float32)
        arr[:, 0] = 1.0                          # uncovered words default to 'O'
        for g, (pv, _) in store.get(doc_id, {}).items():
            if 0 <= g < n:
                arr[g] = pv
        out[doc_id] = (arr, np.asarray(gold, dtype=np.int64))
    return out


def run(subtasks, splits, out_dir, batch_size, max_length, window_stride,
        overwrite=False, limit_docs=None, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_ID)
    if device == "cuda":
        model = model.half()                     # XLM-R-large fp16 ~1.2GB, fits a T4/6GB
    model.to(device)

    results = {}
    for subtask in subtasks:
        for split in splits:
            tag = f"{subtask.replace('-', '_')}_{split}"
            fp = out_dir / f"{tag}_naqta.pkl"
            if fp.exists() and not overwrite and limit_docs is None:
                print(f"  [skip] {fp.name} exists")
                continue

            ds = AraSegDataset(subtask, split, tok, max_length, window_stride)
            if limit_docs is not None:
                keep = sorted(ds.doc_gold)[:limit_docs]
                ds.examples = [e for e in ds.examples if e["doc_id"] in set(keep)]
                ds.doc_gold = {k: ds.doc_gold[k] for k in keep}
            loader = torch.utils.data.DataLoader(
                ds, batch_size=batch_size, shuffle=False,
                collate_fn=make_collate(tok.pad_token_id or 0), num_workers=0,
            )
            docs = naqta_doc_probs(model, loader, device, ds.doc_gold)

            term = {d: (p[:, list(TERMINAL_IDS)].sum(axis=1).astype(np.float32), g)
                    for d, (p, g) in docs.items()}
            gold_all = np.concatenate([g for _, g in term.values()])
            score_all = np.concatenate([s for s, _ in term.values()])
            auc = _auc(score_all, gold_all) if gold_all.sum() else float("nan")
            results[(subtask, split)] = auc

            if limit_docs is None:
                with open(fp, "wb") as f:
                    pickle.dump(term, f)
                with open(out_dir / f"{tag}_naqta_full.pkl", "wb") as f:
                    pickle.dump({d: (p.astype(np.float16), g) for d, (p, g) in docs.items()}, f)
            print(f"  [ok  ] {tag}: {len(docs)} docs, "
                  f"P(terminal) AUC vs gold boundaries = {auc:.4f}")
    return results


def selfcheck():
    """Alignment + signal-direction gate on a few real docs.

    The AUC assertion is the one that matters: if window pooling or the word_ids
    mapping were off by even one position, P(terminal) would stop lining up with
    gold boundaries and this drops to ~0.5. A shape check alone would not catch that.
    """
    res = run(["NoPnx-PA"], ["dev"], out_dir="outputs/prob_cache", batch_size=4,
              max_length=512, window_stride=128, limit_docs=8)
    auc = res[("NoPnx-PA", "dev")]
    assert not np.isnan(auc), "no positive labels in the sample - bad split?"
    assert auc > 0.75, f"P(terminal) barely tracks gold boundaries (AUC {auc:.3f}) - alignment bug?"
    print(f"SELF-CHECK OK - AUC {auc:.4f} on 8 dev docs (nothing written to disk)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subtasks", nargs="+", default=["NoPnx-PA", "NoPnx-NP"])
    ap.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    ap.add_argument("--out_dir", default="outputs/prob_cache")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--window_stride", type=int, default=128)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        selfcheck()
        return
    print(f"[naqta] {MODEL_ID} -> {args.subtasks} x {args.splits}")
    res = run(args.subtasks, args.splits, args.out_dir, args.batch_size,
              args.max_length, args.window_stride, args.overwrite)
    print("\nAUC of P(terminal) against gold boundaries (0.5 = no signal):")
    for (s, sp), a in sorted(res.items()):
        print(f"  {s:10} {sp:6} {a:.4f}")


if __name__ == "__main__":
    main()
