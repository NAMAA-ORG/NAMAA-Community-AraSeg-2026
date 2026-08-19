"""Q9: boundary-verifier cross-encoder over the locked decoder's ambiguous band.

`band_mass.py` established the ceiling: on NoPnx-PA, 4.14% of tokens carry 71.6% of all
false positives, and a delete-only oracle over that band is worth +4.29 F1 with recall
flat. This trains the real thing.

Why a cross-encoder and not another feature: the locked decoder already has all 14 of its
own features, so a second model over the same inputs learns nothing. The verifier's only
new information is **the text itself** — the words on each side of the candidate, which
the decoder never sees. Framing is left-context [SEP] right-context, so the segment split
IS the candidate boundary. Q9's original objection (chicken-and-egg at inference) does not
apply: the candidate positions are handed to us by the decoder.

Legality (closed track): the verifier is fine-tuned ONLY on candidates drawn from
out-of-fold TRAIN predictions (`outputs/oof/`, `outputs/oof_sat/`) with AraSeg train text
and train gold. Dev is used only to pick the application mode and one threshold — ordinary
model selection, identical to how every existing lock chose its threshold. Test/blind are
never fit on. CAMeLBERT is a pretrained backbone, which the organizer ruling permits.

    Implementation note: plain AdamW loop, no Trainer, no scheduler tricks. ~5k candidates and a
    108M encoder means this is a minutes-scale job; the upgrade path if it wins is a
    2-feature blend of [decoder posterior, verifier logit] instead of a hard override.

Run the plumbing check FIRST (no GPU, no training) — it swaps in an oracle verifier and
must reproduce band_mass.py's oracle numbers exactly:

    python verify_band.py --subtask NoPnx-PA --oracle

Then the real run:

    python verify_band.py --subtask NoPnx-PA --output_dir outputs/verify_nopnx_pa
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from band_mass import DECODER_DIR, _load_members, replay
from fit_decoder import build_docs
from scoring import _doc_f1, _macro

MODEL = "CAMeL-Lab/bert-base-arabic-camelbert-msa"
MODES = ("delete_only", "both_ways")


def load_text(subtask, split):
    """{doc_id: [words]} straight from the HF corpus (no windowing needed here).

    AraSeg names its splits train/dev/test -- there is no 'validation'. Accept the
    usual aliases anyway rather than hardcoding, which is what broke job 15104194.
    """
    from datasets import load_dataset
    from data_utils import SUBTASK_DATASETS

    ds = load_dataset(SUBTASK_DATASETS[subtask])
    for name in (split, {"dev": "validation", "validation": "dev"}.get(split)):
        if name in ds:
            return {r["doc_id"]: r["tokens"] for r in ds[name]}
    raise KeyError(f"split {split!r} not in {list(ds)}")


def check_alignment(docs, texts, name):
    """The cached probability arrays and the raw corpus must agree on token count,
    or render() silently windows the wrong words around the candidate."""
    for doc_id, (s, _) in docs.items():
        n = len(texts.get(doc_id, ()))
        if n != s.shape[0]:
            raise ValueError(f"{name}: {doc_id} has {s.shape[0]} cached tokens but "
                             f"{n} corpus tokens -- render() would misalign")


def oof_docs(subtask, members, oof_dir, sat_oof_dir, n_folds, fold_seed):
    """Out-of-fold TRAIN member predictions, in lock order -> build_docs form."""
    from fit_oof_stack import _load_member_oof, _load_sat_oof

    loaded = []
    for mid in members:
        if mid == "sat_ft":
            loaded.append(_load_sat_oof(Path(sat_oof_dir), subtask, n_folds, fold_seed))
        else:
            loaded.append(_load_member_oof(mid, subtask, Path(oof_dir), n_folds, fold_seed))
    return build_docs(loaded)


def candidates(docs, w, b, thr, band):
    """[(doc_id, idx, gold_label)] for every token whose decoder posterior is in band."""
    lo, hi = band
    out = []
    for doc_id, (s, g) in docs.items():
        _, post = replay(s, w, b, thr)
        for i in np.flatnonzero((post >= lo) & (post <= hi)):
            out.append((doc_id, int(i), int(g[i])))
    return out


def render(words, idx, win):
    """Left/right text pair around a boundary that would fall AFTER word `idx`."""
    left = " ".join(words[max(0, idx - win + 1): idx + 1])
    right = " ".join(words[idx + 1: idx + 1 + win])
    return left, right or "."


def encode(cands, texts, tok, win, max_len):
    pairs = [render(texts[d], i, win) for d, i, _ in cands]
    enc = tok([l for l, _ in pairs], [r for _, r in pairs], truncation=True,
              max_length=max_len, padding=True, return_tensors="pt")
    return enc


def train_verifier(cands, texts, args, device):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    enc = encode(cands, texts, tok, args.window, args.max_len)
    y = torch.tensor([c[2] for c in cands], dtype=torch.long)
    pos = int(y.sum())
    print(f"[verify] train candidates {len(y)}  positives {pos} ({100.0 * pos / len(y):.1f}%)")

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2).to(device)
    ds = TensorDataset(enc["input_ids"], enc["attention_mask"], enc["token_type_ids"], y)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # class weights: the band is far more balanced than the raw token stream, but not
    # perfectly — weight so neither direction is learned away.
    wts = torch.tensor([len(y) / (2.0 * (len(y) - pos)), len(y) / (2.0 * max(pos, 1))]).to(device)
    lossf = torch.nn.CrossEntropyLoss(weight=wts)

    model.train()
    for ep in range(args.epochs):
        tot = 0.0
        for ids, am, tt, yy in dl:
            opt.zero_grad(set_to_none=True)
            out = model(input_ids=ids.to(device), attention_mask=am.to(device),
                        token_type_ids=tt.to(device))
            loss = lossf(out.logits, yy.to(device))
            loss.backward()
            opt.step()
            tot += float(loss) * len(yy)
        print(f"[verify] epoch {ep + 1}/{args.epochs} loss {tot / len(y):.4f}")
    return model, tok


def verifier_probs(model, tok, cands, texts, args, device):
    if model is None:  # oracle plumbing check: the verifier is simply right
        return np.array([float(c[2]) for c in cands])
    import torch

    enc = encode(cands, texts, tok, args.window, args.max_len)
    model.eval()
    out = []
    with torch.no_grad():
        for a in range(0, len(cands), args.batch_size):
            sl = slice(a, a + args.batch_size)
            lg = model(input_ids=enc["input_ids"][sl].to(device),
                       attention_mask=enc["attention_mask"][sl].to(device),
                       token_type_ids=enc["token_type_ids"][sl].to(device)).logits
            out.append(torch.softmax(lg, -1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def apply_and_score(docs, w, b, thr, cands, probs, mode, vthr):
    """Re-decode with the verifier's opinion forced on in-band tokens."""
    ov = {}
    for (doc_id, i, _), p in zip(cands, probs):
        ov.setdefault(doc_id, {})[i] = int(p >= vthr)
    rows, preds_out = [], {}
    for doc_id, (s, g) in docs.items():
        o = ov.get(doc_id, {})
        if mode == "delete_only":
            base, _ = replay(s, w, b, thr)
            o = {i: v for i, v in o.items() if v == 0 and base[i] == 1}
        preds = replay(s, w, b, thr, override=o)[0]
        preds_out[doc_id] = preds
        rows.append(_doc_f1(preds, g))
    return _macro(rows), preds_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subtask", required=True, choices=sorted(DECODER_DIR))
    ap.add_argument("--band", type=float, nargs=2, default=(0.2, 0.8))
    ap.add_argument("--all-tokens", dest="all_tokens", action="store_true",
                    help="Train on EVERY train token, not just in-band ones (~30x more "
                         "supervision); application is still in-band only. The first run "
                         "trained on 3253 candidates and memorized them by epoch 4, which "
                         "cannot distinguish 'local context carries no signal' from 'too "
                         "few examples to learn it'. This is the control for that claim.")
    ap.add_argument("--oracle", action="store_true",
                    help="No training: use gold as the verifier. Must reproduce band_mass.py.")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--window", type=int, default=40, help="words of context each side")
    ap.add_argument("--max_len", type=int, default=192)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--cache_dir", default="outputs/prob_cache")
    ap.add_argument("--sat_cache_dir", default="outputs/sat_fullft_caches")
    ap.add_argument("--oof_dir", default="outputs/oof")
    ap.add_argument("--sat_oof_dir", default="outputs/oof_sat")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--fold_seed", type=int, default=42)
    ap.add_argument("--test-split", dest="test_split", default="test")
    ap.add_argument("--output_dir", default=None)
    args = ap.parse_args()
    st_us = args.subtask.replace("-", "_")

    meta = json.load(open(Path(DECODER_DIR[args.subtask]) / f"decoder_{st_us}.json"))
    w = np.array([meta["weights"][n] for n in meta["feature_names"]], dtype=np.float64)
    b, thr, members = meta["bias"], meta["dev_threshold"], meta["members"]

    def split_docs(split):
        return build_docs(_load_members(args.cache_dir, args.sat_cache_dir,
                                        args.subtask, split, members))

    dev_docs, test_docs = split_docs("dev"), split_docs(args.test_split)
    lock_dev = _macro([_doc_f1(replay(s, w, b, thr)[0], g) for s, g in dev_docs.values()])
    lock_test = _macro([_doc_f1(replay(s, w, b, thr)[0], g) for s, g in test_docs.values()])
    assert abs(lock_test["f1"] - meta[args.test_split]["f1"]) < 1e-6, "replay != lock"
    print(f"[verify] lock  dev F1 {lock_dev['f1']:.4f}  {args.test_split} F1 {lock_test['f1']:.4f}")

    dev_c = candidates(dev_docs, w, b, thr, args.band)
    test_c = candidates(test_docs, w, b, thr, args.band)

    model = tok = device = None
    texts = {}
    if args.oracle:
        print("[verify] ORACLE mode — plumbing check only, nothing is trained")
    else:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        train_docs = oof_docs(args.subtask, members, args.oof_dir, args.sat_oof_dir,
                              args.n_folds, args.fold_seed)
        # Load and check every text split BEFORE training: a bad split name or a
        # tokenization mismatch should cost seconds, not a completed training run.
        texts = {sp: load_text(args.subtask, sp) for sp in ("train", "dev", args.test_split)}
        for sp, d in (("train", train_docs), ("dev", dev_docs), (args.test_split, test_docs)):
            check_alignment(d, texts[sp], sp)
        print(f"[verify] text splits loaded and aligned: "
              + ", ".join(f"{sp}={len(t)} docs" for sp, t in texts.items()))
        # Training draws from the whole document with --all-tokens; application stays
        # in-band either way (that is where the headroom is).
        train_band = (0.0, 1.0) if args.all_tokens else args.band
        train_c = candidates(train_docs, w, b, thr, train_band)
        print(f"[verify] training band {train_band} "
              f"({'ALL tokens' if args.all_tokens else 'in-band only'})")
        model, tok = train_verifier(train_c, texts["train"], args, device)

    dev_p = verifier_probs(model, tok, dev_c, texts.get("dev"), args, device)
    test_p = verifier_probs(model, tok, test_c, texts.get(args.test_split), args, device)

    # ── dev picks mode + threshold; test is only reported ────────────────────
    best = None
    for mode in MODES:
        for vthr in np.arange(0.05, 0.96, 0.05):
            sc, _ = apply_and_score(dev_docs, w, b, thr, dev_c, dev_p, mode, float(vthr))
            if best is None or sc["f1"] > best[0]["f1"]:
                best = (sc, mode, float(vthr))
    dev_best, mode, vthr = best
    print(f"[verify] dev picks mode={mode} vthr={vthr:.2f}: P={dev_best['precision']:.4f} "
          f"R={dev_best['recall']:.4f} F1={dev_best['f1']:.4f} "
          f"(lock {lock_dev['f1']:.4f}, {100 * (dev_best['f1'] - lock_dev['f1']):+.2f})")

    test_sc, test_preds = apply_and_score(test_docs, w, b, thr, test_c, test_p, mode, vthr)
    print(f"[verify] {args.test_split.upper()} @{mode}/{vthr:.2f}: P={test_sc['precision']:.4f} "
          f"R={test_sc['recall']:.4f} F1={test_sc['f1']:.4f} "
          f"({100 * (test_sc['f1'] - lock_test['f1']):+.2f} vs lock)")

    if args.oracle:
        exp = {"delete_only": "oracle FP-only", "both_ways": "oracle both ways"}[mode]
        print(f"[verify] plumbing OK if {args.test_split} F1 matches band_mass.py '{exp}'")
        return

    out = Path(args.output_dir or f"outputs/verify_{st_us.lower()}")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"test_predictions_{st_us}.csv", "w") as f:
        f.write("Document ID,Prediction\n")
        for doc_id, preds in test_preds.items():
            f.write(f"{doc_id},{''.join(map(str, preds))}\n")
    with open(out / f"verify_{st_us}.json", "w") as f:
        json.dump({"subtask": args.subtask, "base": "locked oof_decoder", "model": args.model,
                   "band": list(args.band), "all_tokens": args.all_tokens,
                   "window": args.window, "epochs": args.epochs,
                   "lr": args.lr, "mode": mode, "verifier_threshold": vthr,
                   "n_dev_candidates": len(dev_c), "n_test_candidates": len(test_c),
                   "lock_dev": lock_dev, "lock_test": lock_test,
                   "dev": dev_best, args.test_split: test_sc}, f, indent=2)
    pickle.dump({"cands": test_c, "probs": test_p}, open(out / f"band_probs_{st_us}.pkl", "wb"))
    print(f"[verify] -> {out}")
    print("[verify] gate: bootstrap_paired.py against the lock CSV before re-locking.")


if __name__ == "__main__":
    main()
