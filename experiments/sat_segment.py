"""SaT / wtpsplit baseline + calibration for AraSeg 2026 (Track 1).

Segment Any Text (SaT, EMNLP 2024) is the purpose-built SOTA for this exact task:
an XLM-R model trained to emit per-character sentence-boundary probabilities across
85 languages (incl. Arabic), robust to missing/noisy punctuation — our weakest
(NoPnx) regime. This script runs it, aligns its char-level probs back to AraSeg's
per-token labels, scores document-level macro-F1 comparably to our own pipeline,
and saves per-doc prob caches that drop straight into ensemble.py.

CLOSED-TRACK LEGALITY:
  * frozen SaT (no --lora_path)                 -> legal (pretrained model only)
  * LoRA trained ONLY on AraSeg train, loaded   -> legal
  * wtpsplit's *pretrained* ud/opus/ersatz LoRA -> ILLEGAL in closed (trained on
    non-AraSeg data). Use it only for an open-track comparison, clearly labelled.
    Wired 2026-07-30 as --style_or_domain + --language (BOTH required by wtpsplit).
    Its caches are written with an OPENONLY_ prefix so they cannot be folded into a
    closed lock by accident. Open track has no data restrictions, and we have never
    submitted anything open-track-specific -- every open entry is a re-upload.
    NB: LoRA models want a HIGHER threshold than base SaT; tune_threshold handles it,
    but a fixed 0.5 would read as a null result.

Runs on a Kaggle T4. Dep: `pip install wtpsplit` (added to requirements.txt).

Examples:
  python sat_segment.py --selfcheck                         # alignment sanity
  python sat_segment.py --split dev --model sat-3l-sm        # D0 zero-shot, all subtasks
  python sat_segment.py --split dev --model sat-12l-sm --save-cache outputs/sat
  python sat_segment.py --split test --model sat-3l-sm --out outputs/sat   # submission CSVs
  python sat_segment.py --split dev --lora_path adapters/araseg_pa          # D1 (LoRA you trained)

D1 LoRA training is NOT done here — wtpsplit ships its own adapter trainer
(see https://github.com/segment-any-text/wtpsplit, `wtpsplit/train`). Train on
AraSeg train only, then point --lora_path at the saved adapter.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np

SUBTASK_DATASETS = {
    "PA":        "MBZUAI/AraSeg-2026-Shared-Task-PA",
    "NoPnx-PA":  "MBZUAI/AraSeg-2026-Shared-Task-NoPnx-PA",
    "NP":        "MBZUAI/AraSeg-2026-Shared-Task-NP",
    "NoPnx-NP":  "MBZUAI/AraSeg-2026-Shared-Task-NoPnx-NP",
}
_SPLIT_ALIASES = {"dev": ["dev", "validation", "development"], "test": ["test", "testing"],
                  "train": ["train", "training"], "blind": ["blind"]}
# Blind test repos (Testing Phase): split=='blind' routes here. load_docs already
# tolerates the missing label column. Token via HF_TOKEN env, never in code.
_BLIND_DATASETS = {k: v + "-Blind" for k, v in SUBTASK_DATASETS.items()}
_NATIVE_GENRE_COLS = ("genre", "domain", "source", "category", "style")


# ---------------------------------------------------------------- data loading
def _load_split(subtask, split):
    import os
    from datasets import load_dataset
    kw = {}
    if split == "blind":
        ds_name = _BLIND_DATASETS[subtask]
        # organizers' HF, different token than the model repos — pass it explicitly
        # (HF_TOKEN env stays the model token). Never committed.
        tok = os.environ.get("ARASEG_BLIND_TOKEN") or os.environ.get("HF_TOKEN")
        if tok:
            kw["token"] = tok
    else:
        ds_name = SUBTASK_DATASETS[subtask]
    for s in _SPLIT_ALIASES.get(split, [split]):
        try:
            return load_dataset(ds_name, split=s, **kw)
        except Exception:
            continue
    raise ValueError(f"Split '{split}' not found in {ds_name}")


def load_docs(subtask, split):
    """Return [(doc_id, tokens, gold_labels, genre_or_None)]. gold is zeros if no
    label column (hidden test)."""
    ds = _load_split(subtask, split)
    row0 = ds[0]
    tok_col = next(k for k in ("tokens", "words") if k in row0 and isinstance(row0[k], list))
    lab_col = next((k for k in ("labels", "label", "tags", "ner_tags")
                    if k in row0 and isinstance(row0[k], list)), None)
    id_col = next((k for k in ("doc_id", "id", "file_id") if k in row0), None)
    genre_col = next((k for k in _NATIVE_GENRE_COLS if k in row0), None)  # native only

    docs = []
    for i, r in enumerate(ds):
        toks = r[tok_col]
        gold = np.asarray([int(x) for x in r[lab_col]], dtype=np.int64) if lab_col \
            else np.zeros(len(toks), dtype=np.int64)
        did = str(r[id_col]) if id_col else str(i)
        genre = str(r[genre_col]) if genre_col else None
        docs.append((did, toks, gold, genre))
    return docs


# ----------------------------------------------------------- token<->char align
def _offsets(tokens):
    """Char span [start, end) of each token in ' '.join(tokens), and total length."""
    spans, pos = [], 0
    for t in tokens:
        spans.append((pos, pos + len(t)))
        pos += len(t) + 1               # +1 for the joining space
    text_len = pos - 1 if tokens else 0  # drop trailing space
    return spans, text_len


def _token_probs(char_probs, spans, text_len):
    """Per-token boundary prob = max prob over (token's last char, following space).
    SaT places the newline signal at the boundary char; covering both is robust."""
    out = np.zeros(len(spans), dtype=np.float32)
    n = len(char_probs)
    for i, (s, e) in enumerate(spans):
        idxs = [min(max(e - 1, s), n - 1)]      # last char (clamped)
        if e < text_len and e < n:
            idxs.append(e)                       # trailing space
        out[i] = max(float(char_probs[j]) for j in idxs)
    return out


def sat_probs_for_docs(sat, docs, batch_size=16):
    """{doc_id: (prob_per_word float32[n], gold int64[n])} — same shape metrics.py uses."""
    cache = {}
    for start in range(0, len(docs), batch_size):
        chunk = docs[start:start + batch_size]
        texts = [" ".join(toks) for _, toks, _, _ in chunk]
        for (did, toks, gold, _), cprobs in zip(chunk, sat.predict_proba(texts)):
            spans, tlen = _offsets(toks)
            cprobs = np.asarray(cprobs, dtype=np.float32)
            cache[did] = (_token_probs(cprobs, spans, tlen), gold)
    return cache


# --------------------------------------------------------------------- metrics
def _doc_f1(pred, gold):
    tp = int(np.sum((pred == 1) & (gold == 1)))
    fp = int(np.sum((pred == 1) & (gold == 0)))
    fn = int(np.sum((pred == 0) & (gold == 1)))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def macro_f1_at(cache, thr):
    return float(np.mean([_doc_f1((pa >= thr).astype(np.int64), g) for pa, g in cache.values()]))


def tune_threshold(cache, grid=None):
    grid = np.arange(0.05, 0.96, 0.05) if grid is None else grid
    scores = [(float(t), macro_f1_at(cache, float(t))) for t in grid]
    return max(scores, key=lambda x: x[1])  # (thr, f1)


# ------------------------------------------------- boundary-count calibration (D2)
def fit_count_model(train_cache, train_genres):
    """Fit doc_length -> n_boundaries. Per-genre linear if native genres with
    enough docs each, else a single global line. Returns predict(length, genre)."""
    lens = np.array([len(g) for _, g in train_cache.values()], dtype=float)
    cnts = np.array([int(g.sum()) for _, g in train_cache.values()], dtype=float)

    def line(x, y):
        if len(x) >= 3 and np.ptp(x) > 0:
            a, b = np.polyfit(x, y, 1)
            return lambda L: max(0.0, a * L + b)
        rate = (y.sum() / x.sum()) if x.sum() else 0.0   # boundaries per token
        return lambda L: rate * L

    global_fn = line(lens, cnts)
    per_genre = {}
    if train_genres and any(train_genres.values()):
        by_g = {}
        for (did, (_, g)) in train_cache.items():
            gen = train_genres.get(did)
            if gen is not None:
                by_g.setdefault(gen, []).append((len(g), int(g.sum())))
        for gen, rows in by_g.items():
            if len(rows) >= 5:
                x = np.array([r[0] for r in rows], float)
                y = np.array([r[1] for r in rows], float)
                per_genre[gen] = line(x, y)

    def predict(length, genre):
        fn = per_genre.get(genre, global_fn)
        return fn(length)
    return predict


def count_preds(probs, target):
    """Top-k decode: mark the `target` highest-prob tokens as boundaries."""
    n = len(probs)
    k = int(np.clip(round(target), 0, n))
    preds = np.zeros(n, dtype=np.int64)
    if k > 0:
        preds[np.argpartition(-probs, k - 1)[:k]] = 1
    return preds


def macro_f1_count(cache, predict, genres):
    fs = []
    for did, (pa, g) in cache.items():
        tgt = predict(len(pa), genres.get(did) if genres else None)
        fs.append(_doc_f1(count_preds(pa, tgt), g))
    return float(np.mean(fs))


# ------------------------------------------------------------------------- main
def build_sat(model, lora_path, style_or_domain=None, language=None):
    """Three mutually exclusive modes; see the legality block at the top of this file.

    frozen (closed-legal) | --lora_path, ours on AraSeg train (closed-legal)
    | --style_or_domain + --language, wtpsplit's OWN pretrained LoRA (OPEN TRACK ONLY).

    wtpsplit requires BOTH style_or_domain and language for the pretrained adapters;
    passing one alone silently falls back to the base model, which would look like a
    null result rather than an error. Available pairs live in <model_repo>/loras.
    """
    from wtpsplit import SaT
    import torch
    if style_or_domain or language:
        if not (style_or_domain and language):
            raise ValueError("wtpsplit needs BOTH --style_or_domain and --language; "
                             "one alone silently loads the base model")
        if lora_path:
            raise ValueError("--lora_path and --style_or_domain are mutually exclusive")
        sat = SaT(model, style_or_domain=style_or_domain, language=language)
    else:
        sat = SaT(model, lora_path=lora_path) if lora_path else SaT(model)
    if torch.cuda.is_available():
        sat.half().to("cuda")          # wtpsplit GPU pattern (~10x faster than CPU)
    return sat


def selfcheck(model="sat-3l-sm", style_or_domain=None, language=None):
    """Alignment sanity on one dev doc — this checks the char->token MAPPING, not model
    quality. The two hard asserts are the real check (per-character probs, and one prob
    per token). Top-k overlap F1 is only a smoke test that the mapping is not scrambled.

    That smoke test is graded against CHANCE, not a constant: picking k of n tokens at
    random scores about k/n, so a scrambled mapping lands there. A fixed bar instead
    encodes the quality of whichever model it was tuned on — 0.3 was tuned on
    'sat-3l-sm' and false-failed the weaker self-supervised 'sat-3l' base at 0.290
    against a 0.177 chance level, which is comfortably fine.

    Takes model/adapter args so an offline run can check whatever is actually cached,
    and so it can smoke-test the real configuration rather than a proxy for it.
    """
    sat = build_sat(model, None, style_or_domain, language)
    did, toks, gold, _ = load_docs("PA", "dev")[0]
    text = " ".join(toks)
    probs = np.asarray(sat.predict_proba(text), dtype=np.float32)
    assert len(probs) == len(text), (
        f"predict_proba is not per-character: len(probs)={len(probs)} "
        f"len(text)={len(text)} — char-offset mapping is invalid")
    tp = _token_probs(probs, *_offsets(toks))
    assert len(tp) == len(gold) == len(toks), f"length mismatch {len(tp)}/{len(gold)}/{len(toks)}"
    k = int(gold.sum())
    f1 = _doc_f1(count_preds(tp, k), gold)
    chance = k / len(toks)
    lo, hi, sd = float(tp.min()), float(tp.max()), float(tp.std())
    print(f"[selfcheck] doc={did} tokens={len(toks)} gold_boundaries={k} "
          f"top-k_F1={f1:.3f} (chance {chance:.3f})  probs[min={lo:.4f} max={hi:.4f} sd={sd:.4f}]")

    # The two asserts above are the alignment check and they are model-independent.
    # Top-k F1 only smoke-tests that the char->token mapping is not scrambled, so grade
    # it ONLY on the base model, where a low score can mean nothing else. With an adapter
    # loaded, a low score means the adapter is bad on this corpus — a RESULT, not a
    # broken pipeline, and not something to abort the run over.
    if style_or_domain:
        if sd < 1e-4:
            raise AssertionError(
                f"adapter {style_or_domain}/{language} returns near-constant probs "
                f"(sd={sd:.2e}) — that is a loading failure, not a weak model")
        if f1 < 1.5 * chance:
            print(f"[selfcheck] ⚠ adapter {style_or_domain}/{language} ranks at chance on "
                  f"this doc ({f1:.3f} vs {chance:.3f}); probs vary, so it loaded and is "
                  f"simply poor here. Continuing — the run reports the real numbers.")
    else:
        assert f1 >= 1.5 * chance, (
            f"alignment looks scrambled: top-k F1 {f1:.3f} vs chance {chance:.3f}")
    print("[selfcheck] OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subtask", choices=list(SUBTASK_DATASETS), help="default: all 4")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--model", default="sat-3l-sm")
    ap.add_argument("--lora_path", default=None, help="adapter trained on AraSeg train (D1)")
    ap.add_argument("--style_or_domain", default=None,
                    help="wtpsplit's OWN pretrained LoRA, e.g. ud / opus100 / ersatz. "
                         "OPEN TRACK ONLY -- trained on non-AraSeg data. Needs --language.")
    ap.add_argument("--language", default=None,
                    help="language code for --style_or_domain, e.g. ar")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--save-cache", dest="save_cache", default=None,
                    help="dir to write per-(subtask,split) prob pkl for ensemble.py")
    ap.add_argument("--out", default=None, help="dir for test submission CSVs (--split test)")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        selfcheck(args.model, args.style_or_domain, args.language)
        return

    sat = build_sat(args.model, args.lora_path, args.style_or_domain, args.language)
    subtasks = [args.subtask] if args.subtask else list(SUBTASK_DATASETS)
    if args.style_or_domain:
        # OPENONLY_ rides in the cache filename on purpose: these probabilities are
        # closed-track ILLEGAL, and the prefix makes it impossible to fold them into a
        # closed ensemble without noticing.
        tag = f"OPENONLY_{args.model}_{args.style_or_domain}_{args.language}"
        print("\n*** OPEN TRACK ONLY — wtpsplit's pretrained LoRA is closed-track ILLEGAL ***")
    elif args.lora_path:
        tag = args.lora_path.replace("/", "_")
    else:
        tag = args.model

    print(f"\nmodel={args.model} lora={args.style_or_domain or args.lora_path or 'none'} "
          f"split={args.split}")
    print(f"{'subtask':<10} {'thr=0.5':>9} {'global-tuned':>16} {'count-calib':>13}")
    for st in subtasks:
        docs = load_docs(st, args.split)
        cache = sat_probs_for_docs(sat, docs, args.batch_size)
        genres = {d[0]: d[3] for d in docs if d[3] is not None} or None

        f1_half = macro_f1_at(cache, 0.5)
        thr, f1_tuned = tune_threshold(cache)
        # count calibration is fit on TRAIN (gold counts), applied here
        train_docs = load_docs(st, "train")
        train_cache = sat_probs_for_docs(sat, train_docs, args.batch_size)
        train_genres = {d[0]: d[3] for d in train_docs if d[3] is not None} or None
        predict = fit_count_model(train_cache, train_genres)
        f1_count = macro_f1_count(cache, predict, genres)

        print(f"{st:<10} {f1_half:>9.4f} {f1_tuned:>10.4f}@{thr:<4.2f} {f1_count:>13.4f}")

        if args.save_cache:
            d = Path(args.save_cache); d.mkdir(parents=True, exist_ok=True)
            fn = d / f"{st.replace('-', '_')}_{args.split}_{tag}.pkl"
            with open(fn, "wb") as f:
                pickle.dump(cache, f)
            print(f"           cache -> {fn}")

        if args.split == "test" and args.out:
            o = Path(args.out); o.mkdir(parents=True, exist_ok=True)
            csv = o / f"test_predictions_{st.replace('-', '_')}.csv"
            with open(csv, "w", encoding="utf-8") as f:
                f.write("Document ID,Prediction\n")
                for did, (pa, _) in cache.items():
                    tgt = predict(len(pa), (genres or {}).get(did))
                    preds = count_preds(pa, tgt)
                    f.write(f"{did},{''.join(map(str, preds.tolist()))}\n")
            print(f"           submission -> {csv}")


if __name__ == "__main__":
    main()
