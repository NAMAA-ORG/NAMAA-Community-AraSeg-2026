"""Restore punctuation, then segment with the PUNCTUATED lock. (Kareem's suggestion.)

The NoPnx subtasks are the punctuated ones with punctuation tokens deleted, and the
restore-then-segment oracle says the entire NoPnx deficit IS that missing cue: with
gold punctuation put back, our punctuated PA lock scores 0.9442 on NoPnx-PA against
the NoPnx-PA lock's 0.8718 (+7.2). This script replaces "gold punctuation" with
"Naqta-predicted punctuation" and measures what survives.

Why this can beat feeding Naqta into the decoder as a feature (which was noise, +0.15):
a per-word P(terminal) column tells the decoder WHERE punctuation went. Inserting the
token instead changes what the segmenter READS -- the punctuation reshapes the
contextual representation, which is where the +7 actually lives.

    NoPnx tokens ──Naqta──> restored tokens ──PA lock──> boundaries ──map back──> NoPnx

The map back is the inverse of punct_aug's verified alignment: a boundary predicted on
an inserted punctuation token carries to the PRECEDING surviving token, which is exactly
how the official NoPnx labels were derived from the punctuated ones.

ALWAYS RUN --mode gold FIRST. It feeds real PA tokens through the identical insert/map
plumbing, so it must reproduce the oracle's ~0.944. If it doesn't, the alignment is
wrong and any Naqta number is meaningless. That check is the whole reason to trust this.

Usage:
    python restore_then_segment.py --mode gold  --split test     # plumbing check
    python restore_then_segment.py --mode naqta --split test     # the real question
"""
import argparse
from pathlib import Path

import numpy as np

# Heavy/GPU imports are deferred into main() so the alignment helpers below stay
# importable (and unit-testable) on a box with no torch.

_EPS = 1e-6
# NoPnx-<X> is <X> with punctuation deleted; <X> is the head whose lock we borrow.
PUNCTUATED_OF = {"NoPnx-PA": "PA", "NoPnx-NP": "NP"}
# The PA lock: flat logit average of these three @0.25 (docs/handoff.md).
PA_LOCK = (["configs/e25_xlmr_single_pa_stride256.yaml",
            "configs/e32_qwen35_9b_pa.yaml",
            "configs/e33_gemma4_12b_pa.yaml"], 0.25)
from naqta_restore import map_binary_back as map_back

# ⚠️ Class 2 is Naqta's ARABIC comma U+060C, but AraSeg uses the ASCII comma U+002C for
# ALL 16722 of its commas -- U+060C appears ZERO times in the corpus (verified over
# PA train+dev+test, 2026-07-24). Inserting U+060C therefore feeds the frozen PA lock a
# character it never saw in-domain, on the most frequent restorable mark there is. Emit
# the corpus's own form here. This is inference-only (the frozen lock is never
# retrained) -- do NOT hoist this into naqta_restore.NAQTA_MARKS, which is the e71/e72
# TRAINING constant and must stay Naqta-native. (? and ; go the other way: the corpus
# prefers the Arabic forms, 1603 vs 276 and 708 vs 208, so those keep Naqta's.)
NAQTA_MARKS = {1: ".", 2: ",", 3: "؟", 4: "!", 5: ":", 6: "؛", 7: "-"}


def _docs(subtask, split):
    """{doc_id: (tokens, labels)} for one split, labels [] when hidden (blind)."""
    from data_utils import _load_split, _detect_columns
    raw = _load_split(subtask, split)
    tcol, lcol, icol = _detect_columns(raw[0])
    out = {}
    for i, r in enumerate(raw):
        did = str(r[icol]) if icol else str(i)
        out[did] = (list(r[tcol]), list(r[lcol]) if lcol and r.get(lcol) else [])
    return out


def restore_gold(nopnx_tokens, pa_tokens):
    """Perfect restoration: the real punctuated tokens. `owner[k]` = index of the NoPnx
    token that restored position k maps back to (an inserted mark maps to the token
    before it), or -1 for a mark that precedes any surviving token."""
    from punct_aug import _removed_flags
    removed = _removed_flags(pa_tokens, nopnx_tokens)
    owner, j = [], -1
    for i, tok in enumerate(pa_tokens):
        if not removed[i]:
            j += 1
        owner.append(j)          # a removed token's boundary carries to the preceding survivor
    assert j == len(nopnx_tokens) - 1, "alignment consumed the wrong number of tokens"
    return list(pa_tokens), owner


def restore_naqta(nopnx_tokens, cls_probs, min_p):
    """Insert Naqta's predicted mark after each token when it beats `min_p`.
    cls_probs: (n_tokens, 8). Class 0 is "no punctuation here"."""
    toks, owner = [], []
    for j, tok in enumerate(nopnx_tokens):
        toks.append(tok)
        owner.append(j)
        p = cls_probs[j]
        k = int(np.argmax(p[1:])) + 1        # best non-O mark
        if float(p[k]) >= min_p:
            toks.append(NAQTA_MARKS[k])
            owner.append(j)                  # carries back to the token it follows
    return toks, owner


def main():
    import torch
    from ensemble import _member_probs
    from scoring import _doc_f1, _macro
    from train import Config

    ap = argparse.ArgumentParser()
    ap.add_argument("--subtask", default="NoPnx-PA", choices=sorted(PUNCTUATED_OF))
    ap.add_argument("--mode", required=True, choices=["gold", "naqta"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--threshold", type=float, default=None,
                    help="segmentation threshold; default = the borrowed lock's")
    ap.add_argument("--min-punct-p", type=float, default=0.5,
                    help="naqta mode: insert a mark only above this probability")
    ap.add_argument("--naqta-cache", default=None,
                    help="default outputs/prob_cache/<Sub>_<split>_naqta_full.pkl")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--trained-member", default=None,
                    help="score a model TRAINED on restored NoPnx text (Kareem's recipe, "
                         "e.g. configs/e71_...yaml) instead of borrowing the frozen PA lock. "
                         "Runs on the NoPnx subtask directly; use the same --min-punct-p it "
                         "was trained with, and sweep --threshold on --split dev first.")
    args = ap.parse_args()

    # ── Kareem's recipe: a model trained on Naqta-restored NoPnx text ──────────
    # Same restoration builder train.py used, same map-back, so this number is
    # directly comparable to the frozen-lock naqta number above.
    if args.trained_member:
        from naqta_restore import build_restored_docs
        from ensemble import _member_probs
        from scoring import _doc_f1, _macro
        from train import Config
        nopnx = _docs(args.subtask, args.split)
        cfg = Config.from_yaml(args.trained_member)
        recs, owners = build_restored_docs(args.subtask, args.split, args.min_punct_p,
                                            comma=cfg.naqta_comma)
        pr = _member_probs(cfg, args.subtask, args.split, args.device, records=recs)

        def score_at(thr):
            s = [_doc_f1(map_back((pr[d][0] >= thr).astype(int), owners[d], len(nt)),
                         np.asarray(g, np.int64))
                 for d, (nt, g) in nopnx.items() if g]
            return _macro(s) if s else None

        thr_file = Path(cfg.output_dir) / "restore_thr.txt"
        if args.threshold is not None:
            thr = args.threshold                       # explicit wins
        elif args.split == "dev":
            # tune on dev exactly once, persist for the test run -> closed-legal
            grid = [(t, score_at(round(t, 2))["f1"]) for t in np.arange(0.15, 0.71, 0.05)]
            for t, f1 in grid:
                print(f"  dev thr {t:.2f}  F1 {f1:.4f}")
            thr = round(max(grid, key=lambda x: x[1])[0], 2)
            thr_file.parent.mkdir(parents=True, exist_ok=True)
            thr_file.write_text(str(thr))
            print(f"[rts] best dev thr = {thr} -> {thr_file}")
        elif thr_file.exists():
            thr = float(thr_file.read_text().strip())   # reuse the dev-tuned one
        else:
            raise SystemExit("no --threshold and no dev-tuned restore_thr.txt; run --split dev first")

        m = score_at(thr)

        # Dump the MAPPED-BACK predictions in submission format. Without this the
        # run prints an F1 and nothing else, so there is no way to paired-bootstrap
        # e71/e72 against their controls -- and a single-run delta under ~1.5 F1 is
        # not signal on these heads (ledger §5). Same columns as train.py's writer,
        # so bootstrap_paired.py / official_eval.py read it unchanged.
        out_csv = Path(cfg.output_dir) / (
            f"restored_{args.split}_predictions_{args.subtask.replace('-', '_')}.csv")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", encoding="utf-8") as fh:
            fh.write("Document ID,Prediction\n")
            for did, (ntok, _) in nopnx.items():
                bits = map_back((pr[did][0] >= thr).astype(int), owners[did], len(ntok))
                fh.write(f"{did},{''.join(str(int(b)) for b in bits)}\n")
        print(f"[rts] mapped-back predictions -> {out_csv}")

        # Implementation note: the control is the SINGLE model with restoration off, not the lock.
        # The lock is a multi-member decoder ensemble; quoting it next to a single
        # model's F1 invites a category error that already misread e71 as a failure.
        # Ledger §2.0, {subtask: (control, {split: its F1 on that split})}. Must be
        # compared split-for-split: quoting the control's DEV F1 next to a TEST score
        # understates the delta by ~0.4 F1 -- it did exactly that for e71, turning a
        # real +1.06 into a reported +0.64.
        CONTROL = {"NoPnx-PA": ("e17", {"dev": 0.8189, "test": 0.8147}),
                   "NoPnx-NP": ("e19", {"dev": 0.8231, "test": 0.8211})}
        cname, cf1 = CONTROL[args.subtask]
        cf1 = cf1.get(args.split)
        print(f"[rts] {cfg.experiment_id} on restored {args.subtask} {args.split} "
              f"@min_p={args.min_punct_p} thr={thr}: ", end="")
        if m is None:
            print("labels hidden — nothing to score")
        elif cf1 is None:
            print(f"P {m['precision']:.4f} R {m['recall']:.4f} F1 {m['f1']:.4f}  "
                  f"| no {cname} control recorded for split '{args.split}'")
        else:
            print(f"P {m['precision']:.4f} R {m['recall']:.4f} F1 {m['f1']:.4f}  "
                  f"| control {cname} {args.split} {cf1:.4f} -> {m['f1'] - cf1:+.4f}")
        return

    punctuated = PUNCTUATED_OF[args.subtask]
    if punctuated != "PA":
        raise SystemExit(f"only the PA lock is wired here; {punctuated} uses an OOF "
                         f"stack whose weights would also have to be borrowed")
    members, lock_thr = PA_LOCK
    thr = args.threshold if args.threshold is not None else lock_thr

    nopnx = _docs(args.subtask, args.split)
    print(f"[rts] {args.subtask} {args.split}: {len(nopnx)} docs  mode={args.mode}  thr={thr}")

    # ---- build the restored token stream -------------------------------------
    records, owners = [], {}
    if args.mode == "gold":
        pa = _docs(punctuated, args.split)
        missing = set(nopnx) - set(pa)
        assert not missing, f"{len(missing)} docs have no punctuated counterpart"
        for did, (ntok, _) in nopnx.items():
            toks, owner = restore_gold(ntok, pa[did][0])
            records.append({"doc_id": did, "tokens": toks, "labels": [0] * len(toks)})
            owners[did] = owner
    else:
        import pickle
        fp = Path(args.naqta_cache or f"outputs/prob_cache/"
                  f"{args.subtask.replace('-', '_')}_{args.split}_naqta_full.pkl")
        cache = pickle.load(open(fp, "rb"))
        ins = 0
        for did, (ntok, _) in nopnx.items():
            probs = np.asarray(cache[did][0], np.float32)   # (probs_8class, gold) tuple
            assert len(probs) == len(ntok), f"{did}: naqta {len(probs)} vs {len(ntok)} tokens"
            toks, owner = restore_naqta(ntok, probs, args.min_punct_p)
            ins += len(toks) - len(ntok)
            records.append({"doc_id": did, "tokens": toks, "labels": [0] * len(toks)})
            owners[did] = owner
        print(f"[rts] inserted {ins} marks (p>={args.min_punct_p}), "
              f"{ins / sum(len(t) for t, _ in nopnx.values()):.1%} of tokens")

    # ---- run the punctuated lock over the restored text ----------------------
    acc = None
    for cfgp in members:
        cfg = Config.from_yaml(cfgp)
        pr = _member_probs(cfg, punctuated, args.split, args.device, records=records)
        lo = {d: np.log(np.clip(p, _EPS, 1 - _EPS) / (1 - np.clip(p, _EPS, 1 - _EPS)))
              for d, (p, _) in pr.items()}
        acc = lo if acc is None else {d: acc[d] + lo[d] for d in acc}
        print(f"  [ok] {cfg.experiment_id}")
    probs = {d: 1 / (1 + np.exp(-v / len(members))) for d, v in acc.items()}

    # ---- map back to NoPnx space and score ----------------------------------
    scored = []
    for did, (ntok, gold) in nopnx.items():
        pred = map_back((probs[did] >= thr).astype(int), owners[did], len(ntok))
        if gold:
            scored.append(_doc_f1(pred, np.asarray(gold, np.int64)))
    if not scored:
        print("[rts] labels hidden — nothing to score")
        return
    m = _macro(scored)
    print(f"\n[rts] {args.mode}: P {m['precision']:.4f} R {m['recall']:.4f} "
          f"F1 {m['f1']:.4f}   (NoPnx-PA lock = 0.8718)")
    if args.mode == "gold":
        print("  gold mode must land near 0.944 — the oracle's perfect-restoration "
              "number. Far off => the insert/map alignment is wrong, not the idea.")


if __name__ == "__main__":
    main()
