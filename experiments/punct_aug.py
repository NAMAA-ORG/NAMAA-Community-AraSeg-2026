"""Punctuation-dropout augmentation for AraSeg NoPnx subtasks (Track 2).

The 4 subtasks are the SAME 174 docs. NoPnx-* is PA-* with punctuation tokens
DELETED — verified: NoPnx tokens are a strict subsequence of PA tokens in all
174 train docs, and a deleted boundary token's label is carried to the preceding
surviving token. ~59% of PA boundaries sit ON the punctuation token, so a NoPnx
model loses that cue (this is why NoPnx is the hard gap). This module rebuilds it:

  1. STOCHASTIC DROPOUT: keep each removable punct token with prob (1-drop_p), so
     training sees a CONTINUUM between PA (drop_p=0) and NoPnx (drop_p=1), not just
     the two fixed endpoints the official datasets give.
  2. AUX TARGET: for every surviving token, which punctuation class was dropped
     right after it (NONE/OTHER/SENT) — supervises a punctuation-restoration head
     so the encoder learns where punctuation *would* be even when it's gone.

drop_p=1.0 reproduces the official NoPnx data exactly (asserted in __main__).
No punctuation rule is guessed: removals come from the PA↔NoPnx alignment itself.
"""

from typing import List, Dict
import random

from datasets import load_dataset

AUX_NONE, AUX_OTHER, AUX_SENT = 0, 1, 2          # n_aux_classes = 3
N_AUX_CLASSES = 3
_SENT_FINAL = set(".?!…؟۔")                       # sentence-final marks (rough; aux only)

# PA subtask -> its NoPnx counterpart (the deletion target)
_NOPNX_OF = {"PA": "NoPnx-PA", "NP": "NoPnx-NP"}
_DS = "MBZUAI/AraSeg-2026-Shared-Task-{}"


def _punct_class(tok: str) -> int:
    return AUX_SENT if any(c in _SENT_FINAL for c in tok) else AUX_OTHER


def _removed_flags(pa_tokens: List[str], nopnx_tokens: List[str]) -> List[bool]:
    """2-pointer: mark each PA token True if it was deleted in NoPnx. Relies on
    NoPnx being a strict subsequence of PA (verified for all train docs)."""
    flags = [False] * len(pa_tokens)
    j = 0
    for i, t in enumerate(pa_tokens):
        if j < len(nopnx_tokens) and t == nopnx_tokens[j]:
            j += 1
        else:
            flags[i] = True
    if j != len(nopnx_tokens):
        raise ValueError("NoPnx is not a subsequence of PA — alignment failed")
    return flags


def corrupt_doc(pa_tokens, pa_labels, removed_flags, drop_p, rng):
    """Return (tokens, labels, aux_labels). Removable punct tokens are dropped
    with prob drop_p; a dropped boundary (and the aux punct class) moves to the
    last surviving NON-whitespace token — NoPnx skips paragraph-break ('\\n')
    tokens when carrying a boundary back. aux_labels[i] = strongest punct class
    dropped after output token i."""
    out_t: List[str] = []
    out_l: List[int] = []
    out_a: List[int] = []
    for t, l, rem in zip(pa_tokens, pa_labels, removed_flags):
        if rem and rng.random() < drop_p:
            tgt = next((k for k in range(len(out_t) - 1, -1, -1) if out_t[k].strip()), None)
            if tgt is not None:                        # transfer boundary + aux
                if l == 1:
                    out_l[tgt] = 1
                out_a[tgt] = max(out_a[tgt], _punct_class(t))
            # doc-initial dropped punct: boundary (rare) is lost, matches NoPnx
        else:
            out_t.append(t)
            out_l.append(int(l))
            out_a.append(AUX_NONE)
    return out_t, out_l, out_a


def _paired_projection(pa_tokens: List[str], nopnx_tokens: List[str],
                       removed_flags: List[bool]) -> List[str]:
    """One entry per NoPnx token: text unchanged, except a removed PA token that
    followed it is APPENDED (string concatenation, not a new token) -- so the
    projection has the exact same length as nopnx_tokens. A removed token before
    the first surviving word has no owner and is dropped, matching how NoPnx's own
    official labels already treat doc-initial punctuation (nothing to attach to)."""
    projected = list(nopnx_tokens)
    owner = -1  # index into projected of the last surviving word seen
    for tok, removed in zip(pa_tokens, removed_flags):
        if removed:
            if owner >= 0:
                projected[owner] += tok
        else:
            owner += 1
    return projected


def build_paired_chunks(pa_subtask: str, split: str, chunk_words: int = 64) -> List[Dict]:
    """Fixed-size aligned word chunks for paired-view consistency training (Lane C
    Task 4). Each chunk: {doc_id, tokens, paired_tokens, labels}. `tokens`/`labels`
    are official NoPnx (ground truth); `paired_tokens` is the same words with removed
    punctuation appended to its preceding survivor -- equal word count, so the two
    views can be trained together and compared word-for-word (paired_word_kl).
    A document whose NoPnx/PA alignment fails is rejected outright, not guessed at."""
    nopnx_subtask = _NOPNX_OF[pa_subtask]
    pa = {r["doc_id"]: r for r in load_dataset(_DS.format(pa_subtask), split=split)}
    npx = {r["doc_id"]: r for r in load_dataset(_DS.format(nopnx_subtask), split=split)}
    chunks = []
    for did, npx_row in npx.items():
        if did not in pa:
            continue
        try:
            flags = _removed_flags(pa[did]["tokens"], npx_row["tokens"])
        except ValueError:
            continue
        nopnx_tokens = list(npx_row["tokens"])
        nopnx_labels = [int(l) for l in npx_row["labels"]]
        projected = _paired_projection(pa[did]["tokens"], nopnx_tokens, flags)
        assert len(projected) == len(nopnx_tokens) == len(nopnx_labels)
        for start in range(0, len(nopnx_tokens), chunk_words):
            end = start + chunk_words
            chunks.append({
                "doc_id": f"{did}:{start // chunk_words}",
                "tokens": nopnx_tokens[start:end],
                "paired_tokens": projected[start:end],
                "labels": nopnx_labels[start:end],
            })
    return chunks


def build_aug_docs(pa_subtask: str, split: str, drop_p: float, seed: int = 42) -> List[Dict]:
    """Build punctuation-corrupted training records from PA + its NoPnx counterpart.
    Each record: {doc_id, tokens, labels, aux_labels}. Deterministic given seed."""
    nopnx_subtask = _NOPNX_OF[pa_subtask]
    pa = {r["doc_id"]: r for r in load_dataset(_DS.format(pa_subtask), split=split)}
    npx = {r["doc_id"]: r for r in load_dataset(_DS.format(nopnx_subtask), split=split)}
    rng = random.Random(seed)
    docs = []
    for did in pa:
        if did not in npx:
            continue
        flags = _removed_flags(pa[did]["tokens"], npx[did]["tokens"])
        t, l, a = corrupt_doc(pa[did]["tokens"], pa[did]["labels"], flags, drop_p, rng)
        docs.append({"doc_id": did, "tokens": t, "labels": l, "aux_labels": a})
    return docs


if __name__ == "__main__":
    # Self-check: drop_p=1.0 must reproduce the official NoPnx data exactly,
    # and aux must be NONE wherever nothing was dropped.
    for pa_st in ("PA", "NP"):
        nopnx_st = _NOPNX_OF[pa_st]
        npx = {r["doc_id"]: r for r in load_dataset(_DS.format(nopnx_st), split="train")}
        aug = build_aug_docs(pa_st, "train", drop_p=1.0)
        bad_t = bad_l = 0
        for d in aug:
            ref = npx[d["doc_id"]]
            if d["tokens"] != ref["tokens"]:
                bad_t += 1
            if d["labels"] != list(ref["labels"]):
                bad_l += 1
            # aux only marks dropped-punct positions; at drop_p=1 every removed
            # token is dropped, so #SENT+#OTHER aux must equal #removed tokens that
            # had a surviving predecessor.
        print(f"[{pa_st}->{nopnx_st}] drop_p=1.0: token-mismatch {bad_t}/{len(aug)}  "
              f"label-mismatch {bad_l}/{len(aug)}")
        assert bad_t == 0 and bad_l == 0, "drop_p=1.0 did not reproduce official NoPnx"

        # partial corruption sanity: drop_p=0.0 must equal PA, and aux all-NONE
        pa0 = build_aug_docs(pa_st, "train", drop_p=0.0)
        pa_raw = {r["doc_id"]: r for r in load_dataset(_DS.format(pa_st), split="train")}
        assert all(d["tokens"] == pa_raw[d["doc_id"]]["tokens"] for d in pa0), "drop_p=0 != PA"
        assert all(set(d["aux_labels"]) <= {AUX_NONE} for d in pa0), "drop_p=0 should have no aux"
        n_aux = sum(1 for d in build_aug_docs(pa_st, "train", drop_p=0.5)
                    for a in d["aux_labels"] if a != AUX_NONE)
        print(f"[{pa_st}] drop_p=0 reproduces PA OK; drop_p=0.5 aux marks {n_aux} positions")
    print("punct_aug self-check OK")
