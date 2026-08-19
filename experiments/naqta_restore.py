"""Naqta-restored training records: Kareem's recipe as a train-time input transform.

The inference-only version (restore_then_segment.py --mode naqta) borrows a FROZEN
punctuated lock and scores 0.82 on NoPnx-PA -- below the 0.8718 native lock -- because
the lock was trained on FULL punctuation and Naqta restores only ~half of it (its 8
classes cover . ، ؟ ! : ؛ - but not the parens/quotes/list-markers that carry the other
~47% of punctuation boundaries). The frozen model has never seen partial punctuation.

Kareem's version TRAINS on the restored text, so the model adapts to exactly that
partial, noisy restoration. This module builds those records; train.py consumes them
via `records=` (the same seam punct_aug uses).

Labeling (scheme A): the boundary stays on the WORD, as in the official NoPnx data.
Inserted marks get label 0 -- they are input context the encoder reads, not targets.
So the model learns "predict a boundary on the word that Naqta thinks a sentence-final
mark follows", which is where the +7 punctuation headroom actually lives.

Restore train AND dev AND test with the SAME min_p: a model trained to expect
punctuation in its input must see it at inference too, or it faces the mirror of the
frozen-lock mismatch.

    python naqta_restore.py --selfcheck        # offline: alignment + label invariants
"""
import argparse
import pickle
from pathlib import Path

import numpy as np


def map_binary_back(pred_restored, owner, n_original):
    """OR restored predictions into their original-token owners."""
    out = np.zeros(n_original, np.int64)
    for y, original in zip(pred_restored, owner):
        if y and original >= 0:
            out[original] = 1
    return out


def map_probs_back(prob_restored, owner, n_original):
    """Max-reduce restored probabilities into their original-token owners."""
    out = np.zeros(n_original, np.asarray(prob_restored).dtype)
    for probability, original in zip(prob_restored, owner):
        if original >= 0:
            out[original] = max(out[original], probability)
    return out

# Naqta id2label: 0=O 1=. 2=، 3=؟ 4=! 5=: 6=؛ 7=-
# Native Naqta marks -- e71/e72 were TRAINED against these via build_restored_docs
# below. Never flip this to the ASCII comma: that's a training-input constant, and an
# ASCII-comma variant is a distinct config (Lane C's naqta_comma knob), not a global
# edit. The inference-only frozen-lock borrow in restore_then_segment.py keeps its own
# local ASCII constant instead of importing this one.
NAQTA_MARKS = {1: ".", 2: "،", 3: "؟", 4: "!", 5: ":", 6: "؛", 7: "-"}


def _cache_path(nopnx_subtask, split):
    return Path("outputs/prob_cache") / f"{nopnx_subtask.replace('-', '_')}_{split}_naqta_full.pkl"


def restore_doc(tokens, labels, cls_probs, min_p, comma=None):
    """Interleave Naqta marks into one doc. Returns (toks, labs, owner):
      toks[k]   restored token stream
      labs[k]   scheme-A label (original word keeps its NoPnx label; inserted mark = 0)
      owner[k]  index of the NoPnx token position k maps back to (mark -> preceding word)

    `comma` overrides NAQTA_MARKS[2] for this call only -- e.g. Lane C's naqta_comma
    config knob (ASCII "," instead of the Naqta-native "،") -- without touching the
    shared NAQTA_MARKS constant e71/e72 were trained against. Default (None) is the
    Naqta-native mark, so the default path reproduces e71/e72 byte-for-byte.
    """
    marks = NAQTA_MARKS if comma is None else {**NAQTA_MARKS, 2: comma}
    toks, labs, owner = [], [], []
    for j, tok in enumerate(tokens):
        toks.append(tok)
        labs.append(int(labels[j]) if labels else 0)
        owner.append(j)
        k = int(np.argmax(cls_probs[j][1:])) + 1        # best non-O mark for this word
        if float(cls_probs[j][k]) >= min_p:
            toks.append(marks[k])
            labs.append(0)                               # inserted mark is never a target
            owner.append(j)
    return toks, labs, owner


def build_restored_docs(nopnx_subtask, split, min_p, cache_path=None, comma=None):
    """[{doc_id, tokens, labels}] for train.py, plus {doc_id: owner} for eval map-back.

    `comma`: see restore_doc. Default reproduces e71/e72 byte-for-byte.
    """
    from data_utils import _load_split, _detect_columns
    raw = _load_split(nopnx_subtask, split)
    tcol, lcol, icol = _detect_columns(raw[0])
    cache = pickle.load(open(cache_path or _cache_path(nopnx_subtask, split), "rb"))
    records, owners = [], {}
    for i, r in enumerate(raw):
        did = str(r[icol]) if icol else str(i)
        toks_in = list(r[tcol])
        labs_in = list(r[lcol]) if lcol and r.get(lcol) else []
        probs = np.asarray(cache[did][0], np.float32)    # (probs_8class, gold)
        assert len(probs) == len(toks_in), f"{did}: naqta {len(probs)} vs {len(toks_in)} tokens"
        t, l, o = restore_doc(toks_in, labs_in, probs, min_p, comma=comma)
        records.append({"doc_id": did, "tokens": t, "labels": l})
        owners[did] = o
    return records, owners


def _selfcheck():
    """No models, no data: the two invariants that keep training honest."""
    toks = ["w0", "w1", "w2"]
    labs = [0, 1, 0]
    p = np.zeros((3, 8), np.float32); p[:, 0] = 1.0
    p[0, 0], p[0, 1] = 0.1, 0.9        # "." after w0
    p[1, 0], p[1, 4] = 0.4, 0.6        # "!" after w1

    t, l, o = restore_doc(toks, labs, p, min_p=0.5)
    assert t == ["w0", ".", "w1", "!", "w2"], t
    # scheme A: word labels preserved in place, inserted marks are 0
    assert l == [0, 0, 1, 0, 0], l
    assert o == [0, 0, 1, 1, 2], o
    # every original label survives exactly once, in word order (no boundary lost/added)
    assert [l[k] for k in range(len(t)) if o[k] == k or t[k] == toks[o[k]]][:0] == []  # noop guard
    word_labels = [l[k] for k, oo in enumerate(o) if t[k] == toks[oo] and
                   (k == 0 or o[k] != o[k - 1])]
    assert word_labels == labs, (word_labels, labs)

    # min_p high enough -> no insertions -> records are the plain NoPnx data
    t2, l2, o2 = restore_doc(toks, labs, p, min_p=0.99)
    assert t2 == toks and l2 == labs and o2 == [0, 1, 2], (t2, l2, o2)
    print("naqta_restore selfcheck OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        _selfcheck()
    else:
        ap.error("nothing to do; use --selfcheck (record building is driven by train.py)")
