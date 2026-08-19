"""Validate a frozen submission CSV before it gets packaged.

Checks the four ways a prediction file silently breaks a Codabench submission: wrong
header, a document ID set that doesn't match the split, a non-binary prediction
string, a per-document length that doesn't match the split's token count, or a
document whose last character isn't a boundary (every AraSeg doc ends with a gold
boundary -- see docs/handoff.md's final-boundary invariant).

`validate()` is pure stdlib `csv` + plain dicts, so it is unit-testable offline. The
CLI wrapper is the only piece that touches `data_utils._load_split` (network/HF).

Usage:
    python validate_submission.py --subtask PA --split blind \
        --prediction outputs/blind_candidate_pa/test_predictions_PA.csv
"""
import argparse
import csv
import sys

EXPECTED_HEADER = ["Document ID", "Prediction"]


def validate(prediction_path: str, expected_lengths: dict) -> list:
    """Return a list of error strings; empty means the file is valid."""
    errors = []
    with open(prediction_path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    if not rows or rows[0] != EXPECTED_HEADER:
        errors.append(f"header {rows[0] if rows else '(empty file)'} != {EXPECTED_HEADER}")
        return errors  # nothing below is trustworthy without the right columns

    seen = {}
    for i, row in enumerate(rows[1:], start=2):
        if len(row) != 2:
            errors.append(f"line {i}: expected 2 columns, got {len(row)}")
            continue
        doc_id, pred = row
        if doc_id in seen:
            errors.append(f"duplicate document ID: {doc_id}")
            continue
        seen[doc_id] = pred

    expected_ids = set(expected_lengths)
    got_ids = set(seen)
    missing = expected_ids - got_ids
    extra = got_ids - expected_ids
    if missing:
        errors.append(f"missing {len(missing)} document ID(s): {sorted(missing)[:5]}...")
    if extra:
        errors.append(f"{len(extra)} unexpected document ID(s): {sorted(extra)[:5]}...")

    for doc_id in sorted(expected_ids & got_ids):
        pred = seen[doc_id]
        if not pred or any(c not in "01" for c in pred):
            errors.append(f"{doc_id}: prediction is not a binary string: {pred!r}")
            continue
        n_expected = expected_lengths[doc_id]
        if len(pred) != n_expected:
            errors.append(f"{doc_id}: length {len(pred)} != expected {n_expected}")
            continue
        if pred[-1] != "1":
            errors.append(f"{doc_id}: final character is not a boundary (got {pred[-1]!r})")

    return errors


def _expected_lengths(subtask: str, split: str) -> dict:
    from data_utils import _load_split, _detect_columns
    raw = _load_split(subtask, split)
    tcol, _, icol = _detect_columns(raw[0])
    return {
        (str(r[icol]) if icol else str(i)): len(r[tcol])
        for i, r in enumerate(raw)
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subtask", required=True, choices=["PA", "NoPnx-PA", "NP", "NoPnx-NP"])
    ap.add_argument("--split", default="blind")
    ap.add_argument("--prediction", required=True)
    args = ap.parse_args()

    expected = _expected_lengths(args.subtask, args.split)
    errors = validate(args.prediction, expected)
    if errors:
        print(f"[validate] {args.prediction}: {len(errors)} error(s)")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print(f"[validate] {args.prediction}: OK ({len(expected)} documents)")


if __name__ == "__main__":
    main()
