"""Pure helpers for aligning punctuated teacher docs to no-punctuation students."""

import string
import unicodedata
from difflib import SequenceMatcher
from typing import Sequence, List, Tuple


_ARABIC_PUNCT = "،؛؟ـ«»“”‘’…"
_PUNCT_CHARS = set(string.punctuation) | set(_ARABIC_PUNCT)


def is_punctuation_token(token: str) -> bool:
    """Return True when a dataset token carries punctuation only."""
    text = str(token).strip()
    if not text:
        return True
    return all(ch in _PUNCT_CHARS or unicodedata.category(ch).startswith("P") for ch in text)


def _normal_chars(token: str) -> str:
    return "".join(
        ch for ch in str(token).strip()
        if ch not in _PUNCT_CHARS
        and not unicodedata.category(ch).startswith("P")
        and not ch.isspace()
    )


def _spans(tokens: Sequence[str]) -> Tuple[str, List[Tuple[int, int, int]]]:
    text = ""
    spans = []
    for i, token in enumerate(tokens):
        norm = _normal_chars(str(token))
        if not norm:
            continue
        start = len(text)
        text += norm
        spans.append((i, start, len(text)))
    return text, spans


def align_teacher_to_student(
    teacher_tokens: Sequence[str],
    teacher_probs: Sequence[float],
    student_tokens: Sequence[str],
) -> List[float]:
    """Project teacher word probabilities onto a punctuation-stripped student doc.

    Teacher datasets include punctuation tokens for PA/NP. NoPnx datasets keep the same
    document text with punctuation removed, so the valid mapping is obtained by dropping
    punctuation-only teacher tokens and checking the remaining token sequence matches the
    student tokens exactly.
    """
    if len(teacher_tokens) != len(teacher_probs):
        raise ValueError(
            f"teacher token/prob length mismatch: {len(teacher_tokens)} tokens vs {len(teacher_probs)} probs"
        )

    teacher_text, teacher_spans = _spans(teacher_tokens)
    student_text, student_spans = _spans(student_tokens)
    matcher = SequenceMatcher(a=teacher_text, b=student_text, autojunk=False)
    student_to_teacher = {}
    for t0, s0, size in matcher.get_matching_blocks():
        for off in range(size):
            student_to_teacher[s0 + off] = t0 + off

    mapped = len(student_to_teacher)
    if student_text and mapped / len(student_text) < 0.80:
        raise ValueError(
            f"normalized text overlap too low: mapped {mapped}/{len(student_text)} student chars; "
            f"teacher_prefix={teacher_text[:40]!r}, student_prefix={student_text[:40]!r}"
        )

    aligned = []
    tpos = 0
    by_student_index = {}
    for si, s0, s1 in student_spans:
        mapped_positions = [student_to_teacher[p] for p in range(s0, s1) if p in student_to_teacher]
        if not mapped_positions:
            by_student_index[si] = 0.0
            continue
        ms0, ms1 = min(mapped_positions), max(mapped_positions) + 1
        overlaps = []
        while tpos < len(teacher_spans) and teacher_spans[tpos][2] <= ms0:
            tpos += 1
        j = tpos
        while j < len(teacher_spans) and teacher_spans[j][1] < ms1:
            ti, t0, t1 = teacher_spans[j]
            overlap = min(ms1, t1) - max(ms0, t0)
            if overlap > 0:
                overlaps.append((overlap, float(teacher_probs[ti])))
            j += 1
        if not overlaps:
            raise ValueError(f"no teacher probability overlaps student char span {s0}:{s1}")
        by_student_index[si] = max(overlaps, key=lambda x: x[0])[1]
    return [by_student_index.get(i, 0.0) for i in range(len(student_tokens))]
