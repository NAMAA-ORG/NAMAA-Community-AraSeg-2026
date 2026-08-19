"""Dataset loading and tokenization for AraSeg 2026.

Each dataset row is a whole document. Documents routinely exceed the encoder's
max_length (see check_lengths.py: PA P90 ~1700 subwords, max 30k), so we split
each document into overlapping windows instead of truncating it. Every window is
a self-contained input (own CLS/SEP) and carries, per token, the *global* word
index it came from — so predictions can be stitched back to the full document at
evaluation time.

Padding is done dynamically per batch in `make_collate` (NOT to max_length),
which is what makes long-context models (e.g. AraModernBert, 8192 ctx) tractable.
"""

import os
from typing import Dict, List, Tuple, Optional, Callable
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

# Sentinel IDs for the syntax embedding tables (0=PAD, 1=UNK, 2+=real)
_SYN_PAD = 0
_SYN_UNK = 1

SUBTASK_DATASETS = {
    "PA":        "MBZUAI/AraSeg-2026-Shared-Task-PA",
    "NoPnx-PA":  "MBZUAI/AraSeg-2026-Shared-Task-NoPnx-PA",
    "NP":        "MBZUAI/AraSeg-2026-Shared-Task-NP",
    "NoPnx-NP":  "MBZUAI/AraSeg-2026-Shared-Task-NoPnx-NP",
}

# Blind test repos (Testing Phase, Jul 2026). Separate gated repos, split name
# 'blind', columns doc_id/tokens/text — NO labels. Selecting split=='blind'
# routes here. Access via HF_TOKEN env (load_dataset reads it automatically); the
# token is never stored in code (organizer requirement).
_BLIND_DATASETS = {k: v + "-Blind" for k, v in SUBTASK_DATASETS.items()}

# Common split name variants across HuggingFace datasets
_SPLIT_ALIASES = {
    "train":      ["train", "training"],
    "dev":        ["dev", "validation", "development"],
    "validation": ["validation", "dev", "development"],
    "test":       ["test", "testing"],
}


def _load_split(subtask: str, split: str):
    kw = {}
    if split == "blind":
        ds_name = _BLIND_DATASETS[subtask]
        candidates = ["blind"]
        # The blind repos live on the ORGANIZERS' HF, gated by a different token than
        # the user's model repos. HF_TOKEN env stays the model token; the blind token
        # is passed explicitly here (organizer rule: never commit it). Falls back to
        # HF_TOKEN if one token happens to cover both.
        tok = os.environ.get("ARASEG_BLIND_TOKEN") or os.environ.get("HF_TOKEN")
        if tok:
            kw["token"] = tok
    else:
        ds_name = SUBTASK_DATASETS[subtask]
        candidates = _SPLIT_ALIASES.get(split, [split])
    for s in candidates:
        try:
            return load_dataset(ds_name, split=s, **kw)
        except Exception:
            continue
    try:
        available = list(load_dataset(ds_name, **kw).keys())
    except Exception:
        available = ["(could not determine)"]
    raise ValueError(
        f"Split '{split}' not found in {ds_name}. "
        f"Available splits: {available}. "
        f"Update _SPLIT_ALIASES in data_utils.py to match."
    )


def _detect_columns(row: dict) -> Tuple[str, str, Optional[str]]:
    """Return (token_col, label_col, id_col) by inspecting column names."""
    token_col = next(
        (k for k in ["tokens", "words"] if k in row and isinstance(row[k], list)),
        None,
    )
    label_col = next(
        (k for k in ["labels", "label", "tags", "ner_tags"] if k in row and isinstance(row[k], list)),
        None,
    )
    id_col = next((k for k in ["doc_id", "id", "file_id"] if k in row), None)
    if token_col is None:
        raise ValueError(
            f"Cannot find token column. Available columns: {list(row.keys())}"
        )
    # label_col may be None on the blind split (labels hidden) — handled in _process.
    return token_col, label_col, id_col


class AraSegDataset(Dataset):
    """Token-classification dataset for one AraSeg subtask split.

    One example == one *window* of a document. Long documents produce several
    overlapping windows; short ones produce a single window. `self.doc_gold`
    holds the full per-word gold labels for every document, keyed by doc_id, so
    evaluation can score complete documents regardless of windowing.
    """

    def __init__(
        self,
        subtask: str,
        split: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
        window_stride: int = 128,
        syntax_cache: Optional[dict] = None,
        pos2id: Optional[dict] = None,
        dep2id: Optional[dict] = None,
        records: Optional[List[dict]] = None,
        keep_doc_ids: Optional[set] = None,
    ):
        if not tokenizer.is_fast:
            raise ValueError(
                "AraSegDataset needs a fast tokenizer (overflow + word_ids). "
                f"Got a slow tokenizer for this model."
            )
        self.subtask = subtask
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.window_stride = window_stride
        self.syntax_cache = syntax_cache
        self.pos2id = pos2id or {}
        self.dep2id = dep2id or {}

        # `records` lets a caller pass in-memory docs (e.g. punct_aug.py output,
        # which also carries per-word "aux_labels"); otherwise load from HF.
        if records is not None:
            raw = records
            self._token_col, self._label_col, self._id_col = "tokens", "labels", "doc_id"
        else:
            raw = _load_split(subtask, split)
            self._token_col, self._label_col, self._id_col = _detect_columns(raw[0])

        # OOF stacking: restrict to a fixed doc-id subset (train-minus-fold, or a
        # single held-out fold). Filtering here keeps windowing/gold identical to
        # a full run — the kept docs are byte-for-byte what they'd be otherwise.
        if keep_doc_ids is not None:
            keep = set(keep_doc_ids)
            raw = [r for i, r in enumerate(raw)
                   if (str(r[self._id_col]) if self._id_col else str(i)) in keep]

        self.examples: List[dict] = []
        self.doc_gold: Dict[str, List[int]] = {}   # doc_id -> full per-word gold
        for i, row in enumerate(raw):
            self._process(row, i)

    def _process(self, row: dict, idx: int) -> None:
        tokens = row[self._token_col]
        # Blind split has no label column: zeros are placeholders; predictions
        # are what get written. Downstream keys on has_labels = gold.sum() > 0.
        labels = ([int(l) for l in row[self._label_col]] if self._label_col
                  else [0] * len(tokens))
        doc_id = str(row[self._id_col]) if self._id_col else str(idx)
        self.doc_gold[doc_id] = labels
        word_feats = (self.syntax_cache or {}).get(doc_id)
        aux = row["aux_labels"] if "aux_labels" in row else None  # Track 2 punct-restore target

        # Split into overlapping windows. `return_overflowing_tokens` yields one
        # sequence per window, each with its own special tokens; word_ids() per
        # window references the ORIGINAL (global) word positions.
        enc = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            stride=self.window_stride,
            return_overflowing_tokens=True,
            padding=False,
        )

        n_windows = len(enc["input_ids"])
        for w in range(n_windows):
            input_ids = enc["input_ids"][w]
            attn = enc["attention_mask"][w]
            word_ids = enc.word_ids(batch_index=w)

            aligned: List[int] = []     # label per token: first subword -> label, else -100
            gword: List[int] = []       # global word index per token, -1 for special/continuation
            syn_pos: List[int] = []     # POS id per token (0 = PAD/not-first-subword)
            syn_dep: List[int] = []     # DEP id per token
            aux_aligned: List[int] = [] # aux punct class per token (first subword), else -100
            prev = None
            for wid in word_ids:
                if wid is None:
                    aligned.append(-100)
                    gword.append(-1)
                    syn_pos.append(_SYN_PAD)
                    syn_dep.append(_SYN_PAD)
                    aux_aligned.append(-100)
                elif wid != prev:
                    aligned.append(labels[wid] if wid < len(labels) else -100)
                    gword.append(wid)
                    if word_feats is not None and wid < len(word_feats):
                        f = word_feats[wid]
                        syn_pos.append(self.pos2id.get(f["pos"], _SYN_UNK))
                        syn_dep.append(self.dep2id.get(f["dep"], _SYN_UNK))
                    else:
                        syn_pos.append(_SYN_PAD)
                        syn_dep.append(_SYN_PAD)
                    aux_aligned.append(aux[wid] if (aux is not None and wid < len(aux)) else -100)
                else:
                    aligned.append(-100)
                    gword.append(-1)
                    syn_pos.append(_SYN_PAD)
                    syn_dep.append(_SYN_PAD)
                    aux_aligned.append(-100)
                prev = wid

            ttype = enc.get("token_type_ids")
            ttype = ttype[w] if ttype is not None else [0] * len(input_ids)

            example = {
                "doc_id": doc_id,
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attn, dtype=torch.long),
                "token_type_ids": torch.tensor(ttype, dtype=torch.long),
                "labels": torch.tensor(aligned, dtype=torch.long),
                "word_idx": torch.tensor(gword, dtype=torch.long),
                "subtask": self.subtask,
            }
            if word_feats is not None:
                example["syntax_pos_ids"] = torch.tensor(syn_pos, dtype=torch.long)
                example["syntax_dep_ids"] = torch.tensor(syn_dep, dtype=torch.long)
            if aux is not None:
                example["aux_labels"] = torch.tensor(aux_aligned, dtype=torch.long)

            paired_tokens = row["paired_tokens"] if "paired_tokens" in row else None
            if paired_tokens is not None:
                if len(paired_tokens) != len(tokens):
                    raise ValueError(f"{doc_id}: paired_tokens word count "
                                     f"{len(paired_tokens)} != tokens {len(tokens)}")
                if n_windows != 1:
                    raise ValueError(f"{doc_id}: paired view needs exactly one window "
                                     f"for the primary view too, got {n_windows} "
                                     f"(chunk too large for max_length={self.max_length}?)")
                penc = self.tokenizer(
                    paired_tokens, is_split_into_words=True, max_length=self.max_length,
                    truncation=True, stride=self.window_stride,
                    return_overflowing_tokens=True, padding=False,
                )
                if len(penc["input_ids"]) != 1:
                    raise ValueError(f"{doc_id}: paired view needs exactly one window, "
                                     f"got {len(penc['input_ids'])} "
                                     f"(chunk too large for max_length={self.max_length}?)")
                p_word_ids = penc.word_ids(batch_index=0)
                p_gword, p_prev = [], None
                for wid in p_word_ids:
                    p_gword.append(wid if (wid is not None and wid != p_prev) else -1)
                    p_prev = wid
                p_ttype = penc.get("token_type_ids")
                p_ttype = p_ttype[0] if p_ttype is not None else [0] * len(penc["input_ids"][0])
                example["paired_input_ids"] = torch.tensor(penc["input_ids"][0], dtype=torch.long)
                example["paired_attention_mask"] = torch.tensor(penc["attention_mask"][0], dtype=torch.long)
                example["paired_token_type_ids"] = torch.tensor(p_ttype, dtype=torch.long)
                example["paired_word_idx"] = torch.tensor(p_gword, dtype=torch.long)

            self.examples.append(example)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def make_collate(pad_token_id: int) -> Callable[[List[dict]], dict]:
    """Build a collate_fn that pads dynamically to the longest sequence in the
    batch (not to max_length). Returns the closure so the pad id is captured."""

    def _pad(seqs: List[torch.Tensor], value: int, length: int) -> torch.Tensor:
        out = seqs[0].new_full((len(seqs), length), value)
        for i, s in enumerate(seqs):
            out[i, : s.size(0)] = s
        return out

    def collate_fn(batch: List[dict]) -> dict:
        maxlen = max(b["input_ids"].size(0) for b in batch)
        out = {
            "input_ids":      _pad([b["input_ids"]      for b in batch], pad_token_id, maxlen),
            "attention_mask": _pad([b["attention_mask"] for b in batch], 0,            maxlen),
            "token_type_ids": _pad([b["token_type_ids"] for b in batch], 0,            maxlen),
            "labels":         _pad([b["labels"]         for b in batch], -100,         maxlen),
            "word_idx":       _pad([b["word_idx"]       for b in batch], -1,           maxlen),
            "doc_ids":        [b["doc_id"]  for b in batch],
            "subtask":        batch[0]["subtask"],
        }
        if "syntax_pos_ids" in batch[0]:
            out["syntax_pos_ids"] = _pad([b["syntax_pos_ids"] for b in batch], 0, maxlen)
            out["syntax_dep_ids"] = _pad([b["syntax_dep_ids"] for b in batch], 0, maxlen)
        if "aux_labels" in batch[0]:
            out["aux_labels"] = _pad([b["aux_labels"] for b in batch], -100, maxlen)
        if "paired_input_ids" in batch[0]:
            p_maxlen = max(b["paired_input_ids"].size(0) for b in batch)
            out["paired_input_ids"] = _pad([b["paired_input_ids"] for b in batch],
                                           pad_token_id, p_maxlen)
            out["paired_attention_mask"] = _pad([b["paired_attention_mask"] for b in batch],
                                                0, p_maxlen)
            out["paired_token_type_ids"] = _pad([b["paired_token_type_ids"] for b in batch],
                                                0, p_maxlen)
            out["paired_word_idx"] = _pad([b["paired_word_idx"] for b in batch], -1, p_maxlen)
        return out

    return collate_fn
