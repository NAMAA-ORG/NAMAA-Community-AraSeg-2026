"""Extract per-word POS/dep features from AraSeg docs using a frozen parser.

Parser priority: Stanza (Arabic UD, POS+dep) → CAMeL Tools (POS only) → zero features.
Cached to disk so training reads features, never the parser.

By default, the Stanza model lives in ``stanza_resources`` at the repository root.
Override that location with ``STANZA_RESOURCES_DIR`` or ``--stanza_dir``.

One-time setup on a machine with internet access:
  source .venv/bin/activate
  pip install stanza
  python extract_syntax.py --download_model
  python extract_syntax.py --probe

Usage (compute node / login node, offline):
  python extract_syntax.py --subtask NoPnx-NP
  python extract_syntax.py --subtask NoPnx-NP --split train --split dev
  python extract_syntax.py --subtask NoPnx-NP --out outputs/syntax_cache
"""
import argparse
import json
import os
import pickle
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_STANZA_DIR = os.environ.get(
    "STANZA_RESOURCES_DIR", str(PROJECT_DIR / "stanza_resources")
)

# Set the resources directory before the first Stanza import.
os.environ.setdefault("STANZA_RESOURCES_DIR", DEFAULT_STANZA_DIR)

SUBTASK_DATASETS = {
    "PA":        "MBZUAI/AraSeg-2026-Shared-Task-PA",
    "NoPnx-PA":  "MBZUAI/AraSeg-2026-Shared-Task-NoPnx-PA",
    "NP":        "MBZUAI/AraSeg-2026-Shared-Task-NP",
    "NoPnx-NP":  "MBZUAI/AraSeg-2026-Shared-Task-NoPnx-NP",
}

CLAUSE_DEPS = frozenset(["root", "ccomp", "xcomp", "advcl", "relcl", "acl", "conj"])


# ── Parser probe ──────────────────────────────────────────────────────────────

def _try_stanza(stanza_dir: str = DEFAULT_STANZA_DIR):
    try:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ""   # hide GPU before torch/stanza import — prevents slow CUDA context init
        os.environ["STANZA_RESOURCES_DIR"] = stanza_dir
        import stanza
        # Implementation note: use_gpu=False — stanza char-LSTM emits non-contiguous tensors, cuDNN rejects them
        # Implementation note: lemma required by depparse, but the real seq2seq lemmatizer OOMs (>32G); use the
        #           identity lemmatizer — satisfies the prerequisite, near-zero RAM, lemma output unused anyway
        nlp = stanza.Pipeline("ar", processors="tokenize,lemma,pos,depparse",
                               lemma_use_identity=True,
                               tokenize_pretokenized=True, use_gpu=False, verbose=False,
                               download_method=stanza.DownloadMethod.NONE)
        nlp([["مرحبا", "بالعالم"]])
        return nlp
    except Exception as e:
        print(f"  stanza unavailable: {type(e).__name__}: {e}")
        return None


def _try_camel():
    try:
        from camel_tools.morphology.database import MorphologyDB
        from camel_tools.morphology.analyzer import Analyzer
        db = MorphologyDB.builtin_db()
        return Analyzer(db)
    except Exception as e:
        print(f"  camel_tools unavailable: {type(e).__name__}: {e}")
        return None


def download_model(stanza_dir: str = DEFAULT_STANZA_DIR):
    """Download stanza Arabic model to the project directory. Needs internet (login node only)."""
    import os, stanza
    Path(stanza_dir).mkdir(parents=True, exist_ok=True)
    os.environ["STANZA_RESOURCES_DIR"] = stanza_dir  # ensure stanza sees it
    print(f"Downloading stanza Arabic model to {stanza_dir} ...")
    stanza.download("ar")   # uses STANZA_RESOURCES_DIR, no dir= kwarg needed
    print("Done. Now run --probe to verify.")


def probe(stanza_dir: str = DEFAULT_STANZA_DIR):
    print("Probing parsers on this machine...")
    print(f"  stanza_dir: {stanza_dir}")
    nlp = _try_stanza(stanza_dir)
    if nlp:
        print("  [OK] stanza: Arabic POS + dependency parsing available")
        return "stanza"
    ana = _try_camel()
    if ana:
        print("  [OK] camel_tools: Arabic morphology/POS (no dep parsing)")
        return "camel"
    print("  [FALLBACK] No parser found — extraction will write zero features")
    return "empty"


# ── Feature extraction backends ───────────────────────────────────────────────

def _feats_stanza(nlp, words):
    try:
        doc = nlp([words])
        sent = doc.sentences[0]
    except Exception:
        return _feats_empty(words)
    result = []
    for w in sent.words:
        dep = (w.deprel or "dep").split(":")[0]
        result.append({
            "pos": w.upos or "X",
            "dep": dep,
            "head_dist": abs(w.head - w.id) if (w.head is not None and w.head >= 0) else 0,
            "is_clause_head": int(dep in CLAUSE_DEPS),
        })
    # Guard against re-tokenization length mismatch
    while len(result) < len(words):
        result.append({"pos": "X", "dep": "dep", "head_dist": 0, "is_clause_head": 0})
    return result[:len(words)]


def _feats_camel(ana, words):
    result = []
    for w in words:
        analyses = ana.analyze(w)
        pos = str(analyses[0].get("pos", "X")) if analyses else "X"
        result.append({"pos": pos, "dep": "dep", "head_dist": 0, "is_clause_head": 0})
    return result


def _feats_empty(words):
    return [{"pos": "X", "dep": "dep", "head_dist": 0, "is_clause_head": 0}] * len(words)


def _subtask_file_stem(subtask: str) -> str:
    return subtask.lower().replace("-", "_")


def _feats_from_stanza_sentence(words, sent):
    result = []
    for w in sent.words:
        dep = (w.deprel or "dep").split(":")[0]
        result.append({
            "pos": w.upos or "X",
            "dep": dep,
            "head_dist": abs(w.head - w.id) if (w.head is not None and w.head >= 0) else 0,
            "is_clause_head": int(dep in CLAUSE_DEPS),
        })
    while len(result) < len(words):
        result.append({"pos": "X", "dep": "dep", "head_dist": 0, "is_clause_head": 0})
    return result[:len(words)]


def _build_stanza_cache(nlp, ids, token_lists, chunk_size: int = 4, max_words_per_parse: int = 128):
    chunk_size = max(1, int(chunk_size))
    max_words_per_parse = max(1, int(max_words_per_parse))
    cache = {}
    all_pos, all_dep = set(), set()
    for i in range(0, len(token_lists), chunk_size):
        chunk_ids = ids[i:i + chunk_size]
        chunk_tokens = token_lists[i:i + chunk_size]
        for doc_id, words in zip(chunk_ids, chunk_tokens):
            doc_feats = []
            for j in range(0, len(words), max_words_per_parse):
                word_window = words[j:j + max_words_per_parse]
                try:
                    sent = nlp([word_window]).sentences[0]
                    doc_feats.extend(_feats_from_stanza_sentence(word_window, sent))
                except Exception:
                    doc_feats.extend(_feats_empty(word_window))
            cache[doc_id] = doc_feats[:len(words)]
            for f in cache[doc_id]:
                all_pos.add(f["pos"])
                all_dep.add(f["dep"])
    return cache, all_pos, all_dep


# ── Main extraction ───────────────────────────────────────────────────────────

def extract(subtask: str, split: str, out_dir: str, stanza_dir: str = DEFAULT_STANZA_DIR, chunk_size: int = 4, max_words_per_parse: int = 128):
    from datasets import load_dataset

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = _subtask_file_stem(subtask)
    cache_path = out / f"{stem}_{split}.pkl"
    vocab_path = out / f"{stem}_vocab.json"

    if cache_path.exists():
        print(f"Already cached: {cache_path}  (delete to re-extract)")
        if vocab_path.exists():
            v = json.loads(vocab_path.read_text())
            print(f"  Vocab: n_pos={v['n_pos_tags']}, n_dep={v['n_dep_rels']}, mode={v['mode']}")
        return

    # Select parser
    nlp = _try_stanza(stanza_dir)
    if nlp:
        mode = "stanza"
        feat_fn = lambda words: _feats_stanza(nlp, words)
        print(f"Extracting [{subtask}/{split}] with Stanza (POS + dep)...")
    else:
        ana = _try_camel()
        if ana:
            mode = "camel"
            feat_fn = lambda words: _feats_camel(ana, words)
            print(f"Extracting [{subtask}/{split}] with CAMeL Tools (POS only)...")
        else:
            mode = "empty"
            feat_fn = _feats_empty
            print(f"Extracting [{subtask}/{split}] with zero features (no parser)...")

    ds = load_dataset(SUBTASK_DATASETS[subtask], split=split)
    row0 = ds[0]
    token_col = next(k for k in ["tokens", "words"] if k in row0 and isinstance(row0[k], list))
    id_col = next((k for k in ["doc_id", "id"] if k in row0), None)

    ids = [str(row[id_col]) if id_col else str(i) for i, row in enumerate(ds)]
    token_lists = [row[token_col] for row in ds]

    cache = {}
    all_pos, all_dep = set(), set()
    if mode == "stanza":
        import time
        t0 = time.time()
        cache, all_pos, all_dep = _build_stanza_cache(
            nlp, ids, token_lists, chunk_size=chunk_size,
            max_words_per_parse=max_words_per_parse,
        )
        for done in range(chunk_size, len(ds) + chunk_size, chunk_size):
            done = min(done, len(ds))
            print(f"  {done}/{len(ds)}  elapsed={time.time()-t0:.0f}s")
    else:
        for i, (doc_id, words) in enumerate(zip(ids, token_lists)):
            cache[doc_id] = feat_fn(words)
            for f in cache[doc_id]:
                all_pos.add(f["pos"])
                all_dep.add(f["dep"])
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(ds)} docs")

    pos_vocab = sorted(all_pos)
    dep_vocab = sorted(all_dep)

    with open(cache_path, "wb") as f:
        pickle.dump({
            "mode": mode, "subtask": subtask, "split": split,
            "pos_vocab": pos_vocab, "dep_vocab": dep_vocab,
            "cache": cache,
        }, f)

    # Merge vocab across splits (union)
    existing = json.loads(vocab_path.read_text()) if vocab_path.exists() else {"pos_vocab": [], "dep_vocab": []}
    merged_pos = sorted(set(existing["pos_vocab"]) | set(pos_vocab))
    merged_dep = sorted(set(existing["dep_vocab"]) | set(dep_vocab))
    vocab_path.write_text(json.dumps({
        "pos_vocab": merged_pos,
        "dep_vocab": merged_dep,
        "n_pos_tags": len(merged_pos),
        "n_dep_rels": len(merged_dep),
        "mode": mode,
    }, indent=2, ensure_ascii=False))

    print(f"Saved {len(cache)} docs → {cache_path}")
    print(f"POS vocab: {len(pos_vocab)} | DEP vocab: {len(dep_vocab)}")
    print(f"Vocab file → {vocab_path}")
    print(f"  (use n_pos_tags: {len(merged_pos)} and n_dep_rels: {len(merged_dep)} in your config)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="Check which parser is available")
    ap.add_argument("--download_model", action="store_true",
                    help="Download stanza Arabic model to stanza_dir (needs internet, login node only)")
    ap.add_argument("--subtask", choices=list(SUBTASK_DATASETS))
    ap.add_argument("--split", action="append", default=[], dest="splits",
                    help="Split to extract (repeatable; default: train dev test)")
    ap.add_argument("--out", default="outputs/syntax_cache")
    ap.add_argument("--stanza_dir", default=DEFAULT_STANZA_DIR,
                    help=f"Where stanza Arabic model lives (default: {DEFAULT_STANZA_DIR})")
    ap.add_argument("--chunk_size", type=int, default=4,
                    help="Number of documents per Stanza call. Lower values use less RAM.")
    ap.add_argument("--max_words_per_parse", type=int, default=128,
                    help="Maximum words passed to one Stanza parse call. Lower values use less RAM.")
    args = ap.parse_args()

    if args.download_model:
        download_model(args.stanza_dir)
        return

    if args.probe:
        probe(args.stanza_dir)
        return

    if not args.subtask:
        ap.error("--subtask required (unless --probe or --download_model)")

    for split in (args.splits or ["train", "dev", "test"]):
        print(f"\n--- {args.subtask} / {split} ---")
        extract(args.subtask, split, args.out, args.stanza_dir, args.chunk_size, args.max_words_per_parse)


if __name__ == "__main__":
    main()
