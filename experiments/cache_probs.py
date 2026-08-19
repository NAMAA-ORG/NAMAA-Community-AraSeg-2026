"""Generic per-member probability caches (queue A1/A2).

One GPU forward pass per (member, subtask, split) -> {doc_id: (prob_per_word, gold)}
pickled to `<out_dir>/<subtask>_<split>_<exp_id>.pkl`. Same format and naming scheme
the SaT caches already use (tag = exp_id instead of `sat_ft`), so every downstream
consumer — ensembling, threshold search, bootstrap, the OOF decoder, official eval —
reads members off disk and runs on CPU. Nothing but the forward pass needs a GPU.

Idempotent: an existing cache file is skipped unless --overwrite.

Usage:
    python cache_probs.py --subtask NoPnx-PA --splits dev test \
        --members configs/e41_qwen35_9b_nopnx_pa.yaml configs/e17_*.yaml ... \
        --out_dir outputs/prob_cache
"""
import argparse
import pickle
from pathlib import Path

import numpy as np

from naqta_restore import map_probs_back


def cache_path(out_dir: Path, subtask: str, split: str, tag: str) -> Path:
    return out_dir / f"{subtask.replace('-', '_')}_{split}_{tag}.pkl"


def load_cached_member(cache_dir, subtask: str, split: str, tag: str):
    """Read one cached member. Raises if it was never computed."""
    fp = cache_path(Path(cache_dir), subtask, split, tag)
    if not fp.exists():
        raise FileNotFoundError(f"no cache for member {tag} ({subtask}/{split}): {fp} "
                                f"— run cache_probs.py for it first")
    return pickle.load(open(fp, "rb"))


def map_restored_docs(restored_docs, owners, original_gold) -> dict:
    """Map restored-space probabilities back onto each original document."""
    out = {}
    for doc_id, (probs, _) in restored_docs.items():
        gold = np.asarray(original_gold[doc_id])
        # A label-free split (blind) must arrive here as a ZEROS placeholder of the
        # original length, the way data_utils._process builds it -- not as []. With []
        # map_probs_back allocates a size-0 array and dies on `index 0 is out of bounds`
        # about forty lines away from the actual cause. Fail here, by name.
        assert len(gold) > 0, (
            f"{doc_id}: empty gold. restore_then_segment._docs returns [] when labels "
            f"are hidden; the caller must substitute [0]*len(tokens) for blind.")
        mapped = map_probs_back(probs, owners[doc_id], len(gold))
        assert len(mapped) == len(gold), f"{doc_id}: probs {len(mapped)} vs gold {len(gold)}"
        out[doc_id] = (mapped, gold)
    return out


def main():
    import torch
    from ensemble import _member_probs
    from train import Config

    ap = argparse.ArgumentParser()
    ap.add_argument("--subtask", required=True, choices=["PA", "NoPnx-PA", "NP", "NoPnx-NP"])
    ap.add_argument("--members", required=True, nargs="+", help="Member config YAML paths")
    ap.add_argument("--splits", nargs="+", default=["dev", "test"])
    ap.add_argument("--out_dir", default="outputs/prob_cache")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[cache] subtask={args.subtask} splits={args.splits} device={device}")

    for path in args.members:
        cfg = Config.from_yaml(path)
        for split in args.splits:
            fp = cache_path(out, args.subtask, split, cfg.experiment_id)
            if fp.exists() and not args.overwrite:
                print(f"  [skip] {fp.name} exists")
                continue
            if cfg.naqta_restore:
                from naqta_restore import build_restored_docs
                from restore_then_segment import _docs

                records, owners = build_restored_docs(
                    args.subtask, split, cfg.naqta_restore_min_p, comma=cfg.naqta_comma)
                docs = _member_probs(cfg, args.subtask, split, device, records=records)
                # `_docs` yields labels [] when they are hidden. data_utils._process
                # uses [0]*len(tokens) for exactly that case, and every non-restored
                # blind cache on disk carries that placeholder -- so match it, or this
                # member's pickle would have a different shape from the ones it gets
                # decoded beside.
                original_gold = {doc_id: (gold if gold else [0] * len(toks))
                                 for doc_id, (toks, gold) in _docs(
                                     args.subtask, split).items()}
                docs = map_restored_docs(docs, owners, original_gold)
            else:
                docs = _member_probs(cfg, args.subtask, split, device)
            assert all(len(probs) == len(gold) for probs, gold in docs.values())
            with open(fp, "wb") as f:
                pickle.dump(docs, f)
            print(f"  [ok  ] {cfg.experiment_id} {split}: {len(docs)} docs -> {fp}")


if __name__ == "__main__":
    main()
