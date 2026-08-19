"""Shared utilities for AraSeg experiments."""

import os
from pathlib import Path


def _base_remap() -> dict:
    """Off-cluster base-model overrides from $ARASEG_BASE_REMAP="src1=dst1;src2=dst2".

    Lets the SAME configs run on KISSKI (model_name is a KISSKI-local flat dir) and
    off-cluster (remap that path to its HF id, e.g. Qwen/Qwen3.5-9B). Unset on KISSKI
    → no remap → native paths. Empty entries ignored."""
    raw = os.environ.get("ARASEG_BASE_REMAP", "")
    out = {}
    for pair in raw.split(";"):
        pair = pair.strip()
        if "=" in pair:
            src, dst = pair.split("=", 1)
            out[src.strip()] = dst.strip()
    return out


def resolve_model_path(model_name: str) -> str:
    """
    Resolve a model name to a loadable path.

    0. $ARASEG_BASE_REMAP hit → return the mapped id/path (off-cluster override).
    1. HF model ID (no leading /) → return as-is; HF_HOME + HF cache handle it.
    2. Local path, files directly in folder → return as-is.
    3. Local path with a snapshots/ subdirectory → resolve to the snapshot dir.
    """
    remap = _base_remap()
    if model_name in remap:
        print(f"  [remap] {model_name} -> {remap[model_name]}")
        return remap[model_name]

    p = Path(model_name)

    # Not a local path — treat as HF model ID
    if not p.is_absolute():
        return model_name

    if not p.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_name}")

    snapshots_dir = p / "snapshots"
    if not snapshots_dir.is_dir():
        # Files are directly in the folder
        return model_name

    # Find snapshot subdirectory (hash-named). Use the most recently modified
    # one in the unlikely case there are multiple.
    candidates = sorted(
        (s for s in snapshots_dir.iterdir() if s.is_dir()),
        key=lambda s: s.stat().st_mtime,
    )
    if not candidates:
        raise ValueError(f"snapshots/ directory exists but is empty: {snapshots_dir}")

    resolved = str(candidates[-1])
    print(f"  [path] {p.name}/snapshots/{candidates[-1].name}")
    return resolved
