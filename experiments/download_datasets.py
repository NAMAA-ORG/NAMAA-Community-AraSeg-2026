"""
Pre-download all 4 AraSeg datasets to the HF cache.
Run ONCE from the login node (internet access) before submitting jobs.

Usage: python download_datasets.py
"""

from datasets import load_dataset

DATASETS = {
    "PA":       "MBZUAI/AraSeg-2026-Shared-Task-PA",
    "NoPnx-PA": "MBZUAI/AraSeg-2026-Shared-Task-NoPnx-PA",
    "NP":       "MBZUAI/AraSeg-2026-Shared-Task-NP",
    "NoPnx-NP": "MBZUAI/AraSeg-2026-Shared-Task-NoPnx-NP",
}

for subtask, ds_name in DATASETS.items():
    print(f"\nDownloading {subtask} ({ds_name})...")
    try:
        ds = load_dataset(ds_name)
    except Exception as e:
        print(f"  ERROR: {e}")
        print(f"  Try: the dataset name may be wrong — check https://huggingface.co/MBZUAI")
        continue

    for split, data in ds.items():
        print(f"  split='{split}'  docs={len(data)}  columns={data.column_names}")
    print(f"  ✓ cached")

print("\nDone. Copy the split names above into data_utils.py _SPLIT_ALIASES if they differ from 'train'/'dev'.")
