#!/bin/bash
# e70 blind prob cache — the one GPU step between the re-lock and a new NoPnx-NP submission.
# e70 cleared the bootstrap gate (+0.60 F1, CI [+0.19, +1.01]) so it joins the NoPnx-NP
# decoder; the decoder reads members off disk, and e70 has no `blind` cache yet.
#   sbatch experiments/slurm/run_e70_blind_cache.sh
#
# PREREQUISITE — run ONCE on the LOGIN node (compute nodes are offline):
#   export HF_HOME=${ARASEG_HF_HOME:-$HOME/.cache/huggingface}
#   export HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets
#   export ARASEG_BLIND_TOKEN=hf_ORGANIZERS
#   python -c "
#   import os, datasets
#   datasets.load_dataset('MBZUAI/AraSeg-2026-Shared-Task-NoPnx-NP-Blind',
#                         token=os.environ['ARASEG_BLIND_TOKEN'])"
# The blind repos are GATED, so an unprefetched cache dies on an offline-mode error.
#SBATCH --job-name=araseg-e70-blind
#SBATCH -p kisski
#SBATCH -G A100:1
#SBATCH --mem-per-gpu=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm_%j_%x.out
#SBATCH --error=logs/slurm_%j_%x.err
set -uo pipefail

PROJECT=${ARASEG_PROJECT:?set ARASEG_PROJECT}
export HF_HOME=${ARASEG_HF_HOME:-$HOME/.cache/huggingface}
export HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

source ${PROJECT}/.venv/bin/activate
cd ${PROJECT}/experiments
mkdir -p outputs/prob_cache logs

echo "=== e70 blind prob cache (NoPnx-NP) ==="
echo "Node: $(hostname)  |  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "Started: $(date)"

# Fail fast if the login-node blind prefetch was skipped — see header.
python - <<'PY' || exit 1
import sys
import datasets
try:
    d = datasets.load_dataset("MBZUAI/AraSeg-2026-Shared-Task-NoPnx-NP-Blind")
except Exception as e:
    sys.exit("\nBlind NoPnx-NP dataset is NOT in the offline cache and compute nodes have no\n"
             "network. Run the login-node prefetch in this script's header, then resubmit.\n"
             f"({type(e).__name__}: {e})")
print(f"blind split cached: {len(d['blind'])} docs")
PY

python -c "
import torch, sys
if not torch.cuda.is_available():
    sys.exit('CUDA not available. Are you on a compute node? (login nodes have no GPU)')
print('torch', torch.__version__, '| cuda', torch.version.cuda, '|', torch.cuda.get_device_name(0))
" || exit 1

python cache_probs.py --subtask NoPnx-NP --splits blind \
  --members configs/e70_naqta_nopnxnp.yaml \
  --out_dir outputs/prob_cache || exit 1

# Blind has no labels, so there is nothing to score here. What CAN go wrong is e70's
# cache covering a different doc set than the members it gets decoded beside — the
# decoder intersects doc ids, so a mismatch silently shrinks the submission instead of
# erroring. Compare against an existing member rather than a hardcoded count: the first
# version of this script asserted 100 (the runbook's figure) and aborted on the real 212.
python - <<'PY'
import pickle
import sys
from pathlib import Path
new = Path("outputs/prob_cache/NoPnx_NP_blind_e70.pkl")
ref = Path("outputs/prob_cache/NoPnx_NP_blind_e42.pkl")
a, b = pickle.load(open(new, "rb")), pickle.load(open(ref, "rb"))
print(f"{new.name}: {len(a)} docs   {ref.name}: {len(b)} docs")
if set(a) != set(b):
    only_a, only_b = sorted(set(a) - set(b))[:5], sorted(set(b) - set(a))[:5]
    sys.exit(f"doc-id MISMATCH vs e42 — only in e70: {only_a}  only in e42: {only_b}")
print("OK — doc ids match the locked members")
PY

echo "Finished: $(date)"
