#!/bin/bash
# e71 — Kareem's recipe: train XLM-R on Naqta-restored NoPnx-PA text, then eval on
# restored test through the same map-back the inference version used.
#   sbatch experiments/slurm/run_e71_naqta_restore.sh
#
# Why this and not the inference version: restore_then_segment.py --mode naqta borrows
# the FROZEN PA lock and scores ~0.82 (< the 0.8718 NoPnx-PA lock) because that lock
# never saw partial punctuation. e71 TRAINS on the restored text, so it adapts to
# Naqta's noisy/partial restoration -- the difference between Kareem's +2 and our -0.5.
#
# Prereqs already on the cluster from the earlier runs: the NoPnx-PA dataset cache and
# the dev/test naqta_full caches. This script adds the missing train cache itself.
#SBATCH --job-name=araseg-e71-naqta
#SBATCH -p kisski
#SBATCH -G A100:1
#SBATCH --mem-per-gpu=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
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
mkdir -p outputs/e71 logs

echo "=== e71: XLM-R on Naqta-restored NoPnx-PA (Kareem's recipe) ==="
echo "Node: $(hostname)  |  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Started: $(date)"

python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 'no CUDA — login node?')" || exit 1

MINP=0.3   # must match naqta_restore_min_p in the config; the eval reuses it

# (a) Naqta train cache — the one artifact the earlier dev/test runs didn't make.
echo "--- (a) naqta train cache ---"
if [ -f outputs/prob_cache/NoPnx_PA_train_naqta_full.pkl ]; then
  echo "  [skip] NoPnx_PA_train_naqta_full.pkl"
else
  python naqta_cache.py --subtasks NoPnx-PA --splits train || exit 1
fi

# (b) Train e71. Idempotent: train.py skips if outputs/e71/results.json exists.
echo "--- (b) train e71 ---"
if [ -f outputs/e71/results.json ]; then
  echo "  [skip] outputs/e71/results.json"
else
  python train.py --config configs/e71_naqta_restore_nopnxpa.yaml || exit 1
fi

# (c) Tune threshold on restored DEV (writes outputs/e71/restore_thr.txt), then apply
#     it ONCE to restored TEST. Threshold picked on dev, used on test = closed-legal.
# --mode is argparse-required but ignored on the --trained-member path; pass naqta.
echo "--- (c) dev threshold sweep ---"
python restore_then_segment.py --subtask NoPnx-PA --split dev --mode naqta \
  --trained-member configs/e71_naqta_restore_nopnxpa.yaml --min-punct-p $MINP || exit 1

echo "--- (c) test at the dev-tuned threshold ---"
python restore_then_segment.py --subtask NoPnx-PA --split test --mode naqta \
  --trained-member configs/e71_naqta_restore_nopnxpa.yaml --min-punct-p $MINP || exit 1

echo ""
echo "Compare the TEST F1 above against e17 TEST = 0.8147 — the MATCHED CONTROL"
echo "(same config, restoration off). NOT against the 0.8718 lock: that is a"
echo "six-member decoder ensemble, and one model vs six is a category error that"
echo "already made this experiment look dead once. And NOT against e17's DEV 0.8189:"
echo "test-vs-dev is the second category error, and it understated this by 0.4 F1."
echo "Result 2026-07-24: 0.8253 vs 0.8147 = +1.06."
echo "A control win is still not a re-lock — it must clear gate_member.py on the"
echo "TEST CI first (e69 was +0.41 and still rejected)."
echo "Finished: $(date)"
