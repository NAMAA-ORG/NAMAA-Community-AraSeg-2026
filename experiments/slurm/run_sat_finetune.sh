#!/bin/bash
#SBATCH --job-name=araseg-sat-ft
#SBATCH -p kisski
#SBATCH -G A100:1
#SBATCH --mem-per-gpu=40G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm_%j_%x.out
#SBATCH --error=logs/slurm_%j_%x.err
#
# Full fine-tune (default) or LoRA (--lora) of SaT-12l-sm on all 4 AraSeg subtasks,
# ~6 min each on A100. Writes ensemble-ready prob caches to outputs/sat_fullft_caches/
# and logs to the main experiments/mlflow.db (experiment AraSeg-Per-Dataset).
#
# PREREQ (one-time, on a LOGIN node with internet, since compute nodes are offline):
#   source .venv-llm/bin/activate
#   pip install wtpsplit
#   python -c "from wtpsplit import SaT; SaT('sat-12l-sm')"          # cache the model
#   python experiments/download_datasets.py                          # cache the 4 datasets
#
# Submit:
#   sbatch slurm/run_sat_finetune.sh                                 # full-FT, plain
#   sbatch --export=ALL,ARGS="--punct-corrupt --viterbi" slurm/run_sat_finetune.sh
#   sbatch --export=ALL,ARGS="--lora" slurm/run_sat_finetune.sh
#
#   # data-scale variant: train on train+dev (396 docs, 2.3x), no val split, score test.
#   # --tag/--out-dir keep it off the locked sat_ft caches and checkpoints.
#   sbatch --export=ALL,ARGS="--train-splits train dev --val-split none --eval-splits test \
#     --tag sat_ft_traindev --out-dir outputs/sat_ft_traindev" slurm/run_sat_finetune.sh

set -euo pipefail

PROJECT=${ARASEG_PROJECT:?set ARASEG_PROJECT}
export HF_HOME=${ARASEG_HF_HOME:-$HOME/.cache/huggingface}
export HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

VENV=${VENV:-${PROJECT}/.venv-llm}
source ${VENV}/bin/activate

echo "=== SaT fine-tune: ARGS='${ARGS:-}' ==="
echo "Node:  $(hostname)"
echo "GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Started: $(date)"

cd ${PROJECT}/experiments
python sat_finetune.py ${ARGS:-}

echo "Finished: $(date)"
