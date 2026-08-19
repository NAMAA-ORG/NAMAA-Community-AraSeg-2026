#!/bin/bash
# One-off: e64 — AraBERT-large-v02 with the tuned PA recipe. The clean single-model
# test of "does a LARGE Arabic encoder close the gap to leaderboard #1 on PA?"
# Submit from the cluster:  sbatch experiments/slurm/run_e64.sh
#SBATCH --job-name=araseg-e64
#SBATCH -p kisski
#SBATCH -G A100:1
# 32G/1.5h, not the 80G/4h the LLM jobs use: AraBERT-large is a 370M encoder on 174 train docs.
#SBATCH --mem-per-gpu=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=logs/slurm_%j_%x.out
#SBATCH --error=logs/slurm_%j_%x.err

PROJECT=${ARASEG_PROJECT:?set ARASEG_PROJECT}
export HF_HOME=${ARASEG_HF_HOME:-$HOME/.cache/huggingface}
export HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

source ${PROJECT}/.venv/bin/activate
cd ${PROJECT}/experiments
echo "=== AraSeg e64: AraBERT-large-v02 tuned PA ==="
echo "Node: $(hostname)  |  Started: $(date)"
python train.py --config configs/e64_arabertv2_large_pa_tuned.yaml
echo "Finished: $(date)"
