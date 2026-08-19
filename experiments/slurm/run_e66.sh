#!/bin/bash
# e66 — RemBERT on the tuned PA recipe. New pretraining family for the PA pool
# (currently XLM-R + Qwen + Gemma, flat logit average, +0.09 over iMak AI Lab).
# Submit from INSIDE experiments/:  cd experiments && sbatch slurm/run_e66.sh
# (--output is relative to the submission CWD; from the repo root it looks for a nonexistent <root>/logs/)
#
# PREREQ: google/rembert must be in the HF cache — these jobs run with HF_HUB_OFFLINE=1.
#   On a login node (has internet):
#     source ${PROJECT}/.venv/bin/activate
#     HF_HOME=${ARASEG_HF_HOME:-$HOME/.cache/huggingface} \
#       python -c "from huggingface_hub import snapshot_download; snapshot_download('google/rembert')"
#SBATCH --job-name=araseg-e66
#SBATCH -p kisski
#SBATCH -G A100:1
# 32G/1.5h, not the 80G/4h the LLM jobs use: RemBERT is a 576M encoder on 174 train docs.
# A big ask on a `mix` partition can't backfill into a node whose host RAM is already spoken for.
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
echo "=== AraSeg e66: RemBERT tuned PA ==="
echo "Node: $(hostname)  |  Started: $(date)"
python train.py --config configs/e66_rembert_pa_tuned.yaml
echo "Finished: $(date)"
