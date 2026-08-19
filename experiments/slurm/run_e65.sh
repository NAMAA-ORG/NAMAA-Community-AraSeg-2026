#!/bin/bash
# One-off: e65 — Fanar-1-9B (Arabic-native) as bidirectional LoRA encoder, PA, R-Drop.
# Uses .venv-llm (peft + newer transformers), 12h like the other 9B jobs.
# Submit from the cluster:  sbatch experiments/slurm/run_e65.sh
#SBATCH --job-name=araseg-e65
#SBATCH -p kisski
#SBATCH -G A100:1
#SBATCH --mem-per-gpu=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm_%j_%x.out
#SBATCH --error=logs/slurm_%j_%x.err

PROJECT=${ARASEG_PROJECT:?set ARASEG_PROJECT}
export HF_HOME=${ARASEG_HF_HOME:-$HOME/.cache/huggingface}
export HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

source ${PROJECT}/.venv-llm/bin/activate
cd ${PROJECT}/experiments
echo "=== AraSeg e65: Fanar-1-9B tuned PA (R-Drop) ==="
echo "Node: $(hostname)  |  Started: $(date)"
echo "Transf.: $(python -c 'import transformers; print(transformers.__version__)' 2>/dev/null)"
python train.py --config configs/e65_fanar_9b_pa_rdrop.yaml
echo "Finished: $(date)"
