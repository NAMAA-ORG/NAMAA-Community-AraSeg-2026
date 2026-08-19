#!/usr/bin/env bash
# Generic one-GPU SLURM launcher.
# Submit with: sbatch --export=ALL,CONFIG=configs/e25_xlmr_single_pa_stride256.yaml experiments/slurm/run_experiment.sh
# Optional: set VENV=/path/to/venv and standard Hugging Face cache variables.
#SBATCH --job-name=araseg
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm_%j_%x.out
#SBATCH --error=logs/slurm_%j_%x.err
set -euo pipefail

PROJECT="${ARASEG_PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CONFIG="${CONFIG:?Set CONFIG to a path relative to experiments/, for example configs/e25_xlmr_single_pa_stride256.yaml}"
VENV="${VENV:-${PROJECT}/.venv}"

mkdir -p "${PROJECT}/experiments/logs"
if [[ -f "${VENV}/bin/activate" ]]; then
  source "${VENV}/bin/activate"
fi

cd "${PROJECT}/experiments"
python train.py --config "${CONFIG}"
