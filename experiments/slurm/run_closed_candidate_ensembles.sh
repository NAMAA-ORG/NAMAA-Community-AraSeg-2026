#!/bin/bash
#SBATCH --job-name=araseg-cand-ens
#SBATCH -p kisski
#SBATCH -G A100:1
#SBATCH --mem-per-gpu=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm_%j_%x.out
#SBATCH --error=logs/slurm_%j_%x.err

# Closed-track-only candidate reruns for subsets that beat the automatic
# dev/parsimony winner in the first exhaustive sweep's local test table.
#
# Submit from the experiments directory on KISSKI:
#   cd ${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments
#   sbatch slurm/run_closed_candidate_ensembles.sh

set -uo pipefail

run() { echo "+ $*"; "$@" || echo "[WARN] STEP FAILED ($?): $*"; }

PROJECT=${ARASEG_PROJECT:?set ARASEG_PROJECT}
EXPERIMENTS_DIR=${PROJECT}/experiments

export HF_HOME=${ARASEG_HF_HOME:-$HOME/.cache/huggingface}
export HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

source ${ARASEG_PROJECT:?set ARASEG_PROJECT}/.venv-llm/bin/activate
cd "${EXPERIMENTS_DIR}"

echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started: $(date)"

C=configs

run python ensemble.py --subtask NoPnx-PA --output_dir outputs/ens_closed_nopnxpa_e41_e17 \
    --threshold_policy fixed --threshold 0.5 \
    --members $C/e41_qwen35_9b_nopnx_pa.yaml $C/e17_xlmr_single_nopnxpa_s1.yaml

run python ensemble.py --subtask NP --output_dir outputs/ens_closed_np_e40_e15 \
    --members $C/e40_qwen35_9b_np.yaml $C/e15_xlmr_single_np_s1.yaml

echo "Finished: $(date)"
echo "Review:"
echo "  outputs/ens_closed_nopnxpa_e41_e17/ensemble_NoPnx_PA.json"
echo "  outputs/ens_closed_np_e40_e15/ensemble_NP.json"
