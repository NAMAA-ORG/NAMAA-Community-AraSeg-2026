#!/bin/bash
# Phase C: final stacked ensemble sweep — Phase A pool + Phase B recall models.
# Syntax models (e56-e59) are NOT included yet; add them in a follow-up run once
# syntax_cache vocab files land. One array task per subtask.
#
# Usage (login node):
#   cd ${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments
#   sbatch slurm/run_phaseC_ensembles.sh

#SBATCH --job-name=araseg-phaseC
#SBATCH --array=0-3
#SBATCH -p kisski
#SBATCH -G A100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments/logs/slurm_%A_%a_%x.out
#SBATCH --error=${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments/logs/slurm_%A_%a_%x.err

set -euo pipefail

export HF_HOME=${ARASEG_HF_HOME:-$HOME/.cache/huggingface}
export HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

source ${ARASEG_PROJECT:?set ARASEG_PROJECT}/.venv-llm/bin/activate
cd ${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments

SUBTASKS=(PA NP NoPnx-PA NoPnx-NP)
ST=${SUBTASKS[$SLURM_ARRAY_TASK_ID]}
key=$(echo "$ST" | tr 'A-Z-' 'a-z_')

echo "Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) | Subtask: $ST | Started: $(date)"

declare -A POOL
# Phase A members (same as run_phaseA_ensembles.sh) + Phase B recall-tilted members
POOL[PA]="configs/e34_qwen35_9b_pa_rdrop.yaml \
          configs/e25_xlmr_single_pa_stride256.yaml \
          configs/e26_xlmr_pa_bilstm.yaml \
          configs/e30_xlmr_pa_kitchen_sink.yaml \
          configs/e23_araelectra_single_pa.yaml \
          configs/e21_mdeberta_single_pa.yaml \
          configs/e22_marbertv2_single_pa.yaml \
          configs/e55_xlmr_pa_recall_s1.yaml"

POOL[NP]="configs/e40_qwen35_9b_np.yaml \
          configs/e15_xlmr_single_np_s1.yaml \
          configs/e16_xlmr_single_np_s2.yaml \
          configs/e38_xlmr_multitask_joint.yaml \
          configs/e54_xlmr_np_recall_s1.yaml"

POOL[NoPnx-PA]="configs/e41_qwen35_9b_nopnx_pa.yaml \
                configs/e17_xlmr_single_nopnxpa_s1.yaml \
                configs/e18_xlmr_single_nopnxpa_s2.yaml \
                configs/e11_xlmr_single_nopnxpa_w2.yaml \
                configs/e38_xlmr_multitask_joint.yaml \
                configs/e51_xlmr_nopnx_pa_recall_s1.yaml \
                configs/e53_xlmr_nopnx_pa_recall_s2.yaml"

POOL[NoPnx-NP]="configs/e42_qwen35_9b_nopnx_np.yaml \
                configs/e19_xlmr_single_nopnxnp_s1.yaml \
                configs/e20_xlmr_single_nopnxnp_s2.yaml \
                configs/e12_xlmr_single_nopnxnp_w2.yaml \
                configs/e38_xlmr_multitask_joint.yaml \
                configs/e50_xlmr_nopnx_np_recall_s1.yaml \
                configs/e52_xlmr_nopnx_np_recall_s2.yaml"

# Exhaustive prob/logit sweep + full-pool stack
for MODE in prob logit; do
    python ensemble.py --subtask "$ST" --members ${POOL[$ST]} \
        --output_dir "outputs/phaseC_${key}_${MODE}" \
        --combine "$MODE" --exhaustive --exhaustive_top_k 15 --thr_step 0.01
done

python ensemble.py --subtask "$ST" --members ${POOL[$ST]} \
    --output_dir "outputs/phaseC_${key}_stack" \
    --combine stack --thr_step 0.01

echo "Finished: $ST at $(date)"
