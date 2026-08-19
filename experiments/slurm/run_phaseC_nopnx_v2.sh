#!/bin/bash
# Phase C v2: re-run ONLY the two NoPnx stacked ensembles with the aux-punct-head
# members (e60/e61) swapping in for the near-dead recall members (e50/e51/e52/e53),
# plus e62/e63 as diversity and e57 (syntax) as a secondary NoPnx-PA candidate the
# stacker can weigh. Writes to phaseC_v2_* so the locked phaseC/phaseA CSVs stay
# intact for comparison. Inference only (cached probs), ~25 min/subtask.
#
# Usage (login node):
#   cd ${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments
#   sbatch slurm/run_phaseC_nopnx_v2.sh

#SBATCH --job-name=araseg-phaseC-v2
#SBATCH --array=0-1
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

SUBTASKS=(NoPnx-PA NoPnx-NP)
ST=${SUBTASKS[$SLURM_ARRAY_TASK_ID]}
key=$(echo "$ST" | tr 'A-Z-' 'a-z_')

echo "Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) | Subtask: $ST | Started: $(date)"

declare -A POOL
# Phase A/C encoder+LLM base, recall members dropped (near-zero/negative weight),
# aux-punct-head members added (e60/e61 strictly better; e62/e63 for diversity).
POOL[NoPnx-PA]="configs/e41_qwen35_9b_nopnx_pa.yaml \
                configs/e17_xlmr_single_nopnxpa_s1.yaml \
                configs/e18_xlmr_single_nopnxpa_s2.yaml \
                configs/e11_xlmr_single_nopnxpa_w2.yaml \
                configs/e38_xlmr_multitask_joint.yaml \
                configs/e61_xlmr_nopnx_pa_aux_s1.yaml \
                configs/e63_xlmr_nopnx_pa_aux_drop05_s1.yaml \
                configs/e57_xlmr_nopnx_pa_syntax_s1.yaml"

POOL[NoPnx-NP]="configs/e42_qwen35_9b_nopnx_np.yaml \
                configs/e19_xlmr_single_nopnxnp_s1.yaml \
                configs/e20_xlmr_single_nopnxnp_s2.yaml \
                configs/e12_xlmr_single_nopnxnp_w2.yaml \
                configs/e38_xlmr_multitask_joint.yaml \
                configs/e60_xlmr_nopnx_np_aux_s1.yaml \
                configs/e62_xlmr_nopnx_np_aux_drop05_s1.yaml"

# Exhaustive prob/logit sweep + full-pool stack (same as Phase C)
for MODE in prob logit; do
    python ensemble.py --subtask "$ST" --members ${POOL[$ST]} \
        --output_dir "outputs/phaseC_v2_${key}_${MODE}" \
        --combine "$MODE" --exhaustive --exhaustive_top_k 15 --thr_step 0.01
done

python ensemble.py --subtask "$ST" --members ${POOL[$ST]} \
    --output_dir "outputs/phaseC_v2_${key}_stack" \
    --combine stack --thr_step 0.01

echo "Finished: $ST at $(date)"
