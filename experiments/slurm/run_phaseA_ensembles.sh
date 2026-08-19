#!/bin/bash
#SBATCH --job-name=araseg-phaseA-ens
#SBATCH --array=0-3
#SBATCH -p kisski
#SBATCH -G A100:1
#SBATCH --mem-per-gpu=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments/logs/slurm_%A_%a_%x.out
#SBATCH --error=${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments/logs/slurm_%A_%a_%x.err

# Phase A: exhaustive sweep under prob/logit/rank + stack for all four closed
# subtasks, one SLURM array task per subtask (runs in parallel).
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

declare -A POOL
POOL[PA]="configs/e34_qwen35_9b_pa_rdrop.yaml configs/e25_xlmr_single_pa_stride256.yaml configs/e26_xlmr_pa_bilstm.yaml configs/e30_xlmr_pa_kitchen_sink.yaml configs/e23_araelectra_single_pa.yaml configs/e21_mdeberta_single_pa.yaml configs/e22_marbertv2_single_pa.yaml"
POOL[NP]="configs/e40_qwen35_9b_np.yaml configs/e15_xlmr_single_np_s1.yaml configs/e16_xlmr_single_np_s2.yaml configs/e38_xlmr_multitask_joint.yaml"
POOL[NoPnx-PA]="configs/e41_qwen35_9b_nopnx_pa.yaml configs/e17_xlmr_single_nopnxpa_s1.yaml configs/e18_xlmr_single_nopnxpa_s2.yaml configs/e11_xlmr_single_nopnxpa_w2.yaml configs/e38_xlmr_multitask_joint.yaml"
POOL[NoPnx-NP]="configs/e42_qwen35_9b_nopnx_np.yaml configs/e19_xlmr_single_nopnxnp_s1.yaml configs/e20_xlmr_single_nopnxnp_s2.yaml configs/e12_xlmr_single_nopnxnp_w2.yaml configs/e38_xlmr_multitask_joint.yaml"

echo "Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) | Subtask: $ST | Started: $(date)"

for MODE in prob logit rank; do
  python ensemble.py --subtask "$ST" --members ${POOL[$ST]} \
    --output_dir "outputs/phaseA_${key}_${MODE}" \
    --combine "$MODE" --exhaustive --exhaustive_top_k 15 --thr_step 0.01
done

# stacking: full pool, no subset search
python ensemble.py --subtask "$ST" --members ${POOL[$ST]} \
  --output_dir "outputs/phaseA_${key}_stack" \
  --combine stack --thr_step 0.01

echo "Finished: $ST at $(date)"
