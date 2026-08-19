#!/bin/bash
# Phase C v2 (PA): re-run the PA stacked ensemble with an EXPANDED pool that adds
# our strong-but-never-pooled PA singles — e32 (Qwen3.5-9B, our best PA single
# 0.9334), e24 (XLM-R XL), e33 (Gemma-4-12B) — on top of the Phase A/C PA pool.
# PA is the one task we trail closed #1 on (-0.14), and its pool has never seen
# e32. Writes to phaseC_v2_pa_* so the locked phaseA_pa_stack CSV (0.9396) stays
# intact for comparison. Inference only (rebuilds members, no training).
#
# NP is intentionally NOT re-run: its only queued candidate (e16) is already in
# the Phase A/C NP pool, so a rerun would reproduce the locked NP result exactly.
#
# e37 (XLM-R + CRF) is DELIBERATELY EXCLUDED by default: the ensemble reads
# per-word softmax probs, but the CRF head returns per-WORD emissions via
# word_idx (model.py:229-248) — the prob path is unverified and may misalign or
# crash. It is also the lowest-value candidate. To try it, uncomment the e37 line
# in POOL below and rerun into a separate output dir first to confirm it loads.
#
# Usage (login node):
#   cd ${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments
#   sbatch slurm/run_phaseC_pa_v2.sh

#SBATCH --job-name=araseg-phaseC-pa-v2
#SBATCH -p kisski
#SBATCH -G A100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments/logs/slurm_%A_%x.out
#SBATCH --error=${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments/logs/slurm_%A_%x.err

set -euo pipefail

export HF_HOME=${ARASEG_HF_HOME:-$HOME/.cache/huggingface}
export HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

source ${ARASEG_PROJECT:?set ARASEG_PROJECT}/.venv-llm/bin/activate
cd ${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments

ST=PA
key=pa

echo "Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) | Subtask: $ST | Started: $(date)"

# Phase A/C PA pool  +  strong never-pooled PA singles (e32/e24/e33).
# e37 (CRF) left out — see header note; uncomment to include.
POOL="configs/e34_qwen35_9b_pa_rdrop.yaml \
      configs/e25_xlmr_single_pa_stride256.yaml \
      configs/e26_xlmr_pa_bilstm.yaml \
      configs/e30_xlmr_pa_kitchen_sink.yaml \
      configs/e23_araelectra_single_pa.yaml \
      configs/e21_mdeberta_single_pa.yaml \
      configs/e22_marbertv2_single_pa.yaml \
      configs/e55_xlmr_pa_recall_s1.yaml \
      configs/e32_qwen35_9b_pa.yaml \
      configs/e24_xlmrxl_single_pa.yaml \
      configs/e33_gemma4_12b_pa.yaml"
#     configs/e37_xlmr_crf_pa.yaml   # CRF — unverified prob path, add with care

# Exhaustive prob/logit sweep + full-pool stack (same recipe as Phase C).
for MODE in prob logit; do
    python ensemble.py --subtask "$ST" --members $POOL \
        --output_dir "outputs/phaseC_v2_${key}_${MODE}" \
        --combine "$MODE" --exhaustive --exhaustive_top_k 15 --thr_step 0.01
done

python ensemble.py --subtask "$ST" --members $POOL \
    --output_dir "outputs/phaseC_v2_${key}_stack" \
    --combine stack --thr_step 0.01

echo "Finished: $ST at $(date)"
