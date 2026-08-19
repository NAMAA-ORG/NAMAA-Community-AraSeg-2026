#!/bin/bash
# Correct PA logit search + e32 fold-bag + matched e75/e76 restoration gate.
# Submit from ${PROJECT}/experiments:
#   sbatch slurm/run_remaining_gates.sh
#SBATCH --job-name=araseg-remaining-gates
#SBATCH -A kisski-futurmig
#SBATCH -p kisski
#SBATCH -G A100:1
#SBATCH --mem-per-gpu=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments/logs/slurm_%j_%x.out
#SBATCH --error=${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments/logs/slurm_%j_%x.err
set -euo pipefail

PROJECT=${ARASEG_PROJECT:?set ARASEG_PROJECT}
export HF_HOME=${ARASEG_HF_HOME:-$HOME/.cache/huggingface}
export HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

source "${PROJECT}/.venv-llm/bin/activate"
cd "${PROJECT}/experiments"
mkdir -p logs outputs/prob_cache

echo "=== AraSeg remaining gates ==="
echo "Started: $(date)"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 'no CUDA')"

if [ ! -f outputs/prob_cache/PA_dev_foldbag_e32.pkl ] ||
   [ ! -f outputs/prob_cache/PA_test_foldbag_e32.pkl ]; then
  python fold_bag.py \
    --config configs/e32_qwen35_9b_pa.yaml \
    --fold-output-pattern 'outputs/oof_ckpt/e32_fold{fold}' \
    --n-folds 5 --subtask PA \
    --output-dir outputs/candidate_foldbag_e32 \
    --cache-dir outputs/prob_cache --cache-tag foldbag_e32
else
  echo "[skip] e32 fold-bag caches already exist"
fi

python ensemble.py --subtask PA \
  --members configs/e25_xlmr_single_pa_stride256.yaml \
            configs/e32_qwen35_9b_pa.yaml \
            configs/e33_gemma4_12b_pa.yaml \
            configs/e73_qwen35_27b_pa.yaml \
            configs/e74_qwen25_14b_instruct_pa.yaml \
            configs/e80_qwen35_9b_pa_s1.yaml \
            configs/e84_gemma4_12b_pa_s1.yaml \
  --cache-members foldbag_e32 --cache-dir outputs/prob_cache \
  --combine logit --exhaustive --exhaustive_top_k 10 \
  --threshold_policy dev_tuned --thr_step 0.01 \
  --output_dir outputs/pa_candidate_logit_corrected

echo "=== PA candidate versus current lock ==="
python bootstrap_paired.py --subtask PA --split test \
  --a outputs/phaseC_v2_pa_logit/test_predictions_PA.csv \
  --b outputs/pa_candidate_logit_corrected/test_predictions_PA.csv

python cache_probs.py --subtask NoPnx-PA --splits dev test \
  --members configs/e17_xlmr_single_nopnxpa_s1.yaml \
            configs/e18_xlmr_single_nopnxpa_s2.yaml \
            configs/e75_naqta_restore_nopnxpa_comma_s1.yaml \
            configs/e76_naqta_restore_nopnxpa_comma_s2.yaml \
  --out_dir outputs/prob_cache

python ensemble.py --subtask NoPnx-PA \
  --cache-members e17 e18 --cache-dir outputs/prob_cache \
  --combine logit --threshold_policy dev_tuned --thr_step 0.01 \
  --output_dir outputs/nopnxpa_control_e17_e18_logit

python ensemble.py --subtask NoPnx-PA \
  --cache-members e75 e76 --cache-dir outputs/prob_cache \
  --combine logit --threshold_policy dev_tuned --thr_step 0.01 \
  --output_dir outputs/nopnxpa_restore_e75_e76_logit

echo "=== e75+e76 restoration versus matched e17+e18 controls ==="
python bootstrap_paired.py --subtask NoPnx-PA --split test \
  --a outputs/nopnxpa_control_e17_e18_logit/test_predictions_NoPnx_PA.csv \
  --b outputs/nopnxpa_restore_e75_e76_logit/test_predictions_NoPnx_PA.csv

echo "Finished: $(date)"
echo "Authorize e75/e76 OOF only if the matched test CI lower bound is above zero."
