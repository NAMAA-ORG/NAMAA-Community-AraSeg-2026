#!/bin/bash
# Lane B Task 2: the two PA scale/diversity probes.
#   e73 = Qwen3.5-27B, the next dense rung above the Qwen3.5-9B anchors -> isolates scale.
#   e74 = Qwen2.5-14B-Instruct, cross-generation diversity probe, judged on ensemble
#         delta rather than standalone F1.
#
# Request e73 on H100 -- it is the longest single job in the campaign, and 27B needs
# the most headroom of anything here (see docs/handoff.md memory ladder). e74 fits
# BF16 on an 80 GB A100. Both jobs are independent; a failure in one does not block
# the other. Idempotent: skips a job whose results.json already exists.
#
#   sbatch slurm/run_scale_pa_wave.sh   # submit from ${PROJECT}/experiments
set -uo pipefail

PROJECT=${ARASEG_PROJECT:?set ARASEG_PROJECT}
EXPERIMENTS_DIR=${PROJECT}/experiments
VENV=${PROJECT}/.venv-llm
ACCOUNT=kisski-futurmig

mkdir -p "${EXPERIMENTS_DIR}/logs"
cd "${EXPERIMENTS_DIR}"

jids=()

if [ -f outputs/e73/results.json ]; then
  echo "  [skip] e73 — outputs/e73/results.json already exists"
else
  out=$(sbatch <<'SBATCH_EOF'
#!/bin/bash
#SBATCH --job-name=araseg-e73-qwen35-27b-pa
#SBATCH -A kisski-futurmig
#SBATCH -p kisski-h100
#SBATCH -G H100:1
#SBATCH --mem-per-gpu=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments/logs/slurm_%j_%x.out
#SBATCH --error=${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments/logs/slurm_%j_%x.err
export HF_HOME=${ARASEG_HF_HOME:-$HOME/.cache/huggingface}
export HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
source ${ARASEG_PROJECT:?set ARASEG_PROJECT}/.venv-llm/bin/activate
cd ${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments
echo "=== e73 (Qwen3.5-27B PA) — started $(date) ==="
python train.py --config configs/e73_qwen35_27b_pa.yaml
echo "=== finished $(date) ==="
SBATCH_EOF
)
  jids+=("${out##* }")
  echo "  e73: submitted (${out##* })"
fi

if [ -f outputs/e74/results.json ]; then
  echo "  [skip] e74 — outputs/e74/results.json already exists"
else
  out=$(sbatch <<'SBATCH_EOF'
#!/bin/bash
#SBATCH --job-name=araseg-e74-qwen25-14b-pa
#SBATCH -A kisski-futurmig
#SBATCH -p kisski
#SBATCH -G A100:1
#SBATCH --mem-per-gpu=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments/logs/slurm_%j_%x.out
#SBATCH --error=${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments/logs/slurm_%j_%x.err
export HF_HOME=${ARASEG_HF_HOME:-$HOME/.cache/huggingface}
export HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
source ${ARASEG_PROJECT:?set ARASEG_PROJECT}/.venv-llm/bin/activate
cd ${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments
echo "=== e74 (Qwen2.5-14B-Instruct PA) — started $(date) ==="
python train.py --config configs/e74_qwen25_14b_instruct_pa.yaml
echo "=== finished $(date) ==="
SBATCH_EOF
)
  jids+=("${out##* }")
  echo "  e74: submitted (${out##* })"
fi

echo
echo "Submitted ${#jids[@]} job(s): ${jids[*]}"
cat <<'EOF'

--- next (CPU/inference, after both jobs finish) ---
1. cache_probs.py --subtask PA --splits dev test --members configs/e7{3,4}_*.yaml
2. ensemble.py --subtask PA --combine logit --exhaustive, PA lock members + candidate
3. bootstrap_paired.py the resulting test CSV against outputs/phaseC_v2_pa_logit/
   test_predictions_PA.csv
4. Advance only if the dev-selected ensemble's practice-test CI lower bound > 0.
   Standalone F1 improvement alone does not replace the stronger lock.
EOF
