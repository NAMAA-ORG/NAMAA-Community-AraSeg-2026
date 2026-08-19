#!/bin/bash
# Lane B Task 4: second seeds for proven anchors. Each job is byte-identical to its
# recipe (e32/e40/e41/e42/e33/e70) except experiment_id, output_dir, and seed. All six
# are independent -- a failed seed does not block the others.
#
#   sbatch slurm/run_anchor_seed_wave.sh   # submit from ${PROJECT}/experiments
set -uo pipefail

PROJECT=${ARASEG_PROJECT:?set ARASEG_PROJECT}
EXPERIMENTS_DIR=${PROJECT}/experiments
ACCOUNT=kisski-futurmig

mkdir -p "${EXPERIMENTS_DIR}/logs"
cd "${EXPERIMENTS_DIR}"

# expid : config basename : venv (.venv for XLM-R/Naqta, .venv-llm for Qwen/Gemma) : time
JOBS=(
  "e80:e80_qwen35_9b_pa_s1:.venv-llm:10:00:00"
  "e81:e81_qwen35_9b_np_s1:.venv-llm:10:00:00"
  "e82:e82_qwen35_9b_nopnx_pa_s1:.venv-llm:10:00:00"
  "e83:e83_qwen35_9b_nopnx_np_s1:.venv-llm:10:00:00"
  "e84:e84_gemma4_12b_pa_s1:.venv-llm:10:00:00"
  "e85:e85_naqta_nopnxnp_s2:.venv:04:00:00"
)

jids=()
for entry in "${JOBS[@]}"; do
  IFS=: read -r expid cfg venv tlim <<< "$entry"
  if [ -f "outputs/${expid}/results.json" ]; then
    echo "  [skip] ${expid} — outputs/${expid}/results.json already exists"
    continue
  fi
  out=$(sbatch <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=araseg-${expid}-anchor-seed
#SBATCH -A ${ACCOUNT}
#SBATCH -p kisski
#SBATCH -G A100:1
#SBATCH --mem-per-gpu=40G
#SBATCH --cpus-per-task=8
#SBATCH --time=${tlim}
#SBATCH --output=${EXPERIMENTS_DIR}/logs/slurm_%j_%x.out
#SBATCH --error=${EXPERIMENTS_DIR}/logs/slurm_%j_%x.err
export HF_HOME=${ARASEG_HF_HOME:-$HOME/.cache/huggingface}
export HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
source ${PROJECT}/${venv}/bin/activate
cd ${EXPERIMENTS_DIR}
echo "=== ${expid} anchor seed — started \$(date) ==="
python train.py --config configs/${cfg}.yaml
echo "=== finished \$(date) ==="
SBATCH_EOF
)
  jids+=("${out##* }")
  echo "  ${expid}: submitted (${out##* })"
done

echo
echo "Submitted ${#jids[@]} job(s): ${jids[*]}"
cat <<'EOF'

--- next (CPU/inference, per completed pair) ---
1. ensemble.py --combine logit, equal-weight the anchor + its new seed; dev selects
   the threshold.
2. bootstrap_paired.py the pair average against the stronger existing single or
   current lock on practice test.
3. Only after that clears the paired gate, generate the new seed's five OOF folds
   and test decoder/stack integration.
4. Record a rejected seed as a negative result; do not tune a third seed from
   practice-test behavior.
EOF
