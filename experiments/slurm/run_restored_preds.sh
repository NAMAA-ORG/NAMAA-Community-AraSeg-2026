#!/bin/bash
# Dump the MAPPED-BACK test predictions for e71 (NoPnx-PA) and e72 (NoPnx-NP), so the
# +1.06 / +0.12 point estimates can finally get a confidence interval.
#   sbatch experiments/slurm/run_restored_preds.sh
#
# No training. Both models are already trained and their dev-tuned thresholds are
# already persisted (outputs/e7{1,2}/restore_thr.txt, 0.50 and 0.55), so this just
# re-runs the forward pass and writes CSVs -- a few minutes each.
#
# Why it is needed: restore_then_segment.py --trained-member used to print an F1 and
# write nothing, so there was no per-document output for bootstrap_paired.py to
# resample. With ~±1.5 F1 rerun noise on these heads (ledger §5), a single-run delta
# is not signal -- e71's +1.06 and e72's +0.12 are both inside that band until a CI
# says otherwise.
#
# The matched CONTROLS (e17, e19) need no GPU: their probabilities are already cached,
# so `cache_to_csv.py` regenerates their CSVs off-cluster. Only these two need a GPU.
#SBATCH --job-name=araseg-restored-preds
#SBATCH -p kisski
#SBATCH -G A100:1
#SBATCH --mem-per-gpu=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm_%j_%x.out
#SBATCH --error=logs/slurm_%j_%x.err
set -uo pipefail

PROJECT=${ARASEG_PROJECT:?set ARASEG_PROJECT}
export HF_HOME=${ARASEG_HF_HOME:-$HOME/.cache/huggingface}
export HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

source ${PROJECT}/.venv/bin/activate
cd ${PROJECT}/experiments
mkdir -p logs

echo "=== mapped-back test predictions for e71 / e72 ==="
echo "Node: $(hostname)  |  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Started: $(date)"

python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 'no CUDA — login node?')" || exit 1

# The whole job is pointless against the pre-2026-07-24 script: it writes no CSV and
# prints the wrong control. Fail loudly here rather than burn an A100 for nothing.
if ! grep -q "mapped-back predictions" restore_then_segment.py; then
  echo "FATAL: restore_then_segment.py on the cluster is the PRE-FIX version." >&2
  echo "  It writes no predictions CSV, so this job would produce nothing usable." >&2
  echo "  scp the updated file from the repo, then resubmit:" >&2
  echo "  scp experiments/restore_then_segment.py kisski:${PROJECT}/experiments/" >&2
  exit 1
fi

MINP=0.3   # must match naqta_restore_min_p in both configs

# Idempotent: skip a head whose CSV is already there, so a resubmit resumes.
run_head () {                       # $1=exp  $2=subtask  $3=config
  local out="outputs/$1/restored_test_predictions_${2//-/_}.csv"
  echo "--- $1 ($2) ---"
  if [ -f "$out" ]; then
    echo "  [skip] $out"
    return 0
  fi
  if [ ! -f "outputs/$1/restore_thr.txt" ]; then
    echo "  FATAL: no outputs/$1/restore_thr.txt — the dev sweep never ran for $1." >&2
    echo "  Do NOT substitute a threshold here; it must be the dev-tuned one." >&2
    return 1
  fi
  echo "  dev-tuned threshold: $(cat outputs/$1/restore_thr.txt)"
  python restore_then_segment.py --subtask "$2" --split test --mode naqta \
    --trained-member "$3" --min-punct-p $MINP || return 1
}

rc=0
run_head e71 NoPnx-PA configs/e71_naqta_restore_nopnxpa.yaml || rc=1
run_head e72 NoPnx-NP configs/e72_naqta_restore_nopnxnp.yaml || rc=1

echo ""
echo "=== what to pull ==="
ls -la outputs/e71/restored_test_predictions_NoPnx_PA.csv \
       outputs/e72/restored_test_predictions_NoPnx_NP.csv 2>&1

cat <<'EOF'

--- then, OFF-CLUSTER (CPU, seconds) ---
The controls need no GPU -- regenerate them from the cached probabilities, verified
against ledger §2.0 (e17 0.8147, e19 0.8211):

  python cache_to_csv.py --selfcheck

Then the gate itself (bootstrap_paired.py now runs without torch):

  python bootstrap_paired.py --subtask NoPnx-PA --split test \
    --a outputs/e17/test_predictions_NoPnx_PA.csv \
    --b outputs/e71/restored_test_predictions_NoPnx_PA.csv

  python bootstrap_paired.py --subtask NoPnx-NP --split test \
    --a outputs/e19/test_predictions_NoPnx_NP.csv \
    --b outputs/e72/restored_test_predictions_NoPnx_NP.csv

--- how to read it ---
Decide on the CI, not the point estimate. e69 was +0.41 and still rejected; e66 was
a real +0.19 on test and still rejected. If e71's CI spans zero, then "training on
restored text works" is not a finding and the paper must say so -- §4.2 and appendix
negative (8) are already written to survive that outcome.
EOF
echo "Finished: $(date)"
exit $rc
