#!/bin/bash
# Lane C Task 1: comma-corrected restoration retrains (e75/e76) + min-p comparison (e87).
#
# e75: comma-corrected min-p 0.3, seed 1 (naqta_comma ",")
# e76: same recipe, seed 2 (independent seed)
# e87: same comma correction, min-p 0.1 (isolates the min-p contrast from the comma)
#
# e75-vs-e71 measures the comma; e87-vs-e75 measures min-p. Neither run is evidence
# for both. Do not overwrite e71/e72 -- these are new IDs/output dirs.
#
# The Naqta 8-class prediction cache (outputs/prob_cache/NoPnx_PA_<split>_naqta_full.pkl)
# is comma-independent -- naqta_comma only changes which mark restore_doc() inserts for
# class 2, not the 8-class model's own predictions -- so it is built once and shared by
# e71/e75/e76/e87. Skipped here if e71's run already produced it.
#
#   sbatch slurm/run_restoration_retrain_wave.sh   # submit from ${PROJECT}/experiments
#SBATCH --job-name=araseg-restoration-retrain
#SBATCH -p kisski
#SBATCH -G A100:1
#SBATCH --mem-per-gpu=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments/logs/slurm_%j_%x.out
#SBATCH --error=${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments/logs/slurm_%j_%x.err
set -uo pipefail

PROJECT=${ARASEG_PROJECT:?set ARASEG_PROJECT}
export HF_HOME=${ARASEG_HF_HOME:-$HOME/.cache/huggingface}
export HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

source ${PROJECT}/.venv/bin/activate
cd ${PROJECT}/experiments
mkdir -p outputs/prob_cache logs

echo "Node: $(hostname)  |  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Started: $(date)"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 'no CUDA — login node?')" || exit 1

fail=0
run_one () {   # $1 = marker file (skip if present), rest = command
  local marker="$1"; shift
  if [ -f "$marker" ]; then echo "  [skip] $marker"; return 0; fi
  local t0=$SECONDS
  "$@" || { echo "  [FAIL] $marker"; return 1; }
  echo "  [ok  ] $marker  ($(( (SECONDS-t0)/60 )) min)"
}

echo "--- Naqta 8-class caches for NoPnx-PA (shared by e71/e75/e76/e87) ---"
for sp in train dev test; do
  if [ -f "outputs/prob_cache/NoPnx_PA_${sp}_naqta_full.pkl" ]; then
    echo "  [skip] NoPnx_PA_${sp}_naqta_full.pkl"
  else
    python naqta_cache.py --subtasks NoPnx-PA --splits $sp || fail=1
  fi
done

# expid : config basename (no .yaml) : min_p (must match the config's naqta_restore_min_p)
RUNS=(
  "e75:e75_naqta_restore_nopnxpa_comma_s1:0.3"
  "e76:e76_naqta_restore_nopnxpa_comma_s2:0.3"
  "e87:e87_naqta_restore_nopnxpa_minp01:0.1"
)

echo "--- train e75/e76/e87 ---"
for entry in "${RUNS[@]}"; do
  IFS=: read -r expid cfg _ <<< "$entry"
  run_one "outputs/${expid}/results.json" python train.py --config "configs/${cfg}.yaml" || fail=1
done

echo "--- e75/e76/e87 dev threshold sweep + test map-back ---"
for entry in "${RUNS[@]}"; do
  IFS=: read -r expid cfg minp <<< "$entry"
  echo "  ${expid} dev"
  python restore_then_segment.py --subtask NoPnx-PA --split dev --mode naqta \
    --trained-member "configs/${cfg}.yaml" --min-punct-p "$minp" || fail=1
  echo "  ${expid} test"
  python restore_then_segment.py --subtask NoPnx-PA --split test --mode naqta \
    --trained-member "configs/${cfg}.yaml" --min-punct-p "$minp" || fail=1
done

echo ""
echo "Finished: $(date)"
cat <<'EOF'

--- how to read this ---
1. Dev-select min-p 0.1 vs 0.3 using e75 vs e87 test map-back F1 above.
2. Evaluate e76 only as the seed check for whichever recipe wins (1).
3. Equal-average the two winning-recipe seeds (ensemble.py --combine logit) before
   authorizing any new OOF folds.
4. Bootstrap that average against the current NoPnx-PA lock (bootstrap_paired.py).
   Generate OOF only if the paired CI lower bound is positive (gate_member.py after).
EOF
exit $fail
