#!/bin/bash
# Lane A Task 5: equal-weight fold-bagging over existing OOF fold checkpoints.
# fold_bag.py averages the N per-fold checkpoints a candidate already has (from OOF
# fold generation) into one free test-time ensemble member -- no new GPU-hours beyond
# what OOF folding already spent. Idempotent: skips any family whose 5 fold checkpoints
# aren't all present, and skips any candidate whose output already exists.
#
# NOTE: fold checkpoint directories are NOT uniformly named across the codebase.
# e71 was folded by its own script (run_e72_e71oof.sh) into outputs/e71_fold<f>; every
# other family here went through run_oof_pa.sh/run_oof_stack.sh into
# outputs/oof_ckpt/<expid>_fold<f>. This driver knows both patterns per-candidate; it
# does not invent a third.
#
# Usage (login node, after OOF folds exist):
#   cd .../AraSeg/experiments && bash slurm/run_lane_a_recovery.sh
#SBATCH --job-name=araseg-lane-a-foldbag
#SBATCH -p kisski
#SBATCH -G A100:1
#SBATCH --mem-per-gpu=40G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments/logs/slurm_%j_%x.out
#SBATCH --error=${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments/logs/slurm_%j_%x.err
set -uo pipefail

PROJECT=${ARASEG_PROJECT:?set ARASEG_PROJECT}
export HF_HOME=${ARASEG_HF_HOME:-$HOME/.cache/huggingface}
export HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd ${PROJECT}/experiments
mkdir -p outputs/candidate_foldbag logs

N_FOLDS=5
fail=0

# expid : config : subtask : fold-output-pattern : venv
# e71 is XLM-R (.venv); e32/e40/e41/e42 are Qwen3.5-9B and need .venv-llm's newer
# transformers (plain .venv doesn't recognize the qwen3_5 architecture).
FAMILIES=(
  "e71:e71_naqta_restore_nopnxpa:NoPnx-PA:outputs/e71_fold{fold}:.venv"
  "e32:e32_qwen35_9b_pa:PA:outputs/oof_ckpt/e32_fold{fold}:.venv-llm"
  "e40:e40_qwen35_9b_np:NP:outputs/oof_ckpt/e40_fold{fold}:.venv-llm"
  "e41:e41_qwen35_9b_nopnx_pa:NoPnx-PA:outputs/oof_ckpt/e41_fold{fold}:.venv-llm"
  "e42:e42_qwen35_9b_nopnx_np:NoPnx-NP:outputs/oof_ckpt/e42_fold{fold}:.venv-llm"
)

echo "Node: $(hostname)  |  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Started: $(date)"

for entry in "${FAMILIES[@]}"; do
  IFS=: read -r expid cfg subtask pattern venv <<< "$entry"
  tag=${subtask//-/_}
  out_dir="outputs/candidate_foldbag_${expid}"
  marker="${out_dir}/fold_bag_summary.json"

  if [ -f "$marker" ]; then
    echo "[skip] ${expid} — ${marker} already exists"
    continue
  fi

  complete=1
  for f in $(seq 0 $((N_FOLDS - 1))); do
    fold_dir=$(echo "$pattern" | sed "s/{fold}/${f}/")
    ckpt="${fold_dir}/best_${tag}.pt"
    if [ ! -f "$ckpt" ]; then
      echo "[skip] ${expid} — missing fold ${f} checkpoint: ${ckpt}"
      complete=0
      break
    fi
  done
  [ "$complete" -eq 1 ] || continue

  echo "--- fold-bagging ${expid} (${subtask}, ${venv}) ---"
  source "${PROJECT}/${venv}/bin/activate"
  python fold_bag.py --config "configs/${cfg}.yaml" \
    --fold-output-pattern "$pattern" --n-folds "$N_FOLDS" \
    --subtask "$subtask" --output-dir "$out_dir" || { fail=1; echo "[FAIL] ${expid}"; }
done

echo "Finished: $(date)"
echo "Next: bootstrap_paired.py each outputs/candidate_foldbag_<id> against its matched"
echo "full-model or current-lock prediction CSV (see Lane A plan Task 5 Step 5)."
exit $fail
