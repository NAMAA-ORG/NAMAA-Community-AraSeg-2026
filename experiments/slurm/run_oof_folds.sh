#!/bin/bash
# OOF folds for an arbitrary experiment — the GPU half of any decoder-member gate.
#
#   EXP=e85 CONFIG=configs/e85_naqta_nopnxnp_s2.yaml SUBTASK=NoPnx-NP \
#     sbatch --export=ALL,EXP,CONFIG,SUBTASK slurm/run_oof_folds.sh
#
# Generalises run_e72_oof.sh / run_e75_e76_oof.sh, which were copies of each other
# differing only in the config. Without folds the ONLY way to gate a candidate is
# --no-folds, which feeds a fine-tuned model's IN-SAMPLE train predictions to a stacker
# whose other members have honest OOF rows -- biased toward the candidate, so a
# rejection would still mean something but an acceptance would not. gate_member.py
# refuses that path outright now (FROZEN allowlist), which is what sent you here.
#
# train.py applies naqta_restore, when the config sets it, BEFORE the fold filter, so
# each fold trains on restored text minus that fold's docs and the OOF branch maps
# restored-space probabilities back onto original tokens. Folds are seeded (42) and
# shared across members, so every member lands in the same space.
#
# Dev/test caches are built too: gate_member.py needs all three and the fold
# checkpoints are not the same model as the full-data one it caches from.
#SBATCH --job-name=araseg-oof
#SBATCH -A kisski-futurmig
#SBATCH -p kisski
#SBATCH -G A100:1
#SBATCH --mem-per-gpu=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments/logs/slurm_%j_%x.out
#SBATCH --error=${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments/logs/slurm_%j_%x.err
set -uo pipefail

PROJECT=${ARASEG_PROJECT:?set ARASEG_PROJECT}
export HF_HOME=${ARASEG_HF_HOME:-$HOME/.cache/huggingface}
export HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

: "${EXP:?set EXP, e.g. e85}"
: "${CONFIG:?set CONFIG, e.g. configs/e85_naqta_nopnxnp_s2.yaml}"
: "${SUBTASK:?set SUBTASK, e.g. NoPnx-NP}"

source ${PROJECT}/.venv/bin/activate
cd ${PROJECT}/experiments
mkdir -p outputs/oof outputs/prob_cache logs

echo "=== OOF folds: ${EXP} (${CONFIG}) on ${SUBTASK} ==="
echo "Node: $(hostname)  |  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Started: $(date)"

python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 'no CUDA — login node?')" || exit 1
[ -f "$CONFIG" ] || { echo "no such config: $CONFIG"; exit 1; }

fail=0
for f in 0 1 2 3 4; do
  marker="outputs/oof/${EXP}_fold${f}.json"
  if [ -f "$marker" ]; then echo "  [skip] $marker"; continue; fi
  t0=$SECONDS
  python train.py --config "$CONFIG" \
    --holdout-fold "$f" --n-folds 5 --fold-seed 42 \
    --oof-out "$marker" --output-dir "outputs/${EXP}_fold${f}" \
    && echo "  [ok  ] $marker  ($(( (SECONDS-t0)/60 )) min)" \
    || { fail=1; echo "  [FAIL] fold $f"; }
done

# The full-data model, which is what dev/test caches come from -- a different
# checkpoint from any of the folds above.
if compgen -G "outputs/${EXP}/*.pt" > /dev/null; then
  echo "  [skip] outputs/${EXP} already has a checkpoint"
else
  python train.py --config "$CONFIG" --output-dir "outputs/${EXP}" \
    || { fail=1; echo "  [FAIL] full-data run"; }
fi

python cache_probs.py --subtask "$SUBTASK" --splits dev test \
  --members "$CONFIG" --out_dir outputs/prob_cache || fail=1

echo ""
echo "${EXP} OOF folds present: $(ls outputs/oof/${EXP}_fold*.json 2>/dev/null | wc -l)/5"
echo "Finished: $(date)   (fail=${fail})"
echo ""
echo "--- next (CPU, login node) ---"
echo "  python gate_member.py --subtask ${SUBTASK} --candidate ${EXP} <other candidates>"
echo "  Sweep candidates TOGETHER, not one at a time -- on 2026-08-01 e69 (+0.41),"
echo "  e75 (+0.10) and e76 (+0.19) were each individually below the bar, and"
echo "  e69+e76 together cleared it at +0.64, CI [+0.15,+1.16]."

exit $fail
