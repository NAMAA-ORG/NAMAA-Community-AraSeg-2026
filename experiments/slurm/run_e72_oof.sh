#!/bin/bash
# e72 OOF folds (NoPnx-NP) — the GPU half of the NoPnx-NP decoder gate (bar 0.8649).
#
# Section (B) of run_e72_e71oof.sh did this for e71; e72 never got the same treatment,
# so gate_member.py dies on the missing outputs/oof/e72_fold0.json. Without these folds
# the ONLY way to gate e72 is --no-folds, which feeds e72's in-sample train predictions
# to a stacker whose other members have honest OOF rows. e72 was fine-tuned on train,
# so that biases the fit toward the candidate: a rejection under --no-folds would still
# mean something, an acceptance would not. Hence real folds.
#
# train.py applies naqta_restore BEFORE the fold filter, so each fold trains on restored
# text minus that fold's docs, and the OOF branch (train.py:955) maps restored-space
# probabilities back onto original NoPnx tokens -- the JSON lands in the same space as
# every other member's. Folds are seeded (42) and shared across members, so the decoder
# stays aligned.
#
# Submit from ${PROJECT}/experiments:
#   sbatch slurm/run_e72_oof.sh
#SBATCH --job-name=araseg-e72-oof
#SBATCH -A kisski-futurmig
#SBATCH -p kisski
#SBATCH -G A100:1
#SBATCH --mem-per-gpu=32G
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

source ${PROJECT}/.venv/bin/activate
cd ${PROJECT}/experiments
mkdir -p outputs/oof logs

echo "=== e72 OOF folds (NoPnx-NP) ==="
echo "Node: $(hostname)  |  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Started: $(date)"

python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 'no CUDA — login node?')" || exit 1

# The restored-text build reads this; e72's own training run created it. Fail loudly
# rather than silently re-deriving it under different settings.
[ -f outputs/prob_cache/NoPnx_NP_train_naqta_full.pkl ] || {
  echo "missing outputs/prob_cache/NoPnx_NP_train_naqta_full.pkl — run naqta_cache.py first"; exit 1; }

fail=0
for f in 0 1 2 3 4; do
  marker="outputs/oof/e72_fold${f}.json"
  if [ -f "$marker" ]; then echo "  [skip] $marker"; continue; fi
  echo "--- fold $f ---"
  t0=$SECONDS
  python train.py --config configs/e72_naqta_restore_nopnxnp.yaml \
    --holdout-fold "$f" --n-folds 5 --fold-seed 42 \
    --oof-out "$marker" \
    --output-dir "outputs/e72_fold${f}" \
    && echo "  [ok  ] $marker  ($(( (SECONDS-t0)/60 )) min)" \
    || { fail=1; echo "  [FAIL] fold $f"; }
done

echo ""
n=$(ls outputs/oof/e72_fold*.json 2>/dev/null | wc -l)
echo "e72 OOF folds present: ${n}/5"
echo "Finished: $(date)"

cat <<'EOF'

--- next (CPU, login node) ---
  python gate_member.py --subtask NoPnx-NP --candidate e72

  e70 is already a member of this lock, so it is part of the baseline -- the only
  question left on this head is whether e72 adds anything on top of it.

  Decide on the TEST CI, not the point estimate, and not on dev -- dev is
  threshold-tuned per arm and reads optimistic. e69 was +0.41 on test and still
  rejected; e71 just came back -0.15. Expect this to fail too.
EOF

exit $fail
