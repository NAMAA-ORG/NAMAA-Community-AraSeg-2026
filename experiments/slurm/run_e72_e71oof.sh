#!/bin/bash
# Two jobs, one submit:
#   (A) e72 — e71's Naqta-restore recipe on NoPnx-NP. Control = e19 (0.8231 @0.50).
#       Replication check: e71 beat e17 by +0.64 on NoPnx-PA, on ONE head. The e69/e70
#       asymmetry is the standing warning against reading one head as a result.
#   (B) e71 OOF folds — the GPU half of the NoPnx-PA decoder gate (bar 0.8718).
#
#   sbatch slurm/run_e72_e71oof.sh      # submit from ${PROJECT}/experiments, not the
#                                       # repo root: --output is relative to the submit dir
#
# NOT DONE HERE — e71/e72 prob caches. Their probs live in RESTORED token space, which
# has more tokens than NoPnx space, so they cannot enter a decoder until map_back gains
# a probability path (restore_then_segment.py:89 is binary-only: `if y and owner[k]>=0`).
# The folds below don't need it; caching does. Write that, then run cache_probs.py.
#SBATCH --job-name=araseg-e72-e71oof
#SBATCH -p kisski
#SBATCH -G A100:1
#SBATCH --mem-per-gpu=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
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
mkdir -p outputs/oof logs

echo "=== e72 (restore on NoPnx-NP) + e71 OOF folds (NoPnx-PA) ==="
echo "Node: $(hostname)  |  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Started: $(date)"

python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 'no CUDA — login node?')" || exit 1

MINP=0.3   # must match naqta_restore_min_p in BOTH configs; the eval reuses it
fail=0

run_one () {   # $1 = marker file (skip if present), rest = command
  local marker="$1"; shift
  if [ -f "$marker" ]; then echo "  [skip] $marker"; return 0; fi
  local t0=$SECONDS
  "$@" || { echo "  [FAIL] $marker"; return 1; }
  echo "  [ok  ] $marker  ($(( (SECONDS-t0)/60 )) min)"
}

# ── (A) e72 ───────────────────────────────────────────────────────────────────
# NoPnx-NP naqta _full caches: dev/test may exist from the zero-shot probe, train
# almost certainly does not (the e71 run had to build NoPnx-PA's). naqta_cache.py
# skips what is already there unless --overwrite.
echo "--- (A1) naqta caches for NoPnx-NP ---"
for sp in train dev test; do
  if [ -f outputs/prob_cache/NoPnx_NP_${sp}_naqta_full.pkl ]; then
    echo "  [skip] NoPnx_NP_${sp}_naqta_full.pkl"
  else
    python naqta_cache.py --subtasks NoPnx-NP --splits $sp || fail=1
  fi
done

echo "--- (A2) train e72 ---"
run_one "outputs/e72/results.json" python train.py --config configs/e72_naqta_restore_nopnxnp.yaml || fail=1

# Dev-tuned threshold, applied once to test, scored after map-back to NoPnx positions.
# --mode is argparse-required but ignored on the --trained-member path; pass naqta.
echo "--- (A3) e72 dev threshold sweep ---"
python restore_then_segment.py --subtask NoPnx-NP --split dev --mode naqta \
  --trained-member configs/e72_naqta_restore_nopnxnp.yaml --min-punct-p $MINP || fail=1

echo "--- (A4) e72 test at the dev-tuned threshold ---"
python restore_then_segment.py --subtask NoPnx-NP --split test --mode naqta \
  --trained-member configs/e72_naqta_restore_nopnxnp.yaml --min-punct-p $MINP || fail=1

# ── (B) e71 OOF folds ─────────────────────────────────────────────────────────
# train.py applies naqta_restore (:595) BEFORE the fold filter (:611), so each fold
# trains on restored text minus that fold's docs. Folds are seeded (42) and shared
# with every other member, so the decoder stays aligned.
echo "--- (B) e71 OOF folds (NoPnx-PA) ---"
for f in 0 1 2 3 4; do
  echo "  fold $f"
  run_one "outputs/oof/e71_fold${f}.json" \
    python train.py --config configs/e71_naqta_restore_nopnxpa.yaml \
      --holdout-fold "$f" --n-folds 5 --fold-seed 42 \
      --oof-out "outputs/oof/e71_fold${f}.json" \
      --output-dir "outputs/e71_fold${f}" || fail=1
done

# ── summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=== summary ==="
python - <<'PY'
import json
from pathlib import Path
r = Path("outputs/e72/results.json")
if r.exists():
    t = json.load(open(r)).get("test") or {}
    print(f"e72 NoPnx-NP single, restored-space test F1: {t.get('f1')}")
    print("  (NOT the number that counts -- use the map-back F1 from (A4) above)")
else:
    print("e72: NO results.json")
n = len(list(Path("outputs/oof").glob("e71_fold*.json")))
print(f"e71 OOF folds present: {n}/5")
PY

cat <<'EOF'

--- how to read this ---
(A) e72 vs its control e19 TEST = 0.8211.  e71 vs e17 was +1.06 on NoPnx-PA.
    Replicates  -> restoration is a real recipe, worth the min_p sweep and a seed clone.
    Does not    -> e71's +1.06 is one head, and goes in the paper as such.
    Either way it is a result. Two comparisons to refuse: the 0.8589 NoPnx-NP LOCK
    (one model vs a six-member decoder, the mistake that nearly buried e71), and
    e19's DEV 0.8231 (test-vs-dev, the mistake that understated e71 by 0.4 F1).

(B) folds are only the GPU half. Still needed before gate_member.py:
      1. probability-space map_back (max over owned positions, not OR)
      2. cache_probs.py + the OOF dump made restore-aware
      3. gate_member.py --subtask NoPnx-PA --candidate e71   (CPU, off-cluster)
    Decide on the TEST CI, not the point estimate. e69 was +0.41 and still rejected.
EOF

echo "Finished: $(date)"
exit $fail
