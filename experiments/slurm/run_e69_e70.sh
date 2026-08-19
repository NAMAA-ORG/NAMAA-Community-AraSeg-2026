#!/bin/bash
# e69/e70 — Naqta's encoder fine-tuned on AraSeg train as NoPnx members.
# 2 full trains + 10 OOF folds. Runs BOTH ways:
#   interactive A100 session:  bash experiments/slurm/run_e69_e70.sh
#   batch:                     sbatch experiments/slurm/run_e69_e70.sh
#
# PREREQUISITE — run ONCE on the LOGIN node (compute nodes are offline):
#   export HF_HOME=${ARASEG_HF_HOME:-$HOME/.cache/huggingface}
#   python -c "from huggingface_hub import snapshot_download as d; d('MostafaMaroof/Naqta')"
# The script hard-fails with that command if the weights are missing, rather than
# dying 10 minutes in on an offline-cache error.
#SBATCH --job-name=araseg-e69e70
#SBATCH -p kisski
#SBATCH -G A100:1
#SBATCH --mem-per-gpu=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
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

echo "=== AraSeg e69/e70: Naqta encoder fine-tuned on AraSeg train ==="
echo "Node: $(hostname)  |  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "Started: $(date)"

# Fail fast and loud if the login-node prefetch was skipped.
python - <<'PY' || exit 1
import sys
from pathlib import Path
import huggingface_hub as hh
try:
    p = hh.snapshot_download("MostafaMaroof/Naqta", local_files_only=True)
except Exception as e:
    sys.exit("\nNaqta weights are NOT in HF_HOME and compute nodes are offline.\n"
             "Run this ONCE on the LOGIN node, then resubmit:\n"
             "  export HF_HOME=${ARASEG_HF_HOME:-$HOME/.cache/huggingface}\n"
             "  python -c \"from huggingface_hub import snapshot_download as d; d('MostafaMaroof/Naqta')\"\n"
             f"({type(e).__name__}: {e})")
if not any(Path(p).glob("*.safetensors")) and not any(Path(p).glob("*.bin")):
    sys.exit(f"\nNaqta snapshot at {p} has config but NO weights (metadata-only download).\n"
             "Re-run the login-node snapshot_download above.")
print("Naqta weights present:", p)
PY

# GPU sanity — login nodes have no GPU, and a silent CPU fallback would run for hours.
python -c "
import torch, sys
if not torch.cuda.is_available():
    sys.exit('CUDA not available. Are you on a compute node? (login nodes have no GPU)')
print('torch', torch.__version__, '| cuda', torch.version.cuda, '|', torch.cuda.get_device_name(0))
" || exit 1

# Configs carry batch_size 4 + grad_accum 2 + gradient_checkpointing from the Kaggle
# T4 fix. Kept as-is on the A100: effective batch is still 8 (e17/e19 parity) and
# checkpointing is numerically exact (recompute, identical gradients), so KISSKI and
# Kaggle runs stay comparable. Costs ~30% wall-clock on a card that has it to spare.
run_one () {   # $1 = marker file, rest = command
  local marker="$1"; shift
  if [ -f "$marker" ]; then echo "  [skip] $marker"; return 0; fi
  local t0=$SECONDS
  "$@" || { echo "  [FAIL] $marker"; return 1; }
  echo "  [ok  ] $marker  ($(( (SECONDS-t0)/60 )) min)"
}

fail=0
for spec in "e69:configs/e69_naqta_nopnxpa.yaml" "e70:configs/e70_naqta_nopnxnp.yaml"; do
  eid=${spec%%:*}; cfg=${spec#*:}
  echo "--- ${eid} full train ---"
  run_one "outputs/${eid}/results.json" python train.py --config "$cfg" || fail=1
  for f in 0 1 2 3 4; do
    echo "--- ${eid} OOF fold ${f} ---"
    run_one "outputs/oof/${eid}_fold${f}.json" \
      python train.py --config "$cfg" \
        --holdout-fold "$f" --n-folds 5 --fold-seed 42 \
        --oof-out "outputs/oof/${eid}_fold${f}.json" \
        --output-dir "outputs/${eid}_fold${f}" || fail=1
  done
done

# dev/test prob caches, so the decoder fit + bootstrap gate stay CPU-only off-cluster.
echo "--- prob caches ---"
python cache_probs.py --subtask NoPnx-PA --members configs/e69_naqta_nopnxpa.yaml \
  --splits dev test --out_dir outputs/prob_cache || fail=1
python cache_probs.py --subtask NoPnx-NP --members configs/e70_naqta_nopnxnp.yaml \
  --splits dev test --out_dir outputs/prob_cache || fail=1

echo "=== summary ==="
python - <<'PY'
import json
from pathlib import Path
for eid, sub, bar in (("e69", "NoPnx-PA", 0.8718), ("e70", "NoPnx-NP", 0.8589)):
    fp = Path(f"outputs/{eid}/results.json")
    if not fp.exists():
        print(f"{eid} {sub}: NO results.json"); continue
    r = json.load(open(fp))
    t = r.get("test") or {}
    f1 = t.get("f1")
    print(f"{eid} {sub}: test F1 {f1}  (member-level; lock bar for the ENSEMBLE is {bar})")
    n = len(list(Path("outputs/oof").glob(f"{eid}_fold*.json")))
    print(f"     OOF folds present: {n}/5")
PY

echo "Finished: $(date)"
exit $fail
