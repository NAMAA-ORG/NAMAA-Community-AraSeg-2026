#!/bin/bash
# Blind probability caches for an arbitrary member list — the one GPU step between a
# gate passing and a submission. Generic on purpose: run_e75_e76_oof.sh hardcoded the
# Lane C pair, but the gate on 2026-08-01 selected +e69+e76, and e69 was not in it.
#
#   MEMBERS="configs/e69_naqta_nopnxpa.yaml configs/e76_naqta_restore_nopnxpa_comma_s2.yaml" \
#     sbatch --export=ALL,MEMBERS slurm/run_blind_cache.sh
#
# Short by design: inference over ~200 blind docs per member, no training. Small jobs
# backfill, which matters more here than anything else — cutting run_e75_e76_oof.sh from
# 4h to 2h30 did NOT improve its queue position, but that was a resubmit that reset queue
# age, so the walltime effect was never actually isolated. This one is genuinely small.
#
# cache_probs.py detects naqta_restore per config, so a restored member (e76) and a plain
# one (e69) can go in the same call. Restored members additionally need
# NoPnx_PA_blind_naqta_full.pkl, which job 15117536 already built before it died.
# Implementation note: ANY gpu, 16G, 4 cpus. This is inference with a 560M encoder over ~200
# documents -- it does not need an A100, and asking for one by name puts a 10-minute
# job in the queue behind every training run on the cluster. The first version asked
# `-G A100:1 --mem-per-gpu=32G` and was quoted a 30-hour start. If it still will not
# schedule, it runs on CPU: see the device note below.
#SBATCH --job-name=araseg-blind-cache
#SBATCH -A kisski-futurmig
#SBATCH -p kisski
#SBATCH -G 1
#SBATCH --mem-per-gpu=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments/logs/slurm_%j_%x.out
#SBATCH --error=${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments/logs/slurm_%j_%x.err
set -uo pipefail

PROJECT=${ARASEG_PROJECT:?set ARASEG_PROJECT}
export HF_HOME=${ARASEG_HF_HOME:-$HOME/.cache/huggingface}
export HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

SUBTASK="${SUBTASK:-NoPnx-PA}"
: "${MEMBERS:?set MEMBERS to a space-separated list of config YAMLs}"

source ${PROJECT}/.venv/bin/activate
cd ${PROJECT}/experiments
mkdir -p outputs/prob_cache logs

echo "=== blind caches: ${SUBTASK} ==="
echo "members: ${MEMBERS}"
echo "Node: $(hostname)  |  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Started: $(date)"

# NOT fatal without a GPU. cache_probs.py already falls back to CPU on its own, and
# for inference at this size that is a slowdown, not a blocker -- which makes a CPU
# partition (usually far shorter queues) a legitimate way to run this:
#   sbatch -p <cpu_partition> -G 0 --time=02:00:00 ...
python -c "
import torch
print('device:', 'cuda ' + torch.cuda.get_device_name(0) if torch.cuda.is_available()
      else 'CPU — slower but correct; fine for inference at this size')"

python cache_probs.py --subtask "$SUBTASK" --splits blind \
  --members $MEMBERS --out_dir outputs/prob_cache || exit 1

# The decoder INTERSECTS doc ids across members, so a cache covering a different doc set
# silently shrinks the submission instead of erroring. Compare against a locked member
# rather than a hardcoded count — the first version of this check asserted the runbook's
# 100 and aborted on the real 212.
python - <<PY || exit 1
import pickle, sys
from pathlib import Path
st = "${SUBTASK}".replace("-", "_")
ref = Path(f"outputs/prob_cache/{st}_blind_e41.pkl")
if not ref.exists():
    sys.exit(f"no reference member cache at {ref}")
b = pickle.load(open(ref, "rb"))
bad = 0
for p in sorted(Path("outputs/prob_cache").glob(f"{st}_blind_*.pkl")):
    a = pickle.load(open(p, "rb"))
    ok = set(a) == set(b)
    print(f"  {p.name:34} {len(a):4d} docs  {'OK' if ok else 'MISMATCH'}")
    bad |= (not ok)
sys.exit(bad)
PY

echo "Finished: $(date)"
