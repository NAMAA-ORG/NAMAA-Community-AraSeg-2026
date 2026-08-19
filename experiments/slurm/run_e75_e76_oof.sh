#!/bin/bash
# e75 + e76: ALL remaining GPU work for the Lane C re-gate, in ONE allocation.
#
# Lane C was closed on 2026-07-30 because the standalone e75+e76 delta was +0.77 with
# CI [-0.05, +1.57] and the pre-registered rule demanded a strictly positive lower
# bound. Under the revised rule (select on the point estimate / P(B>A), since the gate
# is calibrated at the point estimate — e70 predicted +0.60 and blind returned +0.6)
# that is a lever we should be taking. Gating it as a DECODER MEMBER is also a different
# question from the standalone comparison that was rejected: it asks what e75/e76 add on
# top of the six-member lock, and lets dev choose the subset.
#
# WHY ONE JOB. The obvious split — folds now, blind caches after the gate passes — puts
# two queue waits on the critical path. On 2026-08-01 the queue was quoting a 33-hour
# start against an Aug 3 AoE deadline, so a second wait does not fit. Everything the
# lever can possibly need is therefore computed up front. If the gate then rejects, the
# wasted cost is ~20 min of blind inference; if it passes, every remaining step is CPU
# and can run immediately on the login node.
#
# Every stage skips work that already exists, so this is safe to resubmit after a
# timeout or a partial failure — it resumes rather than restarting.
#
# Submit from ${PROJECT}/experiments:
#   sbatch slurm/run_e75_e76_oof.sh
#SBATCH --job-name=araseg-e75e76
#SBATCH -A kisski-futurmig
#SBATCH -p kisski
#SBATCH -G A100:1
#SBATCH --mem-per-gpu=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:30:00
#SBATCH --output=${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments/logs/slurm_%j_%x.out
#SBATCH --error=${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments/logs/slurm_%j_%x.err
set -uo pipefail

# Implementation note: 2h30 not 4h. The work is ~1h40 (10 XLM-R-large folds + two inference passes)
# and walltime is what the backfill scheduler prices. The 4h ask is what bought a 33h
# queue position. Raise it only if a fold actually times out.

PROJECT=${ARASEG_PROJECT:?set ARASEG_PROJECT}
export HF_HOME=${ARASEG_HF_HOME:-$HOME/.cache/huggingface}
export HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

source ${PROJECT}/.venv/bin/activate
cd ${PROJECT}/experiments
mkdir -p outputs/oof outputs/prob_cache logs

echo "=== e75 + e76: OOF folds + dev/test/blind caches (NoPnx-PA) ==="
echo "Node: $(hostname)  |  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Started: $(date)"

python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 'no CUDA — login node?')" || exit 1

echo ""
echo "--- inventory before ---"
for f in outputs/oof/e75_fold*.json outputs/oof/e76_fold*.json \
         outputs/prob_cache/NoPnx_PA_*_naqta_full.pkl \
         outputs/prob_cache/NoPnx_PA_*_e7[56].pkl; do
  [ -e "$f" ] && echo "  have $f"
done
echo "--- end inventory ---"

# ── (0) Naqta 8-class cache for the BLIND split ──────────────────────────────
# build_restored_docs (used by both training and restore-aware caching) reads
# NoPnx_PA_<split>_naqta_full.pkl. train/dev/test exist from the e71 wave; blind does
# not, and without it stage (2) dies AFTER the folds have already been paid for.
#
# The blind repos are GATED. If this fails on an offline-cache miss, run ONCE on the
# LOGIN node, then resubmit:
#   export HF_HOME=${ARASEG_HF_HOME:-$HOME/.cache/huggingface}
#   export HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets
#   python -c "
#   import os, datasets
#   datasets.load_dataset('MBZUAI/AraSeg-2026-Shared-Task-NoPnx-PA-Blind',
#                         token=os.environ['ARASEG_BLIND_TOKEN'])"
echo ""
echo "=== (0) Naqta blind cache ==="
if [ -f outputs/prob_cache/NoPnx_PA_blind_naqta_full.pkl ]; then
  echo "  [skip] NoPnx_PA_blind_naqta_full.pkl exists"
else
  python naqta_cache.py --subtasks NoPnx-PA --splits blind \
    --out_dir outputs/prob_cache || {
      echo "FATAL: naqta blind cache failed — see the gated-repo prefetch note above."
      echo "Stopping BEFORE the folds so the allocation is not spent on work that"
      echo "cannot be finished in this job."; exit 1; }
fi

# ── (1) OOF folds ────────────────────────────────────────────────────────────
# train.py applies naqta_restore BEFORE the fold filter, so each fold trains on restored
# text minus that fold's docs, and the OOF branch maps restored-space probabilities back
# onto original NoPnx tokens — the JSON lands in the same space as every other member's.
# Folds are seeded (42) and shared across members, so the decoder stays aligned.
echo ""
echo "=== (1) OOF folds ==="
fail=0
for exp in e75 e76; do
  case $exp in
    e75) cfg=configs/e75_naqta_restore_nopnxpa_comma_s1.yaml ;;
    e76) cfg=configs/e76_naqta_restore_nopnxpa_comma_s2.yaml ;;
  esac
  echo "--- ${exp} (${cfg}) ---"
  for f in 0 1 2 3 4; do
    marker="outputs/oof/${exp}_fold${f}.json"
    if [ -f "$marker" ]; then echo "  [skip] $marker"; continue; fi
    t0=$SECONDS
    python train.py --config "$cfg" \
      --holdout-fold "$f" --n-folds 5 --fold-seed 42 \
      --oof-out "$marker" \
      --output-dir "outputs/${exp}_fold${f}" \
      && echo "  [ok  ] $marker  ($(( (SECONDS-t0)/60 )) min)" \
      || { fail=1; echo "  [FAIL] ${exp} fold $f"; }
  done
done

# ── (2) dev / test / blind probability caches ────────────────────────────────
# cache_probs.py detects naqta_restore from the config and does the restore-aware path
# itself, and it skips any cache that already exists.
echo ""
echo "=== (2) dev/test/blind caches ==="
python cache_probs.py --subtask NoPnx-PA --splits dev test blind \
  --members configs/e75_naqta_restore_nopnxpa_comma_s1.yaml \
            configs/e76_naqta_restore_nopnxpa_comma_s2.yaml \
  --out_dir outputs/prob_cache || fail=1

# ── (3) doc-id check against a locked member ─────────────────────────────────
# The decoder INTERSECTS doc ids across members, so a blind cache covering a different
# doc set silently shrinks the submission instead of erroring. This is the check that
# caught the 100-vs-212 assumption on e70.
echo ""
echo "=== (3) blind doc-id check ==="
python - <<'PY' || fail=1
import pickle, sys
from pathlib import Path
ref = Path("outputs/prob_cache/NoPnx_PA_blind_e41.pkl")
if not ref.exists():
    sys.exit(f"no reference member cache at {ref} — cannot verify blind doc ids")
b = pickle.load(open(ref, "rb"))
bad = 0
for tag in ("e75", "e76"):
    fp = Path(f"outputs/prob_cache/NoPnx_PA_blind_{tag}.pkl")
    if not fp.exists():
        print(f"  {fp.name}: MISSING"); bad = 1; continue
    a = pickle.load(open(fp, "rb"))
    print(f"  {fp.name}: {len(a)} docs   {ref.name}: {len(b)} docs")
    if set(a) != set(b):
        print(f"    MISMATCH — only in {tag}: {sorted(set(a)-set(b))[:5]}  "
              f"only in e41: {sorted(set(b)-set(a))[:5]}")
        bad = 1
sys.exit(bad)
PY

echo ""
for exp in e75 e76; do
  n=$(ls outputs/oof/${exp}_fold*.json 2>/dev/null | wc -l)
  echo "${exp} OOF folds present: ${n}/5"
done
echo "Finished: $(date)   (fail=${fail})"

cat <<'EOF'

--- next: ALL CPU, login node, seconds ---
  # e69's folds already exist here. Fit every subset of the four restoration routes
  # in ONE sweep and let DEV pick — they were only ever gated one at a time against
  # the same lock, which is exactly how overlapping small effects go missing.
  python gate_member.py --subtask NoPnx-PA --candidate e69 e75 e76 naqta

  # Decision rule for THIS run (the printed verdict line still states the old
  # CI-excludes-zero bar, deliberately): take the dev-selected subset if the TEST
  # P(B>A) >= 0.90 AND dev agrees in sign. Dev flat + test positive is the e66
  # dev-unselectable pattern -- do not ship that.

  # If it passes, re-fit on blind and build the submission (no GPU needed):
  #   python fit_decoder.py --subtask NoPnx-PA --members <lock + winners> \
  #       --test-split blind --output_dir outputs/blind_nopnx_pa_relock
  #   python force_final_boundary.py ...
EOF

exit $fail
