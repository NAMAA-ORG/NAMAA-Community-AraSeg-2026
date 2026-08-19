#!/bin/bash
# Q9 round 2: the SUPERVISION control for the round-1 negative.
#
# Round 1 (job 15104369) trained on 3253 in-band candidates and returned +0.01 against a
# measured +4.28 delete-only ceiling. But it memorized that set by epoch 4 (loss
# 0.57 -> 0.23 while dev AND test fell), so it cannot distinguish
#
#     "local context carries no signal"   from   "3253 examples is too few to learn it"
#
# and only the first is worth writing down. This trains on EVERY train token (~30x the
# supervision) and still applies in-band. Either it finds the headroom, or the negative
# becomes a supported claim instead of an artefact of my own design choice to restrict
# the training set.
#
# NoPnx-PA only. Scope is inherently limited: the verifier rides on the OOF decoder, so
# it can reach NoPnx-PA and NoPnx-NP; NP would need the band redefined on the stack
# posterior, and PA is hard-blocked because e25/e32/e33 have no OOF train predictions.
# NoPnx-PA is the -1.4 gap and where the ceiling was measured, so it goes first alone --
# if this is another +0.01 there is no reason to extend to the other heads.
#
# Class balance flips from 48.4% positive (in-band) to ~8.5% (all tokens); train_verifier
# already weights the loss, which matters much more here.
#
# Usage (login node):
#   cd ${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments
#   sbatch slurm/run_verify_band_all.sh

#SBATCH --job-name=araseg-verify-all
#SBATCH -p kisski
#SBATCH -G A100:1
# ~100k candidates instead of 3253: bigger tokenized tensors and ~3100 steps/epoch
# instead of ~100, so more memory and time than the round-1 job (16G/45min).
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:45:00
#SBATCH --output=${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments/logs/slurm_%A_%x.out
#SBATCH --error=${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments/logs/slurm_%A_%x.err

set -euo pipefail

export HF_HOME=${ARASEG_HF_HOME:-$HOME/.cache/huggingface}
export HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1

source ${ARASEG_PROJECT:?set ARASEG_PROJECT}/.venv-llm/bin/activate
cd ${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments

ST=NoPnx-PA
MODEL=${ARASEG_HF_HOME:-$HOME/.cache/huggingface}/hub/camelbert-msa
[ -d "$MODEL" ] || { echo "FATAL: backbone folder missing: $MODEL"; exit 1; }

echo "Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) | Subtask: $ST | Started: $(date)"

# Same fail-fast as round 1: no GPU, no backbone, seconds. If the candidate indices are
# wrong every config below measures the wrong tokens.
python verify_band.py --subtask "$ST" --oracle

# window 40 only -- it beat window 20 on dev in round 1. Epochs 1 and 2, not 2 and 4:
# with 30x the data an epoch is 30x the steps, and round 1's failure mode was too much
# fitting, not too little.
for EP in 1 2; do
    echo "=== all-tokens window=40 epochs=$EP ==="
    python verify_band.py --subtask "$ST" --model "$MODEL" --all-tokens \
        --window 40 --epochs "$EP" \
        --output_dir "outputs/verify_nopnx_pa_all_w40_e${EP}" \
        || echo "FAILED all_w40_e${EP} -- traceback in logs/slurm_${SLURM_JOB_ID}_${SLURM_JOB_NAME}.err"
done

echo "--- dev summary (the selection signal; test is reported, never selected on) ---"
python - <<'PY'
import json, glob
rows = []
for fp in sorted(glob.glob("outputs/verify_nopnx_pa_all_w40_e*/verify_NoPnx_PA.json")):
    d = json.load(open(fp))
    rows.append((d["dev"]["f1"], d["epochs"], d["mode"], d["verifier_threshold"],
                 d["lock_dev"]["f1"], d["test"]["f1"], d["lock_test"]["f1"]))
if not rows:
    raise SystemExit("no results -- every config failed, check the .err log")
print(f"{'ep':>3} {'mode':>12} {'vthr':>5} {'devF1':>7} {'d_dev':>7} {'testF1':>7} {'d_test':>7}")
for f1, ep, mode, vthr, ldev, tst, ltst in sorted(rows, reverse=True):
    print(f"{ep:>3} {mode:>12} {vthr:>5.2f} {f1:>7.4f} "
          f"{100*(f1-ldev):>+7.2f} {tst:>7.4f} {100*(tst-ltst):>+7.2f}")
PY

cat <<'NOTE'

HOW TO READ THIS. Round 1 was +0.01 dev / +0.01 test with dev picking the no-op.

  * a dev delta still ~0.00, with vthr pinned to the bottom of the grid -> the negative
    is now SUPPORTED: full supervision, still nothing. That is the paper claim, and it
    is worth more than another null member because the ceiling was measured first.
  * a real dev delta -> gate it. bootstrap_paired.py against the locked NoPnx-PA CSV,
    CI must exclude 0. Only then consider extending to NoPnx-NP.

Do NOT select on the test column. That is how e66 was lost.
NOTE
echo "Finished at $(date)"
