#!/bin/bash
# Q9: boundary-verifier cross-encoder over the locked decoder's ambiguous band.
#
# Gate that authorised this run (band_mass.py, 2026-07-30, CPU): on NoPnx-PA the
# [0.2,0.8] band is 4.14% of tokens but carries 71.6% of all false positives, and a
# delete-only oracle over it is worth +4.28 F1 with recall flat. We need ~1/3 of that
# to take NoPnx-PA #1 (currently #3, -1.4 behind mohamed_mohamed).
#
# NoPnx-PA ONLY, deliberately. NoPnx-NP is a head we currently lead (+0.4) and re-locking
# it would put the sequence-decoder 2x2 -- the paper's headline result -- at risk.
#
# Training set is ~5k candidates drawn from OOF TRAIN predictions with train text and
# train gold, so no learned weight sees dev or test (closed-track legal). Dev picks only
# the application mode + one threshold, as every existing lock picks its threshold.
# Minutes per config, so the four-config dev sweep runs in one slot instead of queueing
# four times.
#
# Backbone: the LOCAL folder e2/e5 already used on this cluster (see
# configs/e2_camelbert_single_pa.yaml). No hub access, no token, nothing to download --
# the cluster runs HF_HUB_OFFLINE=1 and a hub id would 401 against the stale token in
# the environment.
#
# Usage (login node):
#   cd ${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments
#   sbatch slurm/run_verify_band.sh

#SBATCH --job-name=araseg-verify-band
#SBATCH -p kisski
#SBATCH -G A100:1
# Sized for what this actually is: a 108M encoder over ~5k short candidates. Training is
# ~2 min across all four configs; the wall time is the CPU replay in the dev sweep. The
# first version asked for 64G/2h and was scheduled 2 days out -- a short, small job
# backfills into gaps that a big one cannot.
#SBATCH --mem=16G
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

source ${ARASEG_PROJECT:?set ARASEG_PROJECT}/.venv-llm/bin/activate
cd ${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments

ST=NoPnx-PA
MODEL=${ARASEG_HF_HOME:-$HOME/.cache/huggingface}/hub/camelbert-msa
[ -d "$MODEL" ] || { echo "FATAL: backbone folder missing: $MODEL"; exit 1; }

echo "Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) | Subtask: $ST | Started: $(date)"

# Cheap fail-fast: the plumbing check needs no GPU and no backbone. If the oracle run
# does not reproduce band_mass.py's ceiling, the candidate indices are wrong and every
# trained config below would be measuring the wrong tokens.
python verify_band.py --subtask "$ST" --oracle

# window x epochs; dev picks the winner (mode + verifier threshold are swept inside)
for WIN in 20 40; do
    for EP in 2 4; do
        echo "=== window=$WIN epochs=$EP ==="
        python verify_band.py --subtask "$ST" --model "$MODEL" \
            --window "$WIN" --epochs "$EP" \
            --output_dir "outputs/verify_nopnx_pa_w${WIN}_e${EP}" \
            || echo "FAILED w${WIN}_e${EP} -- traceback is in logs/slurm_${SLURM_JOB_ID}_${SLURM_JOB_NAME}.err"
    done
done

echo "--- dev summary (the selection signal; test is reported, never selected on) ---"
python - <<'PY'
import json, glob
rows = []
for fp in sorted(glob.glob("outputs/verify_nopnx_pa_w*/verify_NoPnx_PA.json")):
    d = json.load(open(fp))
    rows.append((d["dev"]["f1"], d["window"], d["epochs"], d["mode"],
                 d["verifier_threshold"], d["lock_dev"]["f1"], d["test"]["f1"],
                 d["lock_test"]["f1"]))
if not rows:
    raise SystemExit("no results -- every config failed, check the .err log")
print(f"{'win':>4} {'ep':>3} {'mode':>12} {'vthr':>5} {'devF1':>7} {'d_dev':>7} {'testF1':>7} {'d_test':>7}")
for f1, win, ep, mode, vthr, ldev, tst, ltst in sorted(rows, reverse=True):
    print(f"{win:>4} {ep:>3} {mode:>12} {vthr:>5.2f} {f1:>7.4f} "
          f"{100*(f1-ldev):>+7.2f} {tst:>7.4f} {100*(tst-ltst):>+7.2f}")
best = max(rows)
print(f"\ndev winner: window={best[1]} epochs={best[2]} mode={best[3]} "
      f"-> outputs/verify_nopnx_pa_w{best[1]}_e{best[2]}")
print("GATE: python bootstrap_paired.py against the locked NoPnx-PA CSV. Do NOT re-lock")
print("on the test delta alone -- CI must exclude 0, per the pre-registered rule.")
PY

echo "Finished at $(date)"
