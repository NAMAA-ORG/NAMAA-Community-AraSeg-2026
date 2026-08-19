#!/bin/bash
# PA + sat_ft — the one cheap, untried lever on the head iMak AI Lab just tied us on.
#
# Context (2026-07-13): iMak AI Lab took #1 on all 4 closed. Our PA lock (logit
# e25 e32 e33 @0.25 = 94.49) now leads them by only +0.09 — inside noise. PA is
# also the ONLY head with no sat_ft member: the SaT prob-probe measured PA
# 0.9422 -> 0.9435 (+0.13) from adding it, and we never followed up because PA
# had a comfortable +0.39 cushion at the time. It doesn't any more.
#
# Legality: prob/logit combiners have NO learned parameters, so this needs no OOF
# folds (which e25/e32/e33 don't have). Subset + threshold are selected on DEV =
# ordinary model selection, same rule every existing lock lives under.
#
# Deliberately NOT doing a PA structural decoder: PA is a punctuated head, and on
# the other punctuated head (NP) the decoder came back at -0.13 with doc_density
# -5.13 — it had no information left to add once punctuation localises the
# boundaries. A PA decoder would also need 15 new GPU fold-trainings. Bad trade.
#
# --exhaustive is incompatible with --sat_cache (exhaustive recomputes members
# from checkpoints), so the candidate subsets are enumerated by hand below.
#
# Usage (login node):
#   cd .../AraSeg/experiments && sbatch slurm/run_pa_sat.sh

#SBATCH --job-name=araseg-pa-sat
#SBATCH -p kisski
#SBATCH -G A100:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
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

SAT=outputs/sat_fullft_caches
echo "Node: $(hostname) | Started: $(date)"
test -f "${SAT}/PA_dev_sat_ft.pkl" || { echo "MISSING ${SAT}/PA_dev_sat_ft.pkl"; exit 1; }
test -f "${SAT}/PA_test_sat_ft.pkl" || { echo "MISSING ${SAT}/PA_test_sat_ft.pkl"; exit 1; }

# Candidate subsets. S1 = the current lock's members (the direct A/B for sat_ft).
S1="configs/e25_xlmr_single_pa_stride256.yaml configs/e32_qwen35_9b_pa.yaml configs/e33_gemma4_12b_pa.yaml"
S2="$S1 configs/e34_qwen35_9b_pa_rdrop.yaml"
S3="configs/e32_qwen35_9b_pa.yaml configs/e33_gemma4_12b_pa.yaml"
S4="$S1 configs/e30_xlmr_pa_kitchen_sink.yaml"

run () {  # run <name> <combine> <members...>
    local name=$1 mode=$2; shift 2
    echo "=== ${name} / ${mode} + sat_ft ==="
    python ensemble.py --subtask PA --members "$@" \
        --output_dir "outputs/pa_sat_${name}_${mode}" \
        --combine "$mode" --sat_cache "$SAT" --thr_step 0.01
}

for MODE in logit prob; do
    run s1 "$MODE" $S1
    run s2 "$MODE" $S2
    run s3 "$MODE" $S3
    run s4 "$MODE" $S4
done

echo "Finished: $(date)"
echo "Bar to beat: PA lock = 94.49 (logit e25 e32 e33 @0.25, NO sat_ft)."
echo "Gate any re-lock with: python bootstrap_paired.py --subtask PA --split test \\"
echo "  --a outputs/phaseC_v2_pa_logit/test_predictions_PA.csv \\"
echo "  --b outputs/pa_sat_<winner>/test_predictions_PA.csv"
