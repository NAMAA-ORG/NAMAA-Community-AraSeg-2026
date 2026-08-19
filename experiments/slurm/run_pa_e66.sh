#!/bin/bash
# PA + e66/RemBERT — the best remaining lever on the tied head.
#
# Context (2026-07-17): PA is a +0.09 tie with iMak AI Lab, and the two levers we
# were counting on are now both dead:
#   - sat_ft: job 14834487 swept it into the pool, every subset LOST (s1 94.39,
#     s2 94.38, s3 94.27 vs the 94.49 lock). Not retried here.
#   - the structural decoder: -0.19, CI [-0.49,+0.11] -> NOISE, do not re-lock.
#     doc_density -4.53 / local_rank -0.19 = the NP signature exactly. Both
#     punctuated heads (NP, PA) reject the decoder; both NoPnx heads accept it.
#     run_pa_sat.sh predicted this on 07-13 before the folds ran.
#
# So e66 is what's left, and it's the best-motivated member the pool has ever had:
# RemBERT is a pretraining family PA has never contained (pool = XLM-R/Qwen/Gemma),
# and at dev 0.9408 / test 0.9336 it is ANCHOR-TIER, not a weak add-on — it ties
# e32 (dev 0.9407) and beats every member's solo PA test except e32 (93.34).
# The rule that held every previous time: a member stronger than the anchor lifts
# the stack; weaker ones get zeroed. e66 is the former.
#
# Legality: prob/logit have NO learned parameters -> no OOF folds needed (e66 has
# none). Subset + threshold picked on DEV = ordinary model selection, the same rule
# every existing lock lives under.
#
# e30 is deliberately absent: it killed job 14834487's s4 stage with
# `cuDNN error: CUDNN_STATUS_NOT_INITIALIZED` (RNN head, flatten_parameters).
# e64 is absent too: dev 0.9205 is far below the anchors -> it is the "weaker
# member that gets zeroed" case, and costs a member slot to prove it.
#
# Usage (login node):
#   cd .../AraSeg/experiments && sbatch slurm/run_pa_e66.sh

#SBATCH --job-name=araseg-pa-e66
#SBATCH -p kisski
#SBATCH -G A100:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
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

echo "Node: $(hostname) | Started: $(date)"
test -f outputs/e66/results.json || { echo "MISSING outputs/e66 — e66 never trained"; exit 1; }

E25=configs/e25_xlmr_single_pa_stride256.yaml
E32=configs/e32_qwen35_9b_pa.yaml
E33=configs/e33_gemma4_12b_pa.yaml
E66=configs/e66_rembert_pa_tuned.yaml

# Cache e66's PA probs while a GPU is in hand: idempotent, skips the 6 existing
# e25/e32/e33 files, and blind day needs every member cached anyway.
python cache_probs.py --subtask PA --splits dev test --members $E66 \
    --out_dir outputs/prob_cache

# t1 = the lock + e66 (the direct A/B). t2 drops e25, the weakest member by dev
# (0.9186 vs ~0.941) — worth one slot to ask whether e66 replaces it rather than
# joins it. t3 = the two strongest + e66.
T1="$E25 $E32 $E33 $E66"
T2="$E32 $E33 $E66"
T3="$E32 $E66"

# Implementation note: `|| echo FAILED` per subset, not `set -e` killing the job. Job 14834487
# lost its entire prob sweep because one bad member aborted the whole script.
run () {  # run <name> <combine> <members...>
    local name=$1 mode=$2; shift 2
    echo "=== ${name} / ${mode} ==="
    python ensemble.py --subtask PA --members "$@" \
        --output_dir "outputs/pa_e66_${name}_${mode}" \
        --combine "$mode" --thr_step 0.01 \
        || echo "!!! ${name}/${mode} FAILED — continuing"
}

for MODE in logit prob; do
    run t1 "$MODE" $T1
    run t2 "$MODE" $T2
    run t3 "$MODE" $T3
done

echo "Finished: $(date)"
echo "Bar to beat: PA lock = 94.49 (logit e25 e32 e33 @0.25)."
echo "A higher test F1 is NOT enough — gate the winner on the paired bootstrap:"
echo "  python bootstrap_paired.py --subtask PA --split test \\"
echo "    --a outputs/phaseC_v2_pa_logit/test_predictions_PA.csv \\"
echo "    --b outputs/pa_e66_<winner>/test_predictions_PA.csv"
echo "Re-lock ONLY if the CI excludes zero. This gate has now killed 2 of 3 PA levers."
