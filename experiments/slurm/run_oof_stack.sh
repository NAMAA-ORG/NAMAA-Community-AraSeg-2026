#!/bin/bash
# Phase 2 — closed-legal OOF stacking for the re-locked heads (NP, NoPnx-PA,
# NoPnx-NP). Recovers the stacking gain we gave up when we dropped the dev-fit
# stacker (see the closed-track legality note in the paper).
#
# Stage 1: retrain every pool member K times, each holding out one train fold,
#          and emit that fold's held-out predictions (train.py --holdout-fold).
# Stage 2: fit the logistic stacker on the OOF *train* matrix and apply it to the
#          full-train members' dev/test preds (fit_oof_stack.py). Each head's
#          stack depends (afterok) only on ITS OWN members' folds, so scoping to
#          one head doesn't wait on the others.
#
# e38 is multitask: its 5 fold-models cover every head, so it is trained once and
# reused. Full run = 12 members x 5 = 60 training + 3 stacking jobs.
#
# Usage (login node):
#   cd ${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments
#   bash slurm/run_oof_stack.sh                 # all 3 heads
#   bash slurm/run_oof_stack.sh nopnx-pa        # just the rank-costing head (25 jobs)
#   bash slurm/run_oof_stack.sh np nopnx-np     # any subset
set -euo pipefail

PROJECT=${ARASEG_PROJECT:?set ARASEG_PROJECT}
EXPERIMENTS_DIR=${PROJECT}/experiments
VENV=${PROJECT}/.venv-llm
cd ${EXPERIMENTS_DIR}
mkdir -p logs outputs/oof outputs/oof_ckpt

N_FOLDS=5
SEED=42

ALL_HEADS=(np nopnx-pa nopnx-np)
HEADS=("$@"); [ ${#HEADS[@]} -eq 0 ] && HEADS=("${ALL_HEADS[@]}")

declare -A SUBTASK=( [np]=NP [nopnx-pa]=NoPnx-PA [nopnx-np]=NoPnx-NP )
declare -A MEMBERS
MEMBERS[np]="e40_qwen35_9b_np e15_xlmr_single_np_s1 e16_xlmr_single_np_s2 e38_xlmr_multitask_joint"
MEMBERS[nopnx-pa]="e41_qwen35_9b_nopnx_pa e17_xlmr_single_nopnxpa_s1 e18_xlmr_single_nopnxpa_s2 e11_xlmr_single_nopnxpa_w2 e38_xlmr_multitask_joint"
MEMBERS[nopnx-np]="e42_qwen35_9b_nopnx_np e19_xlmr_single_nopnxnp_s1 e20_xlmr_single_nopnxnp_s2 e12_xlmr_single_nopnxnp_w2 e38_xlmr_multitask_joint"

for h in "${HEADS[@]}"; do
    [ -n "${SUBTASK[$h]:-}" ] || { echo "unknown head '$h' (choose from: ${ALL_HEADS[*]})"; exit 1; }
done

declare -A MEMBER_JOBS   # expid -> "jid:jid:..."  (all folds; submitted once, reused across heads)

submit_member () {   # $1=member (config basename w/o .yaml)
    local m=$1 expid=${m%%_*}
    [ -n "${MEMBER_JOBS[$m]:-}" ] && return   # already submitted (shared e38)
    local mem tlim
    if [[ $m == *qwen* ]]; then mem=80G; tlim=12:00:00; else mem=40G; tlim=06:00:00; fi
    local jids=()
    for fold in $(seq 0 $((N_FOLDS - 1))); do
        local out
        out=$(sbatch <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=oof-${expid}-f${fold}
#SBATCH -p kisski
#SBATCH -G A100:1
#SBATCH --mem-per-gpu=${mem}
#SBATCH --cpus-per-task=8
#SBATCH --time=${tlim}
#SBATCH --output=${EXPERIMENTS_DIR}/logs/slurm_%j_%x.out
#SBATCH --error=${EXPERIMENTS_DIR}/logs/slurm_%j_%x.err
export HF_HOME=${ARASEG_HF_HOME:-$HOME/.cache/huggingface}
export HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
source ${VENV}/bin/activate
cd ${EXPERIMENTS_DIR}
echo "=== OOF ${expid} fold ${fold}/${N_FOLDS} — started \$(date) ==="
python train.py --config configs/${m}.yaml \
    --holdout-fold ${fold} --n-folds ${N_FOLDS} --fold-seed ${SEED} \
    --oof-out outputs/oof/${expid}_fold${fold}.json \
    --output-dir outputs/oof_ckpt/${expid}_fold${fold}
echo "=== finished \$(date) ==="
SBATCH_EOF
)
        jids+=("${out##* }")
    done
    MEMBER_JOBS[$m]=$(IFS=:; echo "${jids[*]}")
    printf "  %-32s -> %s folds (jobs %s)\n" "$m" "$N_FOLDS" "${MEMBER_JOBS[$m]}"
}

echo "=== Stage 1: OOF training — heads: ${HEADS[*]} ==="
for h in "${HEADS[@]}"; do
    for m in ${MEMBERS[$h]}; do submit_member "$m"; done
done

echo "=== Stage 2: stacking (each head afterok on its own members' folds) ==="
for h in "${HEADS[@]}"; do
    st=${SUBTASK[$h]}
    dep=""; members=""
    for m in ${MEMBERS[$h]}; do
        dep="${dep:+$dep:}${MEMBER_JOBS[$m]}"
        members="${members} configs/${m}.yaml"
    done
    out=$(sbatch --dependency=afterok:${dep} <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=oofstack-${h}
#SBATCH -p kisski
#SBATCH -G A100:1
#SBATCH --mem-per-gpu=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=${EXPERIMENTS_DIR}/logs/slurm_%j_%x.out
#SBATCH --error=${EXPERIMENTS_DIR}/logs/slurm_%j_%x.err
export HF_HOME=${ARASEG_HF_HOME:-$HOME/.cache/huggingface}
export HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
source ${VENV}/bin/activate
cd ${EXPERIMENTS_DIR}
python fit_oof_stack.py --subtask ${st} --members ${members} \
    --oof_dir outputs/oof --n_folds ${N_FOLDS} --fold_seed ${SEED} \
    --output_dir outputs/oof_stack_${h//-/_} --thr_step 0.01
SBATCH_EOF
)
    printf "  stack %-9s -> job %s\n" "$st" "${out##* }"
done

echo "=== submitted heads: ${HEADS[*]}. Results: outputs/oof_stack_*/ensemble_*.json ==="
