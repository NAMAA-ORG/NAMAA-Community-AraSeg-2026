#!/bin/bash
# SaT OOF stacking — does the full-FT SaT member earn stacker weight? For each head:
#   Stage 1: train SaT 5x (each holding out one train fold), write the held-out fold's
#            {doc_id:(prob,gold)} pkl  (sat_finetune.py --holdout-fold). SaT is small,
#            ~3-6 min/fold — cheap next to the 12h Qwen folds.
#   Stage 2: re-fit the head's logistic stacker with SaT appended, REUSING the existing
#            member OOFs in outputs/oof/ (from run_oof_stack.sh) — only SaT is new.
# --punct-corrupt is a no-op on NP (punctuated) and rebuilds NoPnx from its parent,
# matching exactly how the full-FT dev/test caches in sat_fullft_caches were made.
#
# Submit (login node):
#   cd ${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments
#   bash slurm/run_sat_oof.sh                 # all 3 heads
#   bash slurm/run_sat_oof.sh np              # just NP (the only member-competitive head)
set -euo pipefail

PROJECT=${ARASEG_PROJECT:?set ARASEG_PROJECT}
EXPERIMENTS_DIR=${PROJECT}/experiments
VENV=${PROJECT}/.venv-llm
cd ${EXPERIMENTS_DIR}
mkdir -p logs outputs/oof_sat outputs/sat_oof_ckpt

N_FOLDS=5
SEED=42
SATCACHE=outputs/sat_fullft_caches           # dev/test SaT caches (already on disk)

ALL_HEADS=(np nopnx-pa nopnx-np)
HEADS=("$@"); [ ${#HEADS[@]} -eq 0 ] && HEADS=("${ALL_HEADS[@]}")

declare -A SUBTASK=( [np]=NP [nopnx-pa]=NoPnx-PA [nopnx-np]=NoPnx-NP )
declare -A MEMBERS
MEMBERS[np]="e40_qwen35_9b_np e15_xlmr_single_np_s1 e16_xlmr_single_np_s2 e38_xlmr_multitask_joint"
MEMBERS[nopnx-pa]="e41_qwen35_9b_nopnx_pa e17_xlmr_single_nopnxpa_s1 e18_xlmr_single_nopnxpa_s2 e11_xlmr_single_nopnxpa_w2 e38_xlmr_multitask_joint"
MEMBERS[nopnx-np]="e42_qwen35_9b_nopnx_np e19_xlmr_single_nopnxnp_s1 e20_xlmr_single_nopnxnp_s2 e12_xlmr_single_nopnxnp_w2 e38_xlmr_multitask_joint"

for h in "${HEADS[@]}"; do
    [ -n "${SUBTASK[$h]:-}" ] || { echo "unknown head '$h' (choose: ${ALL_HEADS[*]})"; exit 1; }
done

echo "=== Stage 1: SaT OOF folds — heads: ${HEADS[*]} ==="
for h in "${HEADS[@]}"; do
    st=${SUBTASK[$h]}; st_us=${st//-/_}
    jids=()
    for fold in $(seq 0 $((N_FOLDS - 1))); do
        out=$(sbatch <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=satoof-${h}-f${fold}
#SBATCH -A kisski-futurmig
#SBATCH -p kisski
#SBATCH -G A100:1
#SBATCH --mem-per-gpu=40G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=${EXPERIMENTS_DIR}/logs/slurm_%j_%x.out
#SBATCH --error=${EXPERIMENTS_DIR}/logs/slurm_%j_%x.err
export HF_HOME=${ARASEG_HF_HOME:-$HOME/.cache/huggingface}
export HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
source ${VENV}/bin/activate
cd ${EXPERIMENTS_DIR}
python sat_finetune.py --subtask ${st} --punct-corrupt \
    --holdout-fold ${fold} --n-folds ${N_FOLDS} --fold-seed ${SEED} \
    --out-dir outputs/sat_oof_ckpt \
    --oof-out outputs/oof_sat/${st_us}_fold${fold}_sat_ft.pkl
SBATCH_EOF
)
        jids+=("${out##* }")
    done
    dep=$(IFS=:; echo "${jids[*]}")
    printf "  %-9s SaT folds -> jobs %s\n" "$st" "$dep"

    members=""
    for m in ${MEMBERS[$h]}; do members="${members} configs/${m}.yaml"; done
    stack=$(sbatch --dependency=afterok:${dep} <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=satoofstack-${h}
#SBATCH -A kisski-futurmig
#SBATCH -p kisski
#SBATCH -G A100:1
#SBATCH --mem-per-gpu=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=${EXPERIMENTS_DIR}/logs/slurm_%j_%x.out
#SBATCH --error=${EXPERIMENTS_DIR}/logs/slurm_%j_%x.err
export HF_HOME=${ARASEG_HF_HOME:-$HOME/.cache/huggingface}
export HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
source ${VENV}/bin/activate
cd ${EXPERIMENTS_DIR}
python fit_oof_stack.py --subtask ${st} --members ${members} \
    --oof_dir outputs/oof --n_folds ${N_FOLDS} --fold_seed ${SEED} \
    --sat_oof_dir outputs/oof_sat --sat_cache_dir ${SATCACHE} \
    --output_dir outputs/oof_stack_sat_${h//-/_} --thr_step 0.01
SBATCH_EOF
)
    printf "  stack %-9s -> job %s (afterok SaT folds)\n" "$st" "${stack##* }"
done
echo "=== done. Results: outputs/oof_stack_sat_*/ensemble_*.json (compare to outputs/oof_stack_*/) ==="
