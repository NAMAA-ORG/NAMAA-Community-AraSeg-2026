#!/bin/bash
# PA OOF folds — the missing ingredient for a PA structural decoder.
#
# WHY (2026-07-13): iMak AI Lab took #1 on all 4 closed; our PA lock (94.49) leads
# them by +0.09, which is noise. PA is the only head with no OOF folds, so it is the
# only head that never got a decoder — and oracle_headroom.py says it should:
#
#   head       oracle-k (ranking headroom)   real decoder
#   NoPnx-PA        +1.55                       +0.92
#   NP              +0.52                       -0.13
#   PA              +1.06                       ~+0.4 (estimated)
#
# oracle-k is the predictor that matches reality (oracle-thr does NOT — NP had +1.76
# of it and still lost). PA sits well above the ~+0.65 break-even. PA's lock is also
# a FLAT logit average with no learned member weights, so a decoder brings weighting
# AND structure, where NP's could only add structure to an already-stacked baseline.
#
# COST: ~21 A100-hours total, dominated by the two big LLMs (Qwen 9B ~105 min/fold,
# Gemma 12B ~140 min/fold est.). Fold jobs are independent — they fan out across the
# queue, so wall-clock is bounded by the longest single job (~2.5 h), not the sum.
#
# Then, CPU-only:
#   python fit_decoder.py --subtask PA \
#       --members configs/e25_xlmr_single_pa_stride256.yaml \
#                 configs/e32_qwen35_9b_pa.yaml configs/e33_gemma4_12b_pa.yaml \
#       --oof_dir outputs/oof --sat_oof_dir outputs/oof_sat \
#       --cache_dir outputs/prob_cache --sat_cache_dir outputs/sat_fullft_caches \
#       --output_dir outputs/decoder_pa
#   python bootstrap_paired.py --subtask PA --split test \
#       --a outputs/phaseC_v2_pa_logit/test_predictions_PA.csv \
#       --b outputs/decoder_pa/test_predictions_PA.csv
# Re-lock ONLY if the paired CI excludes zero (this gate already killed the NP decoder).
#
# Usage (login node):
#   cd .../AraSeg/experiments && bash slurm/run_oof_pa.sh

set -euo pipefail

PROJECT=${ARASEG_PROJECT:?set ARASEG_PROJECT}
EXPERIMENTS_DIR=${PROJECT}/experiments
VENV=${PROJECT}/.venv-llm
ACCOUNT=kisski-futurmig

N_FOLDS=5
SEED=42          # MUST match fit_decoder.py's --fold_seed default (42) and --n_folds (5)

mkdir -p "${EXPERIMENTS_DIR}/logs" "${EXPERIMENTS_DIR}/outputs/oof" \
         "${EXPERIMENTS_DIR}/outputs/oof_sat"

# expid : config : mem : timelimit  — LLMs get the big box, XLM-R the small one.
MEMBERS=(
  "e25:e25_xlmr_single_pa_stride256:40G:06:00:00"
  "e32:e32_qwen35_9b_pa:80G:12:00:00"
  "e33:e33_gemma4_12b_pa:80G:12:00:00"
)

echo "=== Stage 1: PA OOF member folds (${N_FOLDS} folds x ${#MEMBERS[@]} members) ==="
jids=()
for entry in "${MEMBERS[@]}"; do
    expid=${entry%%:*}; rest=${entry#*:}
    cfg=${rest%%:*};    rest=${rest#*:}
    mem=${rest%%:*};    tlim=${rest#*:}

    for fold in $(seq 0 $((N_FOLDS - 1))); do
        if [ -f "${EXPERIMENTS_DIR}/outputs/oof/${expid}_fold${fold}.json" ]; then
            echo "  [skip] ${expid} fold ${fold} — OOF already exists"
            continue
        fi
        out=$(sbatch <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=oofpa-${expid}-f${fold}
#SBATCH -A ${ACCOUNT}
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
echo "=== PA OOF ${expid} fold ${fold}/${N_FOLDS} — started \$(date) ==="
python train.py --config configs/${cfg}.yaml \
    --holdout-fold ${fold} --n-folds ${N_FOLDS} --fold-seed ${SEED} \
    --oof-out outputs/oof/${expid}_fold${fold}.json \
    --output-dir outputs/oof_ckpt/${expid}_fold${fold}
echo "=== finished \$(date) ==="
SBATCH_EOF
)
        jids+=("${out##* }")
    done
    echo "  ${expid}: submitted"
done

# sat_ft PA folds. PA was never in the SaT OOF sweep (only NP / NoPnx-PA / NoPnx-NP),
# yet sat_ft took the TOP member weight in both NoPnx decoders — so PA wants it too.
# SaT is tiny: ~4 min/fold.
echo "=== Stage 2: PA sat_ft folds ==="
for fold in $(seq 0 $((N_FOLDS - 1))); do
    if [ -f "${EXPERIMENTS_DIR}/outputs/oof_sat/PA_fold${fold}_sat_ft.pkl" ]; then
        echo "  [skip] sat_ft fold ${fold} — OOF already exists"
        continue
    fi
    out=$(sbatch <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=oofpa-sat-f${fold}
#SBATCH -A ${ACCOUNT}
#SBATCH -p kisski
#SBATCH -G A100:1
#SBATCH --mem-per-gpu=40G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=${EXPERIMENTS_DIR}/logs/slurm_%j_%x.out
#SBATCH --error=${EXPERIMENTS_DIR}/logs/slurm_%j_%x.err
export HF_HOME=${ARASEG_HF_HOME:-$HOME/.cache/huggingface}
export HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
source ${VENV}/bin/activate
cd ${EXPERIMENTS_DIR}
echo "=== PA SaT OOF fold ${fold}/${N_FOLDS} — started \$(date) ==="
python sat_finetune.py --subtask PA --punct-corrupt \
    --holdout-fold ${fold} --n-folds ${N_FOLDS} --fold-seed ${SEED} \
    --out-dir outputs/sat_oof_ckpt \
    --oof-out outputs/oof_sat/PA_fold${fold}_sat_ft.pkl
echo "=== finished \$(date) ==="
SBATCH_EOF
)
    jids+=("${out##* }")
done

echo
echo "Submitted ${#jids[@]} jobs: ${jids[*]}"
echo "Watch:  squeue -u \$USER -n \$(squeue -u \$USER -h -o %j | tr '\n' ',' )"
echo "When all 20 OOF files exist, run fit_decoder.py + bootstrap_paired.py (CPU, see header)."
