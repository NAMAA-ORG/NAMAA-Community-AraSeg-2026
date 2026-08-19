#!/bin/bash
# One-shot fix for the NP OOF-stack: e40 fold0 (job 14711398) died in 44s on a
# transient offline-HF-cache race ("Split 'dev' not found ... Available: ['test',
# 'dev','train']") when all 5 folds started at once. Folds 1-4 + e15/e16/e38 are
# done. A lone re-run can't race, so just retrain fold0 and re-gate the NP stack.
# Implementation note: standalone, mirrors run_oof_stack.sh env; delete after NP relocks.
set -euo pipefail

PROJECT=${ARASEG_PROJECT:?set ARASEG_PROJECT}
EXPERIMENTS_DIR=${PROJECT}/experiments
VENV=${PROJECT}/.venv-llm
cd ${EXPERIMENTS_DIR}

N_FOLDS=5
SEED=42
M=e40_qwen35_9b_np

# Stage 1: retrain just e40 fold0
f0=$(sbatch <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=oof-e40-f0
#SBATCH -p kisski
#SBATCH -G A100:1
#SBATCH --mem-per-gpu=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=${EXPERIMENTS_DIR}/logs/slurm_%j_%x.out
#SBATCH --error=${EXPERIMENTS_DIR}/logs/slurm_%j_%x.err
export HF_HOME=${ARASEG_HF_HOME:-$HOME/.cache/huggingface}
export HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
source ${VENV}/bin/activate
cd ${EXPERIMENTS_DIR}
echo "=== OOF e40 fold 0/${N_FOLDS} (rerun) — started \$(date) ==="
python train.py --config configs/${M}.yaml \
    --holdout-fold 0 --n-folds ${N_FOLDS} --fold-seed ${SEED} \
    --oof-out outputs/oof/e40_fold0.json \
    --output-dir outputs/oof_ckpt/e40_fold0
echo "=== finished \$(date) ==="
SBATCH_EOF
)
f0=${f0##* }
echo "  e40 fold0 rerun -> job ${f0}"

# Stage 2: NP stack, gated on fold0 finishing (all other NP OOF files already exist)
MEMBERS="configs/e40_qwen35_9b_np.yaml configs/e15_xlmr_single_np_s1.yaml configs/e16_xlmr_single_np_s2.yaml configs/e38_xlmr_multitask_joint.yaml"
st=$(sbatch --dependency=afterok:${f0} <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=oofstack-np
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
python fit_oof_stack.py --subtask NP --members ${MEMBERS} \
    --oof_dir outputs/oof --n_folds ${N_FOLDS} --fold_seed ${SEED} \
    --output_dir outputs/oof_stack_np --thr_step 0.01
SBATCH_EOF
)
echo "  NP stack (afterok:${f0}) -> job ${st##* }"
echo "=== done. Result: outputs/oof_stack_np/ensemble_*.json ==="
