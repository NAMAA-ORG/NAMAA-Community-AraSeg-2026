#!/bin/bash
# Phase B: XLM-R + recall loss + worst-genre selection (e50-e55)
# Preprocessing: build genre proxy maps before training.
# Syntax extraction is separate (run extract_syntax.py --probe first).
#
# Usage:
#   cd ${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments
#   bash slurm/run_phaseB_wave.sh

PROJECT=${ARASEG_PROJECT:?set ARASEG_PROJECT}
EXPERIMENTS_DIR=${PROJECT}/experiments
VENV=${PROJECT}/.venv-llm

cd ${EXPERIMENTS_DIR}
mkdir -p logs outputs/genre_proxy

CONFIGS=(
    "e50_xlmr_nopnx_np_recall_s1.yaml"
    "e51_xlmr_nopnx_pa_recall_s1.yaml"
    "e52_xlmr_nopnx_np_recall_s2.yaml"
    "e53_xlmr_nopnx_pa_recall_s2.yaml"
    "e54_xlmr_np_recall_s1.yaml"
    "e55_xlmr_pa_recall_s1.yaml"
)

echo "=== Phase B Wave: ${#CONFIGS[@]} experiments ==="
echo ""

# ── Step 1: Genre proxy preprocessing (login-node safe, fast) ─────────────
# Runs in the venv on the login node; needs sklearn.
echo "Building genre proxy maps..."
source ${VENV}/bin/activate

for SUBTASK in NoPnx-NP NoPnx-PA NP PA; do
    KEY=$(echo "$SUBTASK" | tr 'A-Z-' 'a-z_')
    MAP_PATH=outputs/genre_proxy/${KEY}_dev.json
    if [ -f "$MAP_PATH" ]; then
        echo "  [skip] $MAP_PATH already exists"
    else
        echo "  Building genre map for $SUBTASK..."
        python genre_proxy.py --subtask "$SUBTASK" --split dev --out outputs/genre_proxy
    fi
done
echo ""

# ── Step 2: Submit training jobs ───────────────────────────────────────────
echo "Submitting training jobs..."
for config in "${CONFIGS[@]}"; do
    sbatch_out=$(sbatch << SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=araseg-phaseB
#SBATCH -p kisski
#SBATCH -G A100:1
#SBATCH --mem-per-gpu=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=${EXPERIMENTS_DIR}/logs/slurm_%j_%x.out
#SBATCH --error=${EXPERIMENTS_DIR}/logs/slurm_%j_%x.err

export HF_HOME=${ARASEG_HF_HOME:-$HOME/.cache/huggingface}
export HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

source ${VENV}/bin/activate
cd ${EXPERIMENTS_DIR}

echo "=== Phase B: ${config} ==="
echo "Node:    \$(hostname)"
echo "GPU:     \$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "Started: \$(date)"

python train.py --config configs/${config}

echo "Finished: \$(date)"
SBATCH_EOF
)
    job_id=$(echo "$sbatch_out" | awk '{print $NF}')
    printf "  %-48s -> job %s\n" "$config" "$job_id"
done

echo ""
echo "Monitor: squeue -u \$USER"
echo "Results: python ${EXPERIMENTS_DIR}/collect_results.py"
echo ""
echo "After jobs finish, run Phase C ensemble sweep:"
echo "  Update POOL in run_phaseA_ensembles.sh (or a new run_phaseC_ensembles.sh)"
echo "  to include e50-e55 alongside e34/e40/e41/e42."
echo ""
echo "Optional — syntax extraction (run before e56+ configs):"
echo "  python extract_syntax.py --probe"
echo "  python extract_syntax.py --subtask NoPnx-NP"
echo "  python extract_syntax.py --subtask NoPnx-PA"
