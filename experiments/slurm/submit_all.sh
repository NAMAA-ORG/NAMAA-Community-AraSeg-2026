#!/bin/bash
# Run from anywhere:
# bash ${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments/slurm/submit_all.sh

PROJECT=${ARASEG_PROJECT:?set ARASEG_PROJECT}
EXPERIMENTS_DIR=${PROJECT}/experiments

# Virtualenvs. AraModernBert needs transformers>=4.48 in a SEPARATE env so it
# doesn't disturb the working XLM-R/AraBERT env.
DEFAULT_VENV=${PROJECT}/.venv
MODERNBERT_VENV=${PROJECT}/.venv-modernbert   # <-- create this; transformers>=4.48

# sbatch resolves relative --output/--error paths against CWD, so submit from experiments/
cd ${EXPERIMENTS_DIR}
mkdir -p logs

# Fixed-pipeline runs: encoder re-benchmark (e1/e2/e3/e6) + all 4 subtasks with
# tuned pos_weights (e8/e10/e11/e12). These are the competition submission models.
CONFIGS=(
    "e1_arabert_single_pa.yaml"
    "e2_camelbert_single_pa.yaml"
    "e3_xlmr_single_pa.yaml"
    "e6_aramodernbert_single_pa.yaml"
    "e8_xlmr_single_np.yaml"
    "e10_xlmr_single_pa_weighted.yaml"
    "e11_xlmr_single_nopnxpa_w2.yaml"
    "e12_xlmr_single_nopnxnp_w2.yaml"
)

# Per-config venv override (defaults to DEFAULT_VENV).
venv_for() {
    case "$1" in
        e6_aramodernbert*) echo "${MODERNBERT_VENV}" ;;
        *)                 echo "${DEFAULT_VENV}" ;;
    esac
}

echo "Submitting ${#CONFIGS[@]} experiments..."
for config in "${CONFIGS[@]}"; do
    venv=$(venv_for "$config")
    sbatch_out=$(sbatch << SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=araseg
#SBATCH -p kisski
#SBATCH -G A100:1
#SBATCH --mem-per-gpu=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=${EXPERIMENTS_DIR}/logs/slurm_%j_%x.out
#SBATCH --error=${EXPERIMENTS_DIR}/logs/slurm_%j_%x.err

export HF_HOME=${ARASEG_HF_HOME:-$HOME/.cache/huggingface}
export HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

source ${venv}/bin/activate
cd ${EXPERIMENTS_DIR}
echo "=== AraSeg: ${config} ==="
echo "Node:    \$(hostname)"
echo "GPU:     \$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "Venv:    ${venv}"
echo "Python:  \$(python --version)"
echo "Transf.: \$(python -c 'import transformers; print(transformers.__version__)' 2>/dev/null)"
echo "Started: \$(date)"
python train.py --config configs/${config}
echo "Finished: \$(date)"
SBATCH_EOF
)
    job_id=$(echo "$sbatch_out" | awk '{print $NF}')
    printf "  %-40s [venv: %-28s] -> job %s\n" "$config" "$(basename "$venv")" "$job_id"
done

echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs:    ${EXPERIMENTS_DIR}/logs/"
echo "Results: python ${EXPERIMENTS_DIR}/collect_results.py"
