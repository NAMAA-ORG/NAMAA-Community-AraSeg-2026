#!/bin/bash
# Wave A: LLM bidirectional encoders on PA (e31-e33), using .venv-llm
# (newer transformers + peft). Heredoc pattern required — cluster rejects
# sbatch --export (see docs/handoff.md).

PROJECT=${ARASEG_PROJECT:?set ARASEG_PROJECT}
EXPERIMENTS_DIR=${PROJECT}/experiments
VENV=${PROJECT}/.venv-llm

# sbatch resolves relative --output/--error paths against CWD, so submit from experiments/
cd ${EXPERIMENTS_DIR}
mkdir -p logs

CONFIGS=(
    "e31_qwen25_7b_pa.yaml"
    "e32_qwen35_9b_pa.yaml"
    "e33_gemma4_12b_pa.yaml"
)

echo "Submitting ${#CONFIGS[@]} LLM experiments..."
for config in "${CONFIGS[@]}"; do
    sbatch_out=$(sbatch << SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=araseg-llm
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
echo "=== AraSeg LLM: ${config} ==="
echo "Node:    \$(hostname)"
echo "GPU:     \$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "Venv:    ${VENV}"
echo "Transf.: \$(python -c 'import transformers; print(transformers.__version__)' 2>/dev/null)"
echo "Peft:    \$(python -c 'import peft; print(peft.__version__)' 2>/dev/null)"
echo "Started: \$(date)"
python train.py --config configs/${config}
echo "Finished: \$(date)"
SBATCH_EOF
)
    job_id=$(echo "$sbatch_out" | awk '{print $NF}')
    printf "  %-32s -> job %s\n" "$config" "$job_id"
done

echo ""
echo "Monitor: squeue -u \$USER"
echo "Results: python ${EXPERIMENTS_DIR}/collect_results.py"
