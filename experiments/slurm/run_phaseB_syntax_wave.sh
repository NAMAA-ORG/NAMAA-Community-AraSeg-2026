#!/bin/bash
# Phase B syntax wave: XLM-R + recall loss + worst-genre + frozen POS/dep features (e56-e59)
# PREREQUISITE: job 14508125 (extract_syntax.py) must be DONE and vocab JSONs present.
# Fill n_pos_tags / n_dep_rels in configs e56-e59 from outputs/syntax_cache/*_vocab.json
# before running this script.
#
# Usage:
#   cd ${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments
#   bash slurm/run_phaseB_syntax_wave.sh

PROJECT=${ARASEG_PROJECT:?set ARASEG_PROJECT}
EXPERIMENTS_DIR=${PROJECT}/experiments
VENV=${PROJECT}/.venv-llm

cd ${EXPERIMENTS_DIR}
mkdir -p logs

CONFIGS=(
    "e56_xlmr_nopnx_np_syntax_s1.yaml"
    "e57_xlmr_nopnx_pa_syntax_s1.yaml"
    "e58_xlmr_np_syntax_s1.yaml"
    "e59_xlmr_pa_syntax_s1.yaml"
)

# Guard: check that n_pos_tags is filled in (not 0)
for config in "${CONFIGS[@]}"; do
    val=$(grep '^n_pos_tags:' configs/${config} | awk '{print $2}')
    if [ "$val" = "0" ]; then
        echo "ERROR: $config still has n_pos_tags: 0 — fill from vocab.json first"
        exit 1
    fi
done

echo "=== Phase B Syntax Wave: ${#CONFIGS[@]} experiments ==="
echo ""

source ${VENV}/bin/activate

echo "Submitting syntax-augmented training jobs..."
for config in "${CONFIGS[@]}"; do
    sbatch_out=$(sbatch << SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=araseg-syntax
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

echo "=== Syntax: ${config} ==="
echo "Node:    \$(hostname)"
echo "GPU:     \$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "Started: \$(date)"

python train.py --config configs/${config}

echo "Finished: \$(date)"
SBATCH_EOF
)
    job_id=$(echo "$sbatch_out" | awk '{print $NF}')
    printf "  %-52s -> job %s\n" "$config" "$job_id"
done

echo ""
echo "Monitor: squeue -u \$USER"
