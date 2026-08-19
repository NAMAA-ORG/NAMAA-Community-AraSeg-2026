#!/bin/bash
set -euo pipefail
# Extract POS/dep syntax features for one subtask (array: 4 jobs, one per subtask).
# Each job runs a fresh Python process — avoids Stanza memory accumulation OOM.
#
# Usage (login node):
#   cd ${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments
#   sbatch slurm/run_syntax_extraction_array.sh

#SBATCH --job-name=araseg-syntax
#SBATCH -p kisski
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --array=0-3
#SBATCH --output=${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments/logs/slurm_%A_%a_%x.out
#SBATCH --error=${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments/logs/slurm_%A_%a_%x.err

SUBTASKS=(NoPnx-NP NoPnx-PA NP PA)
SUBTASK=${SUBTASKS[$SLURM_ARRAY_TASK_ID]}

PROJECT=${ARASEG_PROJECT:?set ARASEG_PROJECT}
EXPERIMENTS_DIR=${PROJECT}/experiments
VENV=${PROJECT}/.venv-llm

export HF_HOME=${PROJECT}/../hf_home
export HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export STANZA_RESOURCES_DIR=${PROJECT}/stanza_resources

source ${VENV}/bin/activate
cd ${EXPERIMENTS_DIR}
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "Node: $(hostname) | Started: $(date)"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Subtask: ${SUBTASK} (array task ${SLURM_ARRAY_TASK_ID})"

for SPLIT in train dev test; do
    echo "--- ${SUBTASK} / ${SPLIT} ---"
    python -u extract_syntax.py --subtask "${SUBTASK}" --split "${SPLIT}" --out outputs/syntax_cache --chunk_size 1 --max_words_per_parse 64
done

# Print vocab summary for this subtask only
SUBTASK_STEM=$(printf '%s' "${SUBTASK}" | tr '[:upper:]' '[:lower:]' | tr '-' '_')
VOCAB=outputs/syntax_cache/${SUBTASK_STEM}_vocab.json
if [ -f "${VOCAB}" ]; then
    echo ""
    echo "=== Vocab: ${VOCAB} ==="
    python -c "import json; v=json.load(open('${VOCAB}')); print(f'  n_pos_tags={v[\"n_pos_tags\"]}  n_dep_rels={v[\"n_dep_rels\"]}  mode={v[\"mode\"]}')"
else
    echo "WARNING: ${VOCAB} not found — extract_syntax.py may have failed"
    exit 1
fi

echo ""
echo "Finished: $(date)"
