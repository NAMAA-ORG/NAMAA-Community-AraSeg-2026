#!/bin/bash
set -euo pipefail
# Extract POS/dep syntax features for all 4 subtasks (train/dev/test).
# Uses stanza Arabic model already downloaded to stanza_resources/.
# Outputs: outputs/syntax_cache/{subtask}_{split}.pkl + {subtask}_vocab.json
#
# Usage (login node):
#   cd ${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments
#   sbatch slurm/run_syntax_extraction.sh

#SBATCH --job-name=araseg-syntax
#SBATCH -p kisski
# No -G: job is CPU-only (GPU hidden via CUDA_VISIBLE_DEVICES, stanza use_gpu=False).
# Requesting a GPU just makes it wait in the GPU queue for hardware it never uses.
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments/logs/slurm_%j_%x.out
#SBATCH --error=${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments/logs/slurm_%j_%x.err

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

# Hide GPU before any Python process starts — datasets imports torch on load,
# initialising a CUDA context that consumes system RAM before stanza even runs.
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "Node: $(hostname) | Started: $(date)"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

for SUBTASK in NoPnx-NP NoPnx-PA NP PA; do
    echo ""
    echo "=== ${SUBTASK} ==="
    for SPLIT in train dev test; do
        echo "--- ${SUBTASK} / ${SPLIT} ---"
        python -u extract_syntax.py --subtask "${SUBTASK}" --split "${SPLIT}" --out outputs/syntax_cache --chunk_size 1 --max_words_per_parse 64 --max_words_per_parse 64
    done
done

echo ""
echo "=== Vocab summary ==="
for f in outputs/syntax_cache/*_vocab.json; do
    echo "${f}:"
    python -c "import json,sys; v=json.load(open('${f}')); print(f'  n_pos_tags={v[\"n_pos_tags\"]}  n_dep_rels={v[\"n_dep_rels\"]}  mode={v[\"mode\"]}')"
done

echo ""
echo "Finished: $(date)"
