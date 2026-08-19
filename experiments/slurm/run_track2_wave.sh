#!/bin/bash
# Track 2 (Phase D): punctuation-dropout augmentation + aux punctuation head.
# e60/e61 = aux head on exact NoPnx (vs e50/e51 baselines).
# e62/e63 = aux head + partial punctuation dropout (drop_p=0.5 continuum).
# Reuses Phase B genre proxy maps (skipped if present).
#
# Usage:
#   cd ${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments
#   bash slurm/run_track2_wave.sh

PROJECT=${ARASEG_PROJECT:?set ARASEG_PROJECT}
EXPERIMENTS_DIR=${PROJECT}/experiments
VENV=${PROJECT}/.venv-llm

cd ${EXPERIMENTS_DIR}
mkdir -p logs outputs/genre_proxy

CONFIGS=(
    "e60_xlmr_nopnx_np_aux_s1.yaml"
    "e61_xlmr_nopnx_pa_aux_s1.yaml"
    "e62_xlmr_nopnx_np_aux_drop05_s1.yaml"
    "e63_xlmr_nopnx_pa_aux_drop05_s1.yaml"
)

echo "=== Track 2 Wave: ${#CONFIGS[@]} experiments ==="

source ${VENV}/bin/activate

# ── Gate: Track 2 wiring smoke test (CPU, seconds) ─────────────────────────
echo "Running Track 2 smoke test..."
python smoke_test_track2.py || { echo "SMOKE TEST FAILED — aborting wave"; exit 1; }
echo ""

# ── Genre proxy maps (reused from Phase B; built if missing) ───────────────
for SUBTASK in NoPnx-NP NoPnx-PA; do
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

echo "Submitting training jobs..."
for config in "${CONFIGS[@]}"; do
    sbatch_out=$(sbatch << SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=araseg-track2
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

echo "=== Track 2: ${config} ==="
echo "Node:    \$(hostname)"
echo "Started: \$(date)"

python train.py --config configs/${config}

echo "Finished: \$(date)"
SBATCH_EOF
)
    job_id=$(echo "$sbatch_out" | awk '{print $NF}')
    printf "  %-44s -> job %s\n" "$config" "$job_id"
done

echo ""
echo "Monitor: squeue -u \$USER"
echo "Compare: e60 vs e50 (NoPnx-NP), e61 vs e51 (NoPnx-PA) — isolates the aux head."
echo "         e62/e63 add partial-punctuation dropout on top."
