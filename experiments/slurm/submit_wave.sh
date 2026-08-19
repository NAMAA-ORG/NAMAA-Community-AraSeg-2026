#!/bin/bash
# Submit the exploration wave (e13-e30). Run from anywhere:
#   bash ${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments/slurm/submit_wave.sh
#
# PREREQUISITES (offline cluster — see download note in docs/handoff.md):
#   New encoders (e21-e24) must already be in HF_HOME or jobs fail with
#   HF_HUB_OFFLINE=1. Download on an internet-enabled node first:
#     export HF_HOME=${ARASEG_HF_HOME:-$HOME/.cache/huggingface}
#     huggingface-cli download microsoft/mdeberta-v3-base
#     huggingface-cli download UBC-NLP/MARBERTv2
#     huggingface-cli download aubmindlab/araelectra-base-discriminator
#     huggingface-cli download facebook/xlm-roberta-xl
#   mDeBERTa also needs sentencepiece in the venv:  pip install sentencepiece

PROJECT=${ARASEG_PROJECT:?set ARASEG_PROJECT}
EXPERIMENTS_DIR=${PROJECT}/experiments
DEFAULT_VENV=${PROJECT}/.venv

cd ${EXPERIMENTS_DIR}
mkdir -p logs

# Wave: multi-seed (e13-e20) + new encoders (e21-e24) + window sweep (e25)
# + training-trick A/B on PA (e26-e30).
CONFIGS=(
    # --- multi-seed of the 4 competition winners (ensemble material) ---
    "e13_xlmr_single_pa_s1.yaml"
    "e14_xlmr_single_pa_s2.yaml"
    "e15_xlmr_single_np_s1.yaml"
    "e16_xlmr_single_np_s2.yaml"
    "e17_xlmr_single_nopnxpa_s1.yaml"
    "e18_xlmr_single_nopnxpa_s2.yaml"
    "e19_xlmr_single_nopnxnp_s1.yaml"
    "e20_xlmr_single_nopnxnp_s2.yaml"
    # --- new encoders on PA (our worst gap) ---
    "e21_mdeberta_single_pa.yaml"
    "e22_marbertv2_single_pa.yaml"
    "e23_araelectra_single_pa.yaml"
    "e24_xlmrxl_single_pa.yaml"
    # --- window sweep ---
    "e25_xlmr_single_pa_stride256.yaml"
    # --- training-trick A/B vs e10 on PA ---
    "e26_xlmr_pa_bilstm.yaml"
    "e27_xlmr_pa_focal.yaml"
    "e28_xlmr_pa_fgm.yaml"
    "e29_xlmr_pa_swa.yaml"
    "e30_xlmr_pa_kitchen_sink.yaml"
)

echo "Submitting ${#CONFIGS[@]} experiments..."
for config in "${CONFIGS[@]}"; do
    venv=${DEFAULT_VENV}
    sbatch_out=$(sbatch << SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=araseg-wave
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
echo "=== AraSeg wave: ${config} ==="
echo "Node:    \$(hostname)"
echo "GPU:     \$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "Started: \$(date)"
python train.py --config configs/${config}
echo "Finished: \$(date)"
SBATCH_EOF
)
    job_id=$(echo "$sbatch_out" | awk '{print $NF}')
    printf "  %-40s -> job %s\n" "$config" "$job_id"
done

echo ""
echo "Monitor: squeue -u \$USER"
echo "Results: python ${EXPERIMENTS_DIR}/collect_results.py"
echo ""
echo "After runs finish, build ensembles, e.g.:"
echo "  python ensemble.py --subtask PA --output_dir outputs/ens_pa \\"
echo "      --members configs/e10_xlmr_single_pa_weighted.yaml \\"
echo "                configs/e13_xlmr_single_pa_s1.yaml \\"
echo "                configs/e14_xlmr_single_pa_s2.yaml"
