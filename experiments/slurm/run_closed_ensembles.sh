#!/bin/bash
#SBATCH --job-name=araseg-closed-ens
#SBATCH -p kisski
#SBATCH -G A100:1
#SBATCH --mem-per-gpu=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm_%j_%x.out
#SBATCH --error=logs/slurm_%j_%x.err

# Closed-track-only exhaustive ensemble sweeps.
#
# Submit from the experiments directory on KISSKI:
#   cd ${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments
#   sbatch slurm/run_closed_ensembles.sh
#
# This script does inference only. It loads one member at a time, averages
# per-word boundary probabilities, evaluates every non-empty subset on dev, and
# writes predictions for the dev-selected subset. Do not use open-track data or
# open/alhafni scores for candidate selection.

set -uo pipefail

run() { echo "+ $*"; "$@" || echo "[WARN] STEP FAILED ($?): $*"; }

PROJECT=${ARASEG_PROJECT:?set ARASEG_PROJECT}
EXPERIMENTS_DIR=${PROJECT}/experiments

export HF_HOME=${ARASEG_HF_HOME:-$HOME/.cache/huggingface}
export HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

source ${ARASEG_PROJECT:?set ARASEG_PROJECT}/.venv-llm/bin/activate
cd "${EXPERIMENTS_DIR}"

echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started: $(date)"

C=configs

run python ensemble.py --subtask PA --output_dir outputs/ens_closed_pa_exhaustive \
    --members $C/e34_qwen35_9b_pa_rdrop.yaml $C/e24_xlmrxl_single_pa.yaml \
              $C/e30_xlmr_pa_kitchen_sink.yaml $C/e25_xlmr_single_pa_stride256.yaml \
              $C/e13_xlmr_single_pa_s1.yaml $C/e14_xlmr_single_pa_s2.yaml \
              $C/e23_araelectra_single_pa.yaml $C/e21_mdeberta_single_pa.yaml \
              $C/e22_marbertv2_single_pa.yaml \
    --exhaustive --exhaustive_top_k 20

run python ensemble.py --subtask NoPnx-PA --output_dir outputs/ens_closed_nopnxpa_exhaustive \
    --threshold_policy fixed --threshold 0.5 \
    --members $C/e41_qwen35_9b_nopnx_pa.yaml $C/e38_xlmr_multitask_joint.yaml \
              $C/e11_xlmr_single_nopnxpa_w2.yaml $C/e17_xlmr_single_nopnxpa_s1.yaml \
              $C/e18_xlmr_single_nopnxpa_s2.yaml \
    --exhaustive --exhaustive_top_k 20

run python ensemble.py --subtask NP --output_dir outputs/ens_closed_np_exhaustive \
    --members $C/e40_qwen35_9b_np.yaml $C/e38_xlmr_multitask_joint.yaml \
              $C/e8_xlmr_single_np.yaml $C/e15_xlmr_single_np_s1.yaml \
              $C/e16_xlmr_single_np_s2.yaml \
    --exhaustive --exhaustive_top_k 20

run python ensemble.py --subtask NoPnx-NP --output_dir outputs/ens_closed_nopnxnp_exhaustive \
    --threshold_policy fixed --threshold 0.5 \
    --members $C/e42_qwen35_9b_nopnx_np.yaml $C/e38_xlmr_multitask_joint.yaml \
              $C/e9_xlmr_single_nopnxnp.yaml $C/e12_xlmr_single_nopnxnp_w2.yaml \
              $C/e19_xlmr_single_nopnxnp_s1.yaml $C/e20_xlmr_single_nopnxnp_s2.yaml \
    --exhaustive --exhaustive_top_k 20

echo "=== Local official eval.py cross-check ==="
run python eval.py --task PA       --predictions outputs/ens_closed_pa_exhaustive/test_predictions_PA.csv             --split test
run python eval.py --task NoPnx-PA --predictions outputs/ens_closed_nopnxpa_exhaustive/test_predictions_NoPnx_PA.csv  --split test
run python eval.py --task NP       --predictions outputs/ens_closed_np_exhaustive/test_predictions_NP.csv             --split test
run python eval.py --task NoPnx-NP --predictions outputs/ens_closed_nopnxnp_exhaustive/test_predictions_NoPnx_NP.csv  --split test

echo "Finished: $(date)"
echo "Review exhaustive_sweep_*.json and ensemble_*.json under outputs/ens_closed_*_exhaustive/"
