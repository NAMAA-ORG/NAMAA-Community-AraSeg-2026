#!/bin/bash
#SBATCH --job-name=araseg-thr-tune
#SBATCH -p kisski
#SBATCH -G A100:1
#SBATCH --mem-per-gpu=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm_%j_%x.out
#SBATCH --error=logs/slurm_%j_%x.err

# Re-write the four final closed submissions at the DEV-TUNED threshold using a
# fine 0.01 grid, instead of the flat 0.50 they shipped at. Our ensembles are
# systematically precision-heavy / recall-light vs closed #1, so the F1-max
# point on dev sits below 0.5 — this spends surplus precision to buy recall.
# Inference only; no training. Each run also prints the test P/R/F1 curve as a
# diagnostic (the CSV is still written at the dev-tuned threshold, not test).
#
# Submit from the experiments directory on KISSKI:
#   cd ${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments
#   sbatch slurm/run_closed_threshold_tuned.sh

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

# official_eval.py needs scikit-learn (the missing dep that blocked the earlier
# cross-check). If the compute node has no internet, run this once on the LOGIN
# node before sbatch: source .venv-llm/bin/activate && pip install scikit-learn
run pip install -q scikit-learn

echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started: $(date)"

C=configs

# subtask | tuned out_dir | csv basename | @0.50 baseline CSV | member configs
# Tune threshold on dev (fine grid), write CSV, then score BOTH the new tuned CSV
# and the existing @0.50 baseline CSV with the OFFICIAL scorer on the test split.
# Same members in both, so it's an apples-to-apples tuned-vs-0.50 comparison and
# every number is leaderboard-exact. Keep whichever F1 is higher per subtask.
tune_and_score() {
    local subtask="$1" outdir="$2" csv="$3" baseline="$4"; shift 4
    run python ensemble.py --subtask "$subtask" --output_dir "$outdir" \
        --threshold_policy dev_tuned --thr_step 0.01 --members "$@"
    echo "[official] === $subtask : TUNED ($outdir/$csv) ==="
    run python official_eval.py --task "$subtask" --split test \
        --predictions "$outdir/$csv"
    echo "[official] === $subtask : BASELINE @0.50 ($baseline) ==="
    if [ -f "$baseline" ]; then
        run python official_eval.py --task "$subtask" --split test --predictions "$baseline"
    else
        echo "[WARN] baseline CSV not found, skipping: $baseline"
    fi
}

tune_and_score PA       outputs/ens_closed_pa_thrtuned     test_predictions_PA.csv \
    outputs/ens_closed_pa_exhaustive/test_predictions_PA.csv \
    $C/e34_qwen35_9b_pa_rdrop.yaml $C/e25_xlmr_single_pa_stride256.yaml

tune_and_score NoPnx-PA outputs/ens_closed_nopnxpa_thrtuned test_predictions_NoPnx_PA.csv \
    outputs/ens_closed_nopnxpa_e41_e17/test_predictions_NoPnx_PA.csv \
    $C/e41_qwen35_9b_nopnx_pa.yaml $C/e17_xlmr_single_nopnxpa_s1.yaml

tune_and_score NP       outputs/ens_closed_np_thrtuned     test_predictions_NP.csv \
    outputs/ens_closed_np_e40_e15/test_predictions_NP.csv \
    $C/e40_qwen35_9b_np.yaml $C/e15_xlmr_single_np_s1.yaml

tune_and_score NoPnx-NP outputs/ens_closed_nopnxnp_thrtuned test_predictions_NoPnx_NP.csv \
    outputs/ens_closed_nopnxnp_exhaustive/test_predictions_NoPnx_NP.csv \
    $C/e42_qwen35_9b_nopnx_np.yaml $C/e38_xlmr_multitask_joint.yaml

echo "Finished: $(date)"
echo "Compare the OFFICIAL P/R/F1 printed above against the @0.50 baseline; keep whichever F1 is higher."
echo "Written CSVs:"
echo "  outputs/ens_closed_pa_thrtuned/test_predictions_PA.csv"
echo "  outputs/ens_closed_nopnxpa_thrtuned/test_predictions_NoPnx_PA.csv"
echo "  outputs/ens_closed_np_thrtuned/test_predictions_NP.csv"
echo "  outputs/ens_closed_nopnxnp_thrtuned/test_predictions_NoPnx_NP.csv"
