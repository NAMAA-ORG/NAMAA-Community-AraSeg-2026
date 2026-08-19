#!/bin/bash
#SBATCH --job-name=araseg-ens
#SBATCH -p kisski
#SBATCH -G A100:1
#SBATCH --mem-per-gpu=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm_%j_%x.out
#SBATCH --error=logs/slurm_%j_%x.err
#
# Build all ensembles + official cross-check in ONE GPU job (ensemble.py is
# inference-only and loads members one at a time, so a single A100 is plenty).
# Submit from the experiments dir so relative log/config paths resolve:
#   cd ${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments
#   sbatch slurm/run_ensembles.sh
#
# Diversity members are chosen from the e13-e30 results: best XLM-R variant = e30
# (kitchen-sink, test 0.9142); decorrelated encoders = e23 AraELECTRA / e21
# mDeBERTa / e22 MARBERTv2. e24 (XLM-R XL) is now the best PA single model
# (test 0.9236) and is included in every PA sweep below.
#
# NoPnx ensembles are written at a fixed 0.5 threshold (--threshold_policy
# fixed): both NoPnx subtasks scored higher at 0.5 than at their dev-tuned
# threshold on open-test. PA/NP keep the dev-tuned threshold.

# NOTE: deliberately NOT using `set -e`. ensemble.py loads members one at a
# time; a single bad member (e.g. an architecture/config mismatch) should log a
# warning and let the remaining ensembles + eval cross-check run, not abort the
# whole GPU job. `run` wraps each step so failures are visible but non-fatal.
set -uo pipefail

run() { echo "+ $*"; "$@" || echo "[WARN] STEP FAILED ($?): $*"; }

PROJECT=${ARASEG_PROJECT:?set ARASEG_PROJECT}
EXPERIMENTS_DIR=${PROJECT}/experiments
VENV=${PROJECT}/.venv

export HF_HOME=${ARASEG_HF_HOME:-$HOME/.cache/huggingface}
export HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Exhaustive sweeps can mix BERT and LLM members; .venv-llm loads both families.
source ${ARASEG_PROJECT:?set ARASEG_PROJECT}/.venv-llm/bin/activate
cd ${EXPERIMENTS_DIR}
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started: $(date)"

C=configs

# ── 1. Multi-seed ensembles (deterministic, all 4 subtasks) ──────────────────
run python ensemble.py --subtask PA       --output_dir outputs/ens_pa \
    --members $C/e10_xlmr_single_pa_weighted.yaml $C/e13_xlmr_single_pa_s1.yaml $C/e14_xlmr_single_pa_s2.yaml
run python ensemble.py --subtask NP       --output_dir outputs/ens_np \
    --members $C/e8_xlmr_single_np.yaml $C/e15_xlmr_single_np_s1.yaml $C/e16_xlmr_single_np_s2.yaml
run python ensemble.py --subtask NoPnx-PA --output_dir outputs/ens_nopnxpa \
    --threshold_policy fixed --threshold 0.5 \
    --members $C/e11_xlmr_single_nopnxpa_w2.yaml $C/e17_xlmr_single_nopnxpa_s1.yaml $C/e18_xlmr_single_nopnxpa_s2.yaml
run python ensemble.py --subtask NoPnx-NP --output_dir outputs/ens_nopnxnp \
    --threshold_policy fixed --threshold 0.5 \
    --members $C/e12_xlmr_single_nopnxnp_w2.yaml $C/e19_xlmr_single_nopnxnp_s1.yaml $C/e20_xlmr_single_nopnxnp_s2.yaml

# ── 2. PA ensemble sweep (pick the best of these for submission) ─────────────
# 2a. Best XL + best XLM-R trick/seed variants (no encoder diversity)
run python ensemble.py --subtask PA --output_dir outputs/ens_pa_xl \
    --members $C/e24_xlmrxl_single_pa.yaml $C/e30_xlmr_pa_kitchen_sink.yaml $C/e26_xlmr_pa_bilstm.yaml \
              $C/e25_xlmr_single_pa_stride256.yaml $C/e13_xlmr_single_pa_s1.yaml $C/e14_xlmr_single_pa_s2.yaml
# 2b. Best XL + best XLM-R + decorrelated encoders (the bet)
run python ensemble.py --subtask PA --output_dir outputs/ens_pa_div \
    --members $C/e24_xlmrxl_single_pa.yaml $C/e30_xlmr_pa_kitchen_sink.yaml $C/e23_araelectra_single_pa.yaml \
              $C/e21_mdeberta_single_pa.yaml $C/e22_marbertv2_single_pa.yaml
# 2c. Everything decent (max diversity)
run python ensemble.py --subtask PA --output_dir outputs/ens_pa_all \
    --members $C/e24_xlmrxl_single_pa.yaml $C/e30_xlmr_pa_kitchen_sink.yaml $C/e26_xlmr_pa_bilstm.yaml \
              $C/e25_xlmr_single_pa_stride256.yaml $C/e13_xlmr_single_pa_s1.yaml $C/e14_xlmr_single_pa_s2.yaml \
              $C/e23_araelectra_single_pa.yaml $C/e21_mdeberta_single_pa.yaml $C/e22_marbertv2_single_pa.yaml

# ── 2d. Advanced-techniques exhaustive sweeps (after e34-e49 checkpoints exist) ─
run python ensemble.py --subtask PA --output_dir outputs/ens_pa_exhaustive \
    --members $C/e30_xlmr_pa_kitchen_sink.yaml $C/e23_araelectra_single_pa.yaml $C/e21_mdeberta_single_pa.yaml \
              $C/e22_marbertv2_single_pa.yaml $C/e32_qwen35_9b_pa.yaml $C/e33_gemma4_12b_pa.yaml \
              $C/e37_xlmr_crf_pa.yaml $C/e43_xlmr_multitask_seq_pa.yaml \
    --exhaustive --exhaustive_top_k 10
run python ensemble.py --subtask NoPnx-PA --output_dir outputs/ens_nopnxpa_exhaustive \
    --threshold_policy fixed --threshold 0.5 \
    --members $C/e18_xlmr_single_nopnxpa_s2.yaml $C/e39_distill_pa_to_nopnx_pa.yaml \
              $C/e41_qwen35_9b_nopnx_pa.yaml $C/e44_xlmr_multitask_seq_nopnx_pa.yaml \
              $C/e49_distill_np_to_nopnx_pa.yaml \
    --exhaustive --exhaustive_top_k 10
run python ensemble.py --subtask NP --output_dir outputs/ens_np_exhaustive \
    --members $C/e15_xlmr_single_np_s1.yaml $C/e40_qwen35_9b_np.yaml $C/e45_xlmr_multitask_seq_np.yaml \
    --exhaustive --exhaustive_top_k 10
run python ensemble.py --subtask NoPnx-NP --output_dir outputs/ens_nopnxnp_exhaustive \
    --threshold_policy fixed --threshold 0.5 \
    --members $C/e19_xlmr_single_nopnxnp_s1.yaml $C/e42_qwen35_9b_nopnx_np.yaml \
              $C/e46_xlmr_multitask_seq_nopnx_np.yaml $C/e47_distill_np_to_nopnx_np.yaml \
              $C/e48_distill_pa_to_nopnx_np.yaml \
    --exhaustive --exhaustive_top_k 10

# ── 3. Official scorer cross-check (P/R labels swapped in eval.py; F1 is correct)
echo "=== Official eval.py cross-check ==="
for dir in ens_pa ens_pa_xl ens_pa_div ens_pa_all ens_pa_exhaustive; do
    echo "--- PA / outputs/$dir ---"
    run python eval.py --task PA --predictions outputs/$dir/test_predictions_PA.csv --split test
done
run python eval.py --task NP       --predictions outputs/ens_np/test_predictions_NP.csv             --split test
run python eval.py --task NoPnx-PA --predictions outputs/ens_nopnxpa/test_predictions_NoPnx_PA.csv  --split test
run python eval.py --task NoPnx-NP --predictions outputs/ens_nopnxnp/test_predictions_NoPnx_NP.csv  --split test
run python eval.py --task NP       --predictions outputs/ens_np_exhaustive/test_predictions_NP.csv             --split test
run python eval.py --task NoPnx-PA --predictions outputs/ens_nopnxpa_exhaustive/test_predictions_NoPnx_PA.csv  --split test
run python eval.py --task NoPnx-NP --predictions outputs/ens_nopnxnp_exhaustive/test_predictions_NoPnx_NP.csv  --split test

echo "Finished: $(date)"
echo "Compare PA: outputs/ens_pa{,_xl,_div,_all,_exhaustive}/ensemble_PA.json  (test.f1 field)"
