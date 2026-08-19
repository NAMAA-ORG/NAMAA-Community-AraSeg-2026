#!/bin/bash
# Blind Test inference — all four frozen locks on the Official Blind Test set.
# One A100 job: generate the blind member/sat caches, then run each head's locked
# combiner at its FROZEN threshold with --test-split blind. No training, no re-tuning,
# no re-selection. Writes four submission CSVs; package + upload from the login node.
#
# KISSKI has everything already: member checkpoints in outputs/<id>, bases + blind
# data in hf_home, and the OOF/sat/prob DEV caches from earlier runs. So NO downloads,
# and (unlike off-cluster) NO ARASEG_CKPT_ROOT / ARASEG_BASE_REMAP — native paths.
#
# PREREQ (login node, has internet): the four MBZUAI *-Blind datasets must be cached
# into $HF_HOME with the ORGANIZERS' token, or the offline job can't load them:
#   export HF_HOME=$PROJECT/hf_home
#   export HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets
#   export ARASEG_BLIND_TOKEN=$(cat ~/.araseg_blind_token)
#   python -c "from datasets import load_dataset as L; [L(f'MBZUAI/AraSeg-2026-Shared-Task-{t}-Blind', token=__import__('os').environ['ARASEG_BLIND_TOKEN']) for t in ['PA','NP','NoPnx-PA','NoPnx-NP']]"
#
# Usage (login node):
#   cd ${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments
#   sbatch slurm/run_blind_all.sh

#SBATCH --job-name=araseg-blind-all
#SBATCH -p kisski
#SBATCH -G A100:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments/logs/slurm_%A_%x.out
#SBATCH --error=${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments/logs/slurm_%A_%x.err

set -euo pipefail

export HF_HOME=${ARASEG_HF_HOME:-$HOME/.cache/huggingface}
export HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
# Blind datasets carry the organizers' token; under offline it's read from cache and
# the token is unused, but set it so an accidental online path still authenticates.
export ARASEG_BLIND_TOKEN=$(cat ~/.araseg_blind_token 2>/dev/null || true)
# NOTE: deliberately NOT setting ARASEG_CKPT_ROOT / ARASEG_BASE_REMAP — KISSKI native.

source ${ARASEG_PROJECT:?set ARASEG_PROJECT}/.venv-llm/bin/activate
cd ${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments

echo "Node: $(hostname) | Started: $(date)"

SAT_CKPT=outputs/sat_ft         # <head>/checkpoint-*/model.safetensors per head
OOF=outputs/oof                 # OOF train matrices (fit decoder/stack — present)
SAT_OOF=outputs/oof_sat         # sat_ft OOF folds (present)
SAT_CACHE=outputs/sat_fullft_caches   # sat_ft dev (present) + blind (made below)
PCACHE=outputs/prob_cache       # member dev (present) + blind (made below)

# Member config lists (lock order) ------------------------------------------------
PA="configs/e25_xlmr_single_pa_stride256.yaml configs/e32_qwen35_9b_pa.yaml configs/e33_gemma4_12b_pa.yaml"
NP="configs/e40_qwen35_9b_np.yaml configs/e15_xlmr_single_np_s1.yaml configs/e16_xlmr_single_np_s2.yaml configs/e38_xlmr_multitask_joint.yaml"
NOPNX_PA="configs/e41_qwen35_9b_nopnx_pa.yaml configs/e17_xlmr_single_nopnxpa_s1.yaml configs/e18_xlmr_single_nopnxpa_s2.yaml configs/e11_xlmr_single_nopnxpa_w2.yaml configs/e38_xlmr_multitask_joint.yaml"
NOPNX_NP="configs/e42_qwen35_9b_nopnx_np.yaml configs/e19_xlmr_single_nopnxnp_s1.yaml configs/e20_xlmr_single_nopnxnp_s2.yaml configs/e12_xlmr_single_nopnxnp_w2.yaml configs/e38_xlmr_multitask_joint.yaml"

# --- Step 1: sat_ft blind caches (all heads; frozen checkpoint, no retraining) ---
echo "=== [1] sat_ft blind caches ==="
python sat_finetune.py --eval-checkpoint "$SAT_CKPT" --splits blind --cache-dir "$SAT_CACHE" \
    || echo "!!! sat_ft blind caching FAILED — heads needing sat_ft will fail below"

# --- Step 2: member blind prob caches for the two NoPnx decoders -----------------
echo "=== [2] NoPnx member blind prob caches ==="
python cache_probs.py --subtask NoPnx-PA --splits blind --members $NOPNX_PA --out_dir "$PCACHE" \
    || echo "!!! NoPnx-PA cache_probs FAILED"
python cache_probs.py --subtask NoPnx-NP --splits blind --members $NOPNX_NP --out_dir "$PCACHE" \
    || echo "!!! NoPnx-NP cache_probs FAILED"

# --- Step 3: write the four blind submissions at their FROZEN thresholds ----------
echo "=== [3a] PA — logit e25 e32 e33 @0.25 ==="
python ensemble.py --subtask PA --members $PA --output_dir outputs/blind_pa \
    --combine logit --threshold_policy fixed --threshold 0.25 --test-split blind \
    || echo "!!! PA FAILED"

echo "=== [3b] NP — SaT-OOF-stack e40 e15 e16 e38 + sat_ft @0.36 ==="
python fit_oof_stack.py --subtask NP --members $NP --output_dir outputs/blind_np \
    --oof_dir "$OOF" --sat_oof_dir "$SAT_OOF" --sat_cache_dir "$SAT_CACHE" \
    --threshold_policy fixed --threshold 0.36 --test-split blind \
    || echo "!!! NP FAILED"

echo "=== [3c] NoPnx-PA — decoder e41 e17 e18 e11 e38 + sat_ft @0.49 ==="
python fit_decoder.py --subtask NoPnx-PA --members $NOPNX_PA --output_dir outputs/blind_nopnx_pa \
    --oof_dir "$OOF" --sat_oof_dir "$SAT_OOF" --sat_cache_dir "$SAT_CACHE" \
    --cache_dir "$PCACHE" --test-split blind \
    || echo "!!! NoPnx-PA FAILED"

echo "=== [3d] NoPnx-NP — decoder e42 e19 e20 e12 e38 + sat_ft @0.38 ==="
python fit_decoder.py --subtask NoPnx-NP --members $NOPNX_NP --output_dir outputs/blind_nopnx_np \
    --oof_dir "$OOF" --sat_oof_dir "$SAT_OOF" --sat_cache_dir "$SAT_CACHE" \
    --cache_dir "$PCACHE" --test-split blind \
    || echo "!!! NoPnx-NP FAILED"

echo "Finished: $(date)"
echo "CSVs:"
echo "  outputs/blind_pa/test_predictions_PA.csv                 (expect 100 rows)"
echo "  outputs/blind_np/test_predictions_NP.csv                 (expect 212 rows)"
echo "  outputs/blind_nopnx_pa/test_predictions_NoPnx_PA.csv     (expect 100 rows)"
echo "  outputs/blind_nopnx_np/test_predictions_NoPnx_NP.csv     (expect 212 rows)"
echo "VERIFY the decoders reprinted thr 0.49 / 0.38 and weights match decoder_*.json."
echo "Package each on the login node: cp CSV prediction && zip prediction.zip prediction"
