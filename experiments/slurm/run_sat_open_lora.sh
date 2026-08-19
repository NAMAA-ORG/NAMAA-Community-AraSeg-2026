#!/bin/bash
# OPEN TRACK ONLY. wtpsplit's own pretrained ud/opus100/ersatz LoRA adapters.
#
# Why this is the cheapest open-track lever we have:
#   * open track has NO data restrictions (organizer rules page), and we have never
#     submitted anything open-specific -- every open entry is a byte-identical re-upload
#     of the closed system.
#   * these adapters are closed-track ILLEGAL (trained on non-AraSeg data), which is
#     exactly why they are free money in open. sat_segment.py has documented that since
#     Phase D; the flag to actually use them was only wired on 2026-07-30.
#   * no training. Inference only, then the existing ensemble machinery.
#
# Open-track gaps this targets: PA -0.3 (omar_saqr 94.7 vs our 94.4) and NoPnx-NP, where
# mohamed_mohamed tied us at 87.0 on 07-27 and took the tie-break. Smallest gaps on the
# whole board.
#
# Caches are written with an OPENONLY_ prefix so they can never be folded into a closed
# lock by accident.
#
# PRE-FLIGHT (login node, HAS network; the compute nodes do not). This both discovers
# which Arabic adapters exist and warms the offline cache. The token in this environment
# is stale and 401s on public repos, so strip it -- these are public downloads:
#
#   env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN HF_HUB_OFFLINE=0 python - <<'PY'
#   from wtpsplit import SaT
#   for style in ("ud", "opus100", "ersatz"):
#       try:
#           SaT("sat-3l", style_or_domain=style, language="ar")
#           print("OK  ", style)
#       except Exception as e:
#           print("FAIL", style, type(e).__name__, str(e)[:160])
#   PY
#
# Then set STYLES below to whatever printed OK, and submit:
#   sbatch slurm/run_sat_open_lora.sh

#SBATCH --job-name=araseg-sat-open
#SBATCH -p kisski
#SBATCH -G A100:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments/logs/slurm_%A_%x.out
#SBATCH --error=${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments/logs/slurm_%A_%x.err

set -euo pipefail

export HF_HOME=${ARASEG_HF_HOME:-$HOME/.cache/huggingface}
export HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
# The shared hf_home carries a stale token file that 401s even on public repos. Offline
# should never authenticate, but this makes sure of it rather than trusting that.
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1

source ${ARASEG_PROJECT:?set ARASEG_PROJECT}/.venv-llm/bin/activate
cd ${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments

# Set from the pre-flight output above. Anything not cached will fail fast offline.
STYLES="${STYLES:-ud opus100 ersatz}"
MODEL="${MODEL:-sat-3l}"
CACHE=outputs/sat_open_lora

echo "Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) | Started: $(date)"
echo "model=$MODEL styles=$STYLES"

# Alignment sanity, on the FIRST STYLE's real configuration rather than a proxy for it.
# Graded against chance, so it does not encode the quality of whichever model tuned it.
FIRST_STYLE=$(echo $STYLES | awk '{print $1}')
python sat_segment.py --selfcheck --model "$MODEL" \
    --style_or_domain "$FIRST_STYLE" --language ar

for STYLE in $STYLES; do
    for SPLIT in dev test; do
        echo "=== style=$STYLE split=$SPLIT ==="
        python sat_segment.py --model "$MODEL" \
            --style_or_domain "$STYLE" --language ar \
            --split "$SPLIT" --save-cache "$CACHE" \
            || echo "FAILED $STYLE/$SPLIT -- traceback in logs/slurm_${SLURM_JOB_ID}_${SLURM_JOB_NAME}.err"
    done
done

echo "--- caches written ---"
ls -la "$CACHE" || true
cat <<'NOTE'

READ THE STANDALONE NUMBERS FIRST. The zero-shot SaT ceiling here is ~0.68 and was
confirmed twice (ours + Kareem's); the corpus is line-structured short-unit text
(median segment 9 words, exam worksheets and subtitle dialogue), not the prose these
adapters were trained on. If the adapters land near 0.68 too, they will not help the
ensemble either and this lane closes cheaply.

If a style is clearly better, THEN ensemble it -- as an added member on open PA
(our lock e25+e32+e33 has no SaT member at all) and open NoPnx-NP. Gate on dev, then
bootstrap_paired.py against the locked CSV, CI must exclude 0. Same discipline as
closed; the only thing that changes is which data is legal.
NOTE
echo "Finished at $(date)"
