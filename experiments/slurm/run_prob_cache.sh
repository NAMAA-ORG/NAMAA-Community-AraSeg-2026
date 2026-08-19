#!/bin/bash
# Queue A1/A2 — cache every locked member's dev+test per-word probabilities.
#
# This is the LAST GPU step before the deadline work: one forward pass per
# (member, split), pickled to outputs/prob_cache/. After it lands, the decoder
# (fit_decoder.py), the paired bootstrap (bootstrap_paired.py), threshold sweeps
# and official scoring all run on CPU — on a login node or off-cluster — because
# they only ever read those arrays. No A100 for arithmetic.
#
# One job per head (members are sequential inside a job; the Qwen/Gemma members
# dominate the runtime). Idempotent: cache_probs.py skips files that exist.
#
# Submit (login node):
#   cd ${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments
#   bash slurm/run_prob_cache.sh                # all 4 heads
#   bash slurm/run_prob_cache.sh nopnx-pa       # just the least-secure head
set -euo pipefail

PROJECT=${ARASEG_PROJECT:?set ARASEG_PROJECT}
EXPERIMENTS_DIR=${PROJECT}/experiments
VENV=${PROJECT}/.venv-llm
cd ${EXPERIMENTS_DIR}
mkdir -p logs outputs/prob_cache

ALL_HEADS=(pa np nopnx-pa nopnx-np)
HEADS=("$@"); [ ${#HEADS[@]} -eq 0 ] && HEADS=("${ALL_HEADS[@]}")

declare -A SUBTASK=( [pa]=PA [np]=NP [nopnx-pa]=NoPnx-PA [nopnx-np]=NoPnx-NP )

# Exactly the members of each locked ensemble (handoff "Final locked submissions").
# sat_ft is NOT here — it already has its own caches in outputs/sat_fullft_caches.
declare -A MEMBERS
MEMBERS[pa]="e25_xlmr_single_pa_stride256 e32_qwen35_9b_pa e33_gemma4_12b_pa"
MEMBERS[np]="e40_qwen35_9b_np e15_xlmr_single_np_s1 e16_xlmr_single_np_s2 e38_xlmr_multitask_joint"
MEMBERS[nopnx-pa]="e41_qwen35_9b_nopnx_pa e17_xlmr_single_nopnxpa_s1 e18_xlmr_single_nopnxpa_s2 e11_xlmr_single_nopnxpa_w2 e38_xlmr_multitask_joint"
MEMBERS[nopnx-np]="e42_qwen35_9b_nopnx_np e19_xlmr_single_nopnxnp_s1 e20_xlmr_single_nopnxnp_s2 e12_xlmr_single_nopnxnp_w2 e38_xlmr_multitask_joint"

for h in "${HEADS[@]}"; do
    [ -n "${SUBTASK[$h]:-}" ] || { echo "unknown head '$h' (choose: ${ALL_HEADS[*]})"; exit 1; }
done

for h in "${HEADS[@]}"; do
    st=${SUBTASK[$h]}
    members=""
    for m in ${MEMBERS[$h]}; do members="${members} configs/${m}.yaml"; done
    out=$(sbatch <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=probcache-${h}
#SBATCH -A kisski-futurmig
#SBATCH -p kisski
#SBATCH -G A100:1
#SBATCH --mem-per-gpu=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=${EXPERIMENTS_DIR}/logs/slurm_%j_%x.out
#SBATCH --error=${EXPERIMENTS_DIR}/logs/slurm_%j_%x.err
export HF_HOME=${ARASEG_HF_HOME:-$HOME/.cache/huggingface}
export HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
source ${VENV}/bin/activate
cd ${EXPERIMENTS_DIR}
python cache_probs.py --subtask ${st} --members ${members} \
    --splits dev test --out_dir outputs/prob_cache
SBATCH_EOF
)
    printf "  %-9s prob cache -> job %s\n" "$st" "${out##* }"
done

cat <<'EOF'

=== next (CPU only, no sbatch needed) ===
  # 1. uncertainty on the current locks
  python bootstrap_paired.py --subtask NoPnx-PA --split test \
      --a outputs/oof_stack_sat_nopnx_pa/test_predictions_NoPnx_PA.csv

  # 2. train-only structural decoder, least-secure head first
  python fit_decoder.py --subtask NoPnx-PA \
      --members configs/e41_qwen35_9b_nopnx_pa.yaml configs/e17_xlmr_single_nopnxpa_s1.yaml \
                configs/e18_xlmr_single_nopnxpa_s2.yaml configs/e11_xlmr_single_nopnxpa_w2.yaml \
                configs/e38_xlmr_multitask_joint.yaml \
      --oof_dir outputs/oof --sat_oof_dir outputs/oof_sat \
      --cache_dir outputs/prob_cache --sat_cache_dir outputs/sat_fullft_caches \
      --output_dir outputs/decoder_nopnx_pa

  # 3. gate the re-lock on paired evidence, not the aggregate delta
  python bootstrap_paired.py --subtask NoPnx-PA --split test \
      --a outputs/oof_stack_sat_nopnx_pa/test_predictions_NoPnx_PA.csv \
      --b outputs/decoder_nopnx_pa/test_predictions_NoPnx_PA.csv
EOF
