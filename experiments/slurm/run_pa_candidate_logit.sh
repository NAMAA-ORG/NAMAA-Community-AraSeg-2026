#!/bin/bash
# PA candidate sweep — does anything from the July-26 wave improve the PA logit lock?
#
# The PA lock (§3.1) is a `logit` ensemble of e25/e32/e33 with NO learned weights, so
# unlike the two NoPnx decoder heads it needs no OOF matrix — a candidate only has to
# be forward-passed on dev+test. That makes PA the cheap head to ask this on.
#
# Candidates, all judged on ENSEMBLE delta, never on their standalone F1:
#   e73  Qwen3.5-27B      test 0.9392  — best single we have, but +0.17 over e80 while
#                                        the 9B recipe's own seed spread is +0.41
#   e74  Qwen2.5-14B-Inst test 0.9173  — cross-generation diversity probe
#   e80  Qwen3.5-9B s2    test 0.9375  — second seed of lock member e32
#   e84  Gemma-4-12B s2   test 0.9222  — second seed of lock member e33
#
# --exhaustive fits every non-empty subset and DEV picks the winner (legal model
# selection); the test table is printed for eyeballing only. The re-lock decision is
# the paired bootstrap at the bottom, not the dev winner and not the point estimate.
#
# Submit from ${PROJECT}/experiments:
#   sbatch slurm/run_pa_candidate_logit.sh
#SBATCH --job-name=araseg-pa-candidate-logit
#SBATCH -A kisski-futurmig
#SBATCH -p kisski
#SBATCH -G A100:1
#SBATCH --mem-per-gpu=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments/logs/slurm_%j_%x.out
#SBATCH --error=${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments/logs/slurm_%j_%x.err
set -uo pipefail

PROJECT=${ARASEG_PROJECT:?set ARASEG_PROJECT}
export HF_HOME=${ARASEG_HF_HOME:-$HOME/.cache/huggingface}
export HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Qwen3.5 members (e32/e73/e80) need .venv-llm's newer transformers; plain .venv does
# not recognise the qwen3_5 architecture. Same column run_lane_a_recovery.sh learned.
source ${PROJECT}/.venv-llm/bin/activate
cd ${PROJECT}/experiments
mkdir -p logs

OUT=outputs/pa_candidate_logit

echo "=== PA candidate logit sweep ==="
echo "Node: $(hostname)  |  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Started: $(date)"

python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 'no CUDA — login node?')" || exit 1

# e73 is 27B: ~52 GB of bf16 weights before activations. Inference-only (no grads, no
# R-Drop) so it fits an 80 GB A100, but fail here rather than 40 minutes in.
python - <<'PY'
from utils import resolve_model_path
from transformers import AutoConfig
for name in ("Qwen/Qwen3.5-27B", "Qwen/Qwen3.5-9B"):
    AutoConfig.from_pretrained(resolve_model_path(name))
print("[preflight] candidate backbones resolve")
PY

python ensemble.py --subtask PA \
  --members configs/e25_xlmr_single_pa_stride256.yaml \
            configs/e32_qwen35_9b_pa.yaml \
            configs/e33_gemma4_12b_pa.yaml \
            configs/e73_qwen35_27b_pa.yaml \
            configs/e74_qwen25_14b_instruct_pa.yaml \
            configs/e80_qwen35_9b_pa_s1.yaml \
            configs/e84_gemma4_12b_pa_s1.yaml \
  --combine logit --exhaustive --exhaustive_top_k 10 \
  --threshold_policy dev_tuned --thr_step 0.01 \
  --output_dir ${OUT}
rc=$?

echo "Finished: $(date)"

cat <<EOF

--- the only decision that counts (CPU, login node) ---
  python bootstrap_paired.py --subtask PA --split test \\
      --a outputs/phaseC_v2_pa_logit/test_predictions_PA.csv \\
      --b ${OUT}/test_predictions_PA.csv

Re-lock ONLY if the 95% CI lower bound is above zero. The bar is the current lock
0.9449 (94.56 official after the final-boundary fix -- use the ledger figure, not
the pre-invariant one).

Read the dev/test table with §6.13 in hand: e71 gained +1.06 standalone and still
scored -0.15 in a decoder because its control was already zero-weighted. A big
standalone number is not evidence about the ensemble. e73's whole case here is
whether it earns a place, not that it is our best single.

NOT included: foldbag_e32. fold_bag.py writes only binarised CSVs, and ensemble.py's
one cache-member slot (--sat_cache) is hardcoded to the tag 'sat_ft' AND is appended
after the exhaustive sweep, so it would be forced into every subset instead of being
evaluated. Wiring a bag in needs a real cache-member path -- separate change.
EOF

exit $rc
