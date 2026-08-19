#!/bin/bash
# Lane A Task 2 — restore-aware probability caches for the two Naqta-restored
# candidates (e71 NoPnx-PA, e72 NoPnx-NP).
#
# Replaces the ad-hoc `sbatch --wrap` submissions that died as jobs 15065747 /
# 15065749: those exported no HF_HOME, so `xlm-roberta-large` (a bare HF id that
# resolve_model_path returns as-is) resolved against the user hub cache, which has
# no weights -> OSError at build_model. Both jobs burned 43 A100-minutes on the
# Naqta restoration pass FIRST and only then hit the missing backbone, so the
# preflight below loads the backbone before anything expensive runs.
#
# Same header as run_prob_cache.sh, which has always worked. .venv (not .venv-llm)
# is what e71/e72 were trained under and what run_restoration_retrain_wave.sh uses
# for the naqta_restore path.
#
# Submit (login node):
#   cd ${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments
#   bash slurm/run_lane_a_prob_cache.sh              # both heads
#   bash slurm/run_lane_a_prob_cache.sh nopnx-pa     # just the -1.4 head
set -euo pipefail

PROJECT=${ARASEG_PROJECT:?set ARASEG_PROJECT}
EXPERIMENTS_DIR=${PROJECT}/experiments
VENV=${PROJECT}/.venv
cd ${EXPERIMENTS_DIR}
mkdir -p logs outputs/prob_cache

ALL_HEADS=(nopnx-pa nopnx-np)
HEADS=("$@"); [ ${#HEADS[@]} -eq 0 ] && HEADS=("${ALL_HEADS[@]}")

declare -A SUBTASK=( [nopnx-pa]=NoPnx-PA        [nopnx-np]=NoPnx-NP )
declare -A CONFIG=(  [nopnx-pa]=e71_naqta_restore_nopnxpa [nopnx-np]=e72_naqta_restore_nopnxnp )
declare -A CKPTDIR=( [nopnx-pa]=outputs/e71     [nopnx-np]=outputs/e72 )

for h in "${HEADS[@]}"; do
    [ -n "${SUBTASK[$h]:-}" ] || { echo "unknown head '$h' (choose: ${ALL_HEADS[*]})"; exit 1; }
    # Fail on the login node, not 43 minutes into an A100 job.
    ckpt="${CKPTDIR[$h]}/best_${SUBTASK[$h]//-/_}.pt"
    [ -f "$ckpt" ] || { echo "missing trained checkpoint: ${ckpt}"; exit 1; }
done

for h in "${HEADS[@]}"; do
    st=${SUBTASK[$h]}
    out=$(sbatch <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=lane-a-probcache-${h}
#SBATCH -A kisski-futurmig
#SBATCH -p kisski
#SBATCH -G A100:1
#SBATCH --mem-per-gpu=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=${EXPERIMENTS_DIR}/logs/slurm_%j_%x.out
#SBATCH --error=${EXPERIMENTS_DIR}/logs/slurm_%j_%x.err
set -euo pipefail
export HF_HOME=${ARASEG_HF_HOME:-$HOME/.cache/huggingface}
export HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
source ${VENV}/bin/activate
cd ${EXPERIMENTS_DIR}

# Implementation note: the exact call that failed, run first and cheap. AutoConfig is not
# enough -- 15065747 had a readable config.json and still had no weights file.
python - <<'PY'
from utils import resolve_model_path
from transformers import AutoModel
path = resolve_model_path("xlm-roberta-large")
AutoModel.from_pretrained(path)
print(f"[preflight] backbone loads: {path}")
PY

python cache_probs.py --subtask ${st} \
    --members configs/${CONFIG[$h]}.yaml \
    --splits train dev test --out_dir outputs/prob_cache
SBATCH_EOF
)
    printf "  %-9s restore-aware prob cache -> job %s\n" "$st" "${out##* }"
done

cat <<'EOF'

=== on success, all CPU-only ===
  # gate_member reads outputs/prob_cache by convention (no --cache_dir flag) and
  # fits every non-empty subset; dev picks the winner, the paired CI decides.
  # It wants the candidate's OOF-TRAIN matrix, which these caches do NOT provide --
  # e71 has folds from job 15038761; if e72 has none, gate it with --no-folds
  # (uses the train cache directly) and treat that as the weaker evidence it is.
  python gate_member.py --subtask NoPnx-PA --candidate e71
  python gate_member.py --subtask NoPnx-NP --candidate e72 --no-folds

  # e69/e71 and e70/e72 are the paired questions Lane A actually asks:
  python gate_member.py --subtask NoPnx-PA --candidate e69 e71
EOF
