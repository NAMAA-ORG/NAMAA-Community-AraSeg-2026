#!/bin/bash
# Inference-time window-overlap ablation on e25 (XLM-R-Large, PA).
#
# Khloud's res_k exp1 reports +0.2 F1 on all 8 cells from "Overlap=128" over her
# unannotated ensemble -- but 128 is OUR default (train.py window_stride), and
# data_utils passes it straight to the HF tokenizer's `stride=`, which IS the
# overlap. So her ablation is 0 -> 128, which we have had since the first run.
# What we have never tested is holding a CHECKPOINT fixed and varying the
# inference stride. e25 was TRAINED at 256; that arm is the control and its
# cache already exists (outputs/prob_cache/PA_{dev,test}_e25.pkl).
#
# CPU-only: no -G. cache_probs.py picks its device off torch.cuda.is_available(),
# which is False on a GPU-less allocation -- no flag needed. Same reasoning as
# run_syntax_extraction.sh: asking for an A100 only queues for hardware the job
# never uses, and this backfills instead.
#
# KISSKI-ONLY TRICK: with ARASEG_CKPT_ROOT unset, ensemble.py resolves the
# checkpoint from cfg.output_dir, NOT experiment_id -- so experiment_id is free
# to use as the cache tag. Off-cluster the opposite holds and this would break.
#
# Submit (login node):
#   cd ${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments
#   bash slurm/run_stride_ablation.sh          # strides 0 64 128 384
#   bash slurm/run_stride_ablation.sh 128      # just one, to time it first
set -euo pipefail

PROJECT=${ARASEG_PROJECT:?set ARASEG_PROJECT}
EXPERIMENTS_DIR=${PROJECT}/experiments
VENV=${PROJECT}/.venv-llm
cd ${EXPERIMENTS_DIR}
mkdir -p logs outputs/prob_cache configs/tmp

BASE=configs/e25_xlmr_single_pa_stride256.yaml
STRIDES=("$@"); [ ${#STRIDES[@]} -eq 0 ] && STRIDES=(0 64 128 384)

for s in "${STRIDES[@]}"; do
    cfg=configs/tmp/e25_s${s}.yaml
    # experiment_id -> cache tag; output_dir MUST stay outputs/e25 (checkpoint lives there)
    sed -e "s/^window_stride:.*/window_stride: ${s}/" \
        -e "s/^experiment_id:.*/experiment_id: e25_s${s}/" "${BASE}" > "${cfg}"
    grep -q "^output_dir: outputs/e25$" "${cfg}" || { echo "output_dir drifted in ${cfg}"; exit 1; }

    out=$(sbatch <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=stride-abl-${s}
#SBATCH -A kisski-futurmig
#SBATCH -p kisski
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
# Measured: stride 128, dev+test, 16 cores -> 14 min (job 15353673, 2026-08-18).
# 1 h is ~4x headroom; stride 0 makes FEWER windows, 384 makes more but not 4x.
#SBATCH --output=${EXPERIMENTS_DIR}/logs/slurm_%j_%x.out
#SBATCH --error=${EXPERIMENTS_DIR}/logs/slurm_%j_%x.err
export HF_HOME=${ARASEG_HF_HOME:-$HOME/.cache/huggingface}
export HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=""
# torch defaults to 1 thread under Slurm; without this it is ~10x slower.
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
source ${VENV}/bin/activate
cd ${EXPERIMENTS_DIR}
python -c "import torch; torch.set_num_threads(16)"
python cache_probs.py --subtask PA --members ${cfg} \
    --splits dev test --out_dir outputs/prob_cache
SBATCH_EOF
)
    printf "  stride %-4s -> job %s\n" "${s}" "${out##* }"
done

cat <<'EOF'

=== next (CPU, login node, once the jobs land) ===
# e25's own dev-tuned threshold is 0.55 (ledger 2.0); test F1 0.9124 / P 92.88 / R 90.63.
# Matched threshold keeps this a clean paired comparison -- retuning per arm with
# sweep_thr.py is the second question, not this one.
python cache_to_csv.py --subtask PA --split test --member e25    --threshold 0.55 --out outputs/stride_abl/e25_s256.csv
for s in 0 64 128 384; do
  python cache_to_csv.py --subtask PA --split test --member e25_s$s --threshold 0.55 --out outputs/stride_abl/e25_s$s.csv
done
for s in 0 64 128 384; do
  echo "== stride $s vs trained 256 =="
  python bootstrap_paired.py --subtask PA --split test \
      --a outputs/stride_abl/e25_s256.csv --b outputs/stride_abl/e25_s$s.csv
done
EOF
