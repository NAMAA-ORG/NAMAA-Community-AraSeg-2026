#!/bin/bash
#SBATCH --job-name=sat-prob-probe
#SBATCH -A kisski-futurmig
#SBATCH -p kisski
#SBATCH -G A100:1
#SBATCH --mem-per-gpu=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm_%j_%x.out
#SBATCH --error=logs/slurm_%j_%x.err
#
# SaT ensemble probe (PROB) — does adding the full-FT SaT member help each head's
# locked pool? For every head, run the pool WITHOUT and WITH the sat_ft cache under
# --combine prob, dev-tuned threshold, and print both TEST F1s side by side. Cheap:
# reuses the sat_fullft_caches (dev/test) already on disk; only recomputes the
# checkpoint members' probs once per run. One GPU, ~30-40 min.
#
# Submit (login node):
#   cd ${ARASEG_PROJECT:?set ARASEG_PROJECT}/experiments
#   sbatch slurm/run_sat_prob_probe.sh
set -euo pipefail

PROJECT=${ARASEG_PROJECT:?set ARASEG_PROJECT}
EXPERIMENTS_DIR=${PROJECT}/experiments
VENV=${PROJECT}/.venv-llm
export HF_HOME=${ARASEG_HF_HOME:-$HOME/.cache/huggingface}
export HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
source ${VENV}/bin/activate
cd ${EXPERIMENTS_DIR}

SAT=outputs/sat_fullft_caches                    # {ST}_{dev,test}_sat_ft.pkl live here

declare -A MEMBERS
MEMBERS[PA]="e25_xlmr_single_pa_stride256 e32_qwen35_9b_pa e33_gemma4_12b_pa"
MEMBERS[NP]="e40_qwen35_9b_np e15_xlmr_single_np_s1 e16_xlmr_single_np_s2 e38_xlmr_multitask_joint"
MEMBERS[NoPnx-PA]="e41_qwen35_9b_nopnx_pa e17_xlmr_single_nopnxpa_s1 e18_xlmr_single_nopnxpa_s2 e11_xlmr_single_nopnxpa_w2 e38_xlmr_multitask_joint"
MEMBERS[NoPnx-NP]="e42_qwen35_9b_nopnx_np e19_xlmr_single_nopnxnp_s1 e20_xlmr_single_nopnxnp_s2 e12_xlmr_single_nopnxnp_w2 e38_xlmr_multitask_joint"

for st in PA NP NoPnx-PA NoPnx-NP; do
    cfgs=""
    for m in ${MEMBERS[$st]}; do cfgs="${cfgs} configs/${m}.yaml"; done
    us=${st//-/_}
    echo "############## ${st}: pool WITHOUT SaT ##############"
    python ensemble.py --subtask ${st} --members ${cfgs} --combine prob --thr_step 0.01 \
        --output_dir outputs/sat_probe/${us}_baseline
    echo "############## ${st}: pool WITH SaT ##############"
    python ensemble.py --subtask ${st} --members ${cfgs} --combine prob --thr_step 0.01 \
        --sat_cache ${SAT} --output_dir outputs/sat_probe/${us}_withsat
done
echo "=== probe done. Compare TEST F1 lines (WRITTEN@dev_thr) baseline vs withsat. ==="
