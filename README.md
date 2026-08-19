# AraSeg experiment code

This repository contains the training, evaluation, ensembling, and analysis code used
for the NAMAA-Community submission to the AraSeg Shared Task 2026. It includes every
numbered experiment configuration, including ablations and unsuccessful runs.

## Contents

- `experiments/train.py`, `model.py`, and `data_utils.py`: shared training pipeline.
- `experiments/configs/`: 86 experiment configurations.
- `experiments/ensemble.py`, `fit_oof_stack.py`, and `fit_decoder.py`: ensemble and
  train-only sequence-decoder pipelines.
- `experiments/sat_finetune.py`, `punct_aug.py`, and `naqta_*.py`: SaT adaptation,
  punctuation augmentation, and punctuation-restoration experiments.
- `experiments/official_eval.py` and `validate_submission.py`: scoring and structural
  submission checks.
- `experiments/test_*.py`: lightweight regression checks.
- `experiments/error_by_cluster.py` and `genre_proxy.py`: the per-cluster error analysis
  behind RQ3 in the paper.
- `experiments/collect_results.py`: rebuilds the per-experiment P/R/F1 table from the
  `results.json` files a run leaves behind. This table is the source of the paper's
  73-row ledger appendix; the appendix itself is published with the paper.
- `experiments/slurm/`: the submission scripts as actually run. Beyond launching jobs they
  are the record of each wave's exact invocation -- ensemble member lists, gate arguments,
  and threshold policies that live in no config file. Site-specific paths were replaced
  with `ARASEG_PROJECT` and `ARASEG_HF_HOME`; the `#SBATCH` partition, account, and GPU
  lines are left as run and will need editing for another cluster.

Datasets, checkpoints, logs, predictions, and probability caches are deliberately not
stored in Git. The four datasets are loaded from the public
`MBZUAI/AraSeg-2026-Shared-Task-*` repositories on Hugging Face. Gated model or blind
dataset access is supplied through environment variables; credentials must never be
written into this repository.

## Environment

Python 3.11 was used for the release preparation. Create an environment and install the
base dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
pip install -r experiments/requirements.txt
```

The Qwen3.5 and Gemma 4 LoRA configurations require the separately pinned packages in
`experiments/requirements-llm.txt`.

## Download datasets

```bash
cd experiments
python download_datasets.py
```

## Run one experiment

```bash
cd experiments
python train.py --config configs/e25_xlmr_single_pa_stride256.yaml
```

On a SLURM cluster:

```bash
CONFIG=configs/e25_xlmr_single_pa_stride256.yaml \
  sbatch --export=ALL,CONFIG experiments/slurm/run_experiment.sh
```

## Checks

The tests are executable without a test runner:

```bash
cd experiments
for test_file in test_*.py; do python "$test_file"; done
```

Some integration and smoke checks download models or datasets and therefore require
network access, cached resources, or a GPU.

## Pinned base revisions

- `Qwen/Qwen3.5-9B`: `c202236235762e1c871ad0ccb60c8ee5ba337b9a`
- `Qwen/Qwen3.5-27B`: `fc05daec18b0a78c049392ed2e771dde82bdf654`
- `google/gemma-4-12B`: `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`

Checkpoint and cache release locations and their checksums should be added before the
repository is published.
