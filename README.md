# NAMAA-Community at AraSeg Shared Task 2026

Experiment code for the NAMAA-Community closed-track submission to the
**AraSeg 2026** Arabic sentence-segmentation shared task.

AraSeg frames segmentation as binary token classification over four *aligned*
versions of the same documents, which makes punctuation a controlled variable:

| Subtask | Paragraph breaks | Punctuation |
|---|---|---|
| **PA** | ✓ | ✓ |
| **NP** | ✗ | ✓ |
| **NoPnx-PA** | ✓ | ✗ |
| **NoPnx-NP** | ✗ | ✗ |

The repository holds the full record — every numbered configuration, including
ablations and runs that failed their gate — not just the four submitted systems.

## Results

Macro-averaged document-level boundary F1. P/R/F1 are on the labeled practice
test; the blind column is the held-out official Testing split (F1 only is
reported by the organizers).

| Subtask | Submitted system | P | R | F1 | Blind F1 | Rank |
|---|---|---:|---:|---:|---:|---:|
| PA | logit average (3 members) | 95.01 | 94.59 | **94.49** | 94.4 | 9/13 |
| NP | OOF stack (4 + SaT) | 91.91 | 94.64 | **92.84** | 91.3 | 4/10 |
| NoPnx-PA | MEMM decoder (7 + SaT) | 87.74 | 89.31 | **87.82** | 89.9 | 4/10 |
| NoPnx-NP | MEMM decoder (6 + SaT) | 86.07 | 88.43 | **86.49** | 87.0 | 3/12 |

The mean blind change was +0.24 F1. The Testing phase accepted repeated
submissions, and every version we uploaded had already cleared its prespecified
paired-bootstrap gate on the practice test, so no blind score fed back into a
selection decision.

## What the paper reports

- **Model diversity helps.** Ensembles of bidirectional LoRA LLMs, XLM-R
  variants, a character-level SaT model and punctuation-informed members beat
  their strongest single-model anchors; architecturally distinct but weak
  members still receive large fitted weights.
- **Sequence decoding helps only without punctuation.** The same train-only
  logistic MEMM adds +0.92 / +0.55 F1 on the two punctuation-free subtasks and
  is rejected on both punctuated ones — a clean four-way split that also holds
  directionally on the blind data.
- **Punctuation removal is the dominant remaining cost.** Mapping punctuated
  predictions onto punctuation-free tokens exposes +7.2 / +7.1 F1 of oracle
  headroom (+6.6 / +6.5 against the later submitted systems, which the map-back
  study predates), and the loss concentrates in scriptural text.

## Layout

```
experiments/
  train.py model.py data_utils.py     shared training pipeline
  configs/                            86 experiment configurations
  ensemble.py fit_oof_stack.py        probability ensembling
  fit_decoder.py                      train-only sequence decoder (MEMM)
  sat_finetune.py sat_segment.py      SaT / wtpsplit adaptation
  punct_aug.py naqta_*.py             punctuation augmentation + restoration
  restore_then_segment.py             oracle map-back and restoration study
  error_by_cluster.py genre_proxy.py  per-cluster error analysis
  bootstrap_paired.py gate_member.py  paired document bootstrap, member gates
  oracle_headroom.py                  decoding-ceiling probes
  force_final_boundary.py             final-boundary structural constraint
  official_eval.py validate_submission.py   scoring + structural checks
  collect_results.py                  rebuilds the P/R/F1 ledger from results.json
  test_*.py                           regression checks
  slurm/                              submission scripts as actually run
```

`experiments/slurm/` is part of the record, not just plumbing: ensemble member
lists, gate arguments and threshold policies live in those invocations and in no
config file. Site paths were replaced with `ARASEG_PROJECT` / `ARASEG_HF_HOME`;
`#SBATCH` partition, account and GPU lines are left as run and need editing for
another cluster.

Datasets, checkpoints, logs, predictions and probability caches are deliberately
not in Git. The four datasets load from the public
`MBZUAI/AraSeg-2026-Shared-Task-*` Hugging Face repositories. Gated model and
blind dataset access come from environment variables — never commit credentials.

## Quickstart

Python 3.11.

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r experiments/requirements.txt
```

The Qwen3.5 and Gemma 4 LoRA configurations need the separately pinned packages
in `experiments/requirements-llm.txt`.

```bash
cd experiments
python download_datasets.py
python train.py --config configs/e25_xlmr_single_pa_stride256.yaml
```

On SLURM:

```bash
CONFIG=configs/e25_xlmr_single_pa_stride256.yaml sbatch --export=ALL,CONFIG experiments/slurm/run_experiment.sh
```

Checks:

```bash
cd experiments && pytest test_*.py
```

Four of the seven also run standalone as `python test_<name>.py`; the other
three use pytest fixtures and need the runner.

Some integration and smoke checks download models or datasets, so they need
network access, a warm cache, or a GPU.

## Pinned base revisions

| Model | Revision |
|---|---|
| `Qwen/Qwen3.5-9B` | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| `Qwen/Qwen3.5-27B` | `fc05daec18b0a78c049392ed2e771dde82bdf654` |
| `google/gemma-4-12B` | `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089` |

LoRA bases are stock Hugging Face weights at these revisions, so a run is
pinned by `base@revision` plus the adapter its config produces. Trained
adapters and probability caches are not currently released.

## Team & license

**NAMAA-Community** — Muhammed Ragab, Khloud Al Jallad, Karim Elsayed, Omer Nacar.

Code released under the **MIT** license. The four `AraSeg-2026-Shared-Task-*`
datasets are provided by the shared-task organisers under their own terms, and
the pretrained base models under their respective licenses; this repository
contains only our code, configurations, and derived analysis.
