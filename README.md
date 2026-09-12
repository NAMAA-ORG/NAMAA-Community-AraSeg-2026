# NAMAA-Community at AraSeg Shared Task 2026

Experiment code for the NAMAA-Community submission to the **AraSeg 2026** Arabic
sentence-segmentation shared task.

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

Macro-averaged document-level boundary F1. Practice columns are the labeled
practice test; the blind columns are the held-out official Testing split, scored
by the organizers to one decimal. Ranks are among registered teams that
submitted a system-description paper, which is why they differ from the raw
CodaBench leaderboards. The same four systems were entered in both tracks.

| Subtask | Submitted system | Practice P / R / F1 | Blind P / R / F1 | Closed rank | Open rank |
|---|---|---:|---:|---:|---:|
| PA | logit average (3 members) | 95.01 / 94.59 / **94.49** | 93.5 / 95.6 / **94.4** | 6 / 12 | 4 / 7 |
| NP | OOF stack (4 + SaT) | 91.91 / 94.64 / **92.84** | 92.3 / 91.2 / **91.3** | 3 / 10 | 2 / 6 |
| NoPnx-PA | MEMM decoder (7 + SaT) | 87.74 / 89.31 / **87.82** | 88.4 / 92.3 / **89.9** | 4 / 9 | 3 / 6 |
| NoPnx-NP | MEMM decoder (6 + SaT) | 86.07 / 88.43 / **86.49** | 85.9 / 89.2 / **87.0** | 3 / 10 | 3 / 7 |

The mean blind change was +0.24 F1. The Testing phase accepted repeated
submissions, and every version we uploaded had already cleared its prespecified
paired-bootstrap gate on the practice test, so no blind score fed back into a
selection decision.

## Models

The nineteen checkpoints behind the four submitted systems, plus the fitted
combiner weights, are released under
[NAMAA-Community-AraSeg-2026](https://huggingface.co/collections/NAMAA-Space/namaa-community-araseg-2026).

Start at [`NAMAA-Space/araseg-2026`](https://huggingface.co/NAMAA-Space/araseg-2026):
it holds the decoder and stacker weights, the thresholds, and the frozen
out-of-fold TRAIN matrices they were fit on. The members are ensemble inputs —
each emits uncalibrated per-word boundary probabilities and reproduces no
published score on its own.

| Subtask | Members | Thr. |
|---|---|---|
| PA | `e25-xlmr-pa` `e32-qwen35-9b-pa` `e33-gemma4-12b-pa` | 0.25 |
| NP | `e40-qwen35-9b-np` `e15-xlmr-np-s1` `e16-xlmr-np-s2` `e38-xlmr-multitask` `sat-ft` | 0.36 |
| NoPnx-PA | `e41-qwen35-9b-nopnx-pa` `e17-xlmr-nopnx-pa-s1` `e18-xlmr-nopnx-pa-s2` `e11-xlmr-nopnx-pa-w2` `e38-xlmr-multitask` `e69-naqta-nopnx-pa` `e76-naqta-restore-nopnx-pa-s2` `sat-ft` | 0.46 |
| NoPnx-NP | `e42-qwen35-9b-nopnx-np` `e19-xlmr-nopnx-np-s1` `e20-xlmr-nopnx-np-s2` `e12-xlmr-nopnx-np-w2` `e38-xlmr-multitask` `e70-naqta-nopnx-np` `sat-ft` | 0.34 |

All repos are `NAMAA-Space/araseg-<name>`. `e38` and `sat-ft` are members of three
systems each; PA has no fitted head, being a plain logit average.

Weights are PyTorch `state_dict` files, not HF-format checkpoints — build the
architecture from the config YAML and base model, then `load_state_dict`, as
`ensemble.py` and `verify_offcluster.py` do. The LoRA members need the pinned
`requirements-llm.txt` stack to instantiate at all.

Two caveats worth reading before reproducing:

- `e76` was trained on text with Naqta-predicted commas inserted, and needs
  `MostafaMaroof/Naqta` at **inference** time (`min_p=0.3`, comma `,`), not only
  during training. NoPnx-PA does not reproduce without it.
- The `oof/` matrices in the hub cannot be regenerated — the five-fold models
  that produced them are gone. They include the `e75` folds, which belong to no
  lock but are required to reproduce the 15-subset gate that *selected* the final
  NoPnx-PA membership.

## What the paper reports

- **Model diversity helps.** Ensembles of bidirectional LoRA LLMs, XLM-R
  variants, a character-level SaT model and punctuation-informed members beat
  their strongest single-model anchors by +1.2 / +1.6 / +3.2 / +4.1 F1 from PA
  through NoPnx-NP; architecturally distinct but weak members still receive
  large fitted weights.
- **Sequence decoding helps only without punctuation.** The same train-only
  logistic MEMM adds +0.9 / +0.6 F1 on the two punctuation-free subtasks and
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
| `FacebookAI/xlm-roberta-large` | `c23d21b0620b635a76227c604d44e43a9f0ee389` |
| `MostafaMaroof/Naqta` | `3ce58ba7dd6ae2fa7eaf505a5fc72db479efe01d` |
| `segment-any-text/sat-12l-sm` | `d70c72a9331b2d5a9e82baad00c64964a23a09bb` |

LoRA bases are stock Hugging Face weights at these revisions, so a run is
pinned by `base@revision` plus the adapter its config produces. Trained
adapters and probability caches are not currently released.

## Team & license

**NAMAA-Community** — Muhammed Ragab, Khloud Al Jallad, Karim Elsayed, Omer Nacar.

Code released under the **MIT** license. The four `AraSeg-2026-Shared-Task-*`
datasets are provided by the shared-task organisers under their own terms. The
released checkpoints inherit their base models' licenses: MIT for the XLM-R and
SaT members, Apache-2.0 for the Qwen members, and — for `e33` alone — the
**Gemma Terms of Use**, which restrict redistribution and downstream use more
than the rest of the collection. Check that license before reusing `e33` or any
ensemble that includes it.
