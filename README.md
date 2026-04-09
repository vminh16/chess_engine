# Chess Engine Research Repository

[Tiếng Việt](README.vi.md)

This repository combines a custom chess engine, neural value evaluators, and an experiment history focused on diagnosing and improving value calibration for search. The current research line centers on `architecture_v2`, the `18x8x8` STM-relative representation, and a sequence of experiment suites that isolate two major failure modes:

- **Failure A: mid-band magnitude compression**
- **Failure B: ultra-center over-confidence**

## What this repository contains

- A custom chess rules and move-generation stack in [`core`](core).
- Classical search code in [`search`](search) with a neural/static evaluation bridge in [`evaluation`](evaluation).
- Two model families in [`model`](model), including the current research baseline in [`model/architecture_v2`](model/architecture_v2).
- Data preparation, sharded datasets, and training entry points in [`data`](data), [`train`](train), and [`train_v2_TF1`](train_v2_TF1).
- Experiment suites and cached reports in [`experiments`](experiments).
- Structured technical documentation in [`docs`](docs).

## Documentation map

- [`docs/README.md`](docs/README.md): documentation index
- [`docs/research/experiment_journal.md`](docs/research/experiment_journal.md): consolidated experiment journal
- [`docs/design/ft1_full_retrain_pipeline_spec.md`](docs/design/ft1_full_retrain_pipeline_spec.md): FT1 training design
- [`docs/reports/project_report_vi.md`](docs/reports/project_report_vi.md): Vietnamese project report

## Repository structure

```text
core/               Chess rules, board state, move generation
evaluation/         Static and neural evaluation entry points
search/             Negamax, ordering, SEE, transposition table
model/              Network architectures and model code
data/               Data preparation notebooks and dataset tooling
train/              Legacy/local training entry points
train_v2_TF1/       FT1 Colab-oriented training notebook and helper
experiments/        Root-cause diagnostics, ablations, pilot runs
docs/               Architecture, design notes, research logs, reports
```

## Current research takeaway

The strongest current evidence in this repo points to:

- Failure A being primarily **objective-side**.
- Failure B being a combination of **center-label impurity** and **gradient interference**.
- Late short-horizon polish runs being unreliable as a primary fix for Failure B.
- Full training from epoch 0 with cleaner center supervision being the most plausible next direction.

## Running the project

- Web UI: `python app.py`
- UCI loop: `python uci.py`
- Legacy training: open [`train/train.ipynb`](train/train.ipynb)
- FT1 Colab run: open [`train_v2_TF1/train.ipynb`](train_v2_TF1/train.ipynb)

## Notes

- Large datasets and experiment outputs are stored in-repo for research convenience.
- Some paths inside older documents are historical references to external checkpoints or downloads; the new docs layout groups the active references under [`docs`](docs).
