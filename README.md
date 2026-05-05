# Chess Engine

> A neural-network-augmented chess engine with a web UI and UCI interface

![Python](https://img.shields.io/badge/python-%E2%89%A53.8-blue)
![PyTorch](https://img.shields.io/badge/framework-PyTorch-ee4c2c)
![License](https://img.shields.io/badge/license-MIT-green)

[Tiếng Việt](README.vi.md)

---

## Project Overview

This repository serves two purposes:

1. **Playable chess engine** — exposed as a Flask web application and a standard UCI engine, so it can be plugged into any UCI-compatible GUI (Arena, Cute Chess, etc.).
2. **Research platform** — a structured experiment history for diagnosing and improving neural value calibration during tree search.

Two neural model families are included:

| Family | Location | Role |
|--------|----------|------|
| `PhantomChessNet` / `DGRNChessNet` | [`model/architecture`](model/architecture) | Legacy v1, used by the web UI |
| `DGRNChessNetV2` | [`model/architecture_v2`](model/architecture_v2) | Current research baseline |

The current research line isolates two failure modes in value calibration:

- **Failure A: mid-band magnitude compression**
- **Failure B: ultra-center over-confidence**

---

## Architecture Overview

### Board representation

Positions are encoded as an **18 × 8 × 8** side-to-move-relative tensor (see [`representation/encode.py`](representation/encode.py)):

| Channels | Content |
|----------|---------|
| 0 – 5 | Own pieces: P, N, B, R, Q, K |
| 6 – 11 | Opponent pieces: P, N, B, R, Q, K |
| 12 | Side to move (1 = White, 0 = Black) |
| 13 – 16 | Castling rights (own kingside/queenside, opponent kingside/queenside) |
| 17 | En-passant target square |

When it is Black's turn the board is flipped vertically so the network always sees its own pieces at the bottom.

### Search

Negamax with the following enhancements (see [`search/negamax.py`](search/negamax.py)):

- Alpha-beta pruning
- Quiescence search
- Null-move pruning (R = 3, min depth 3)
- Killer moves (2 slots per ply)
- History heuristic
- Late Move Reduction (LMR)
- Static Exchange Evaluation (SEE) for capture ordering
- Transposition table

### Evaluation

A **hybrid evaluator** blends a fast material score with a neural network score:

```
eval = (1 − ε) × material_score + ε × neural_score
```

`ε` is controlled by the `ENGINE_EPSILON` environment variable (default: `0.2` for the UCI engine, `0.1` for the web app).

### Neural network backbone

```
Input (18 × 8 × 8)
  → Conv2d stem (3 × 3, BatchNorm, Mish)
  → N × DFGBlock (Dual-Focus Gated residual, stochastic depth)
  → ResidualGainValueHead
  → scalar output in [−1, 1]
```

`DGRNChessNetV2` uses 12 blocks and 128 hidden channels by default; the larger `DGRNChessNet` alias uses 20 blocks and 256 channels.

---

## Repository Structure

```text
core/                   Board state, FEN parsing, move generation, rules
evaluation/             naive.py (material), nn.py (neural bridge), static_eval.py
search/                 negamax.py, ordering.py, see.py, transition_table.py, utils.py
representation/         encode.py — board → 18×8×8 tensor
model/architecture/     PhantomChessNet / DGRNChessNet v1 (legacy web UI)
model/architecture_v2/  DGRNChessNetV2 (current research baseline)
data/                   Dataset tooling and sharded .npz files
train/                  Legacy local training notebook
train_v2_TF1/           FT1 Colab-oriented training notebook
experiments/            Root-cause diagnostics, ablations, pilot runs
docs/                   Architecture specs, design docs, research journal
bench/                  Benchmark results (NPS, inference latency)
scripts/                Utility scripts
static/ templates/      Flask web UI assets
```

---

## Prerequisites & Installation

**Requirements:** Python ≥ 3.8, pip

```bash
# 1. Clone the repository
git clone https://github.com/vminh16/chess_engine.git
cd chess_engine

# 2. Install dependencies
pip install torch numpy flask
```

**Model weights** are required for the engine to evaluate positions:

| Weight file | Used by |
|-------------|---------|
| `model/param_model/PhantomChessNet.pth` | Web UI (`app.py`) |
| `model/nn_parameters.pth` | UCI engine (`uci.py`) |

Place the `.pth` files in the paths above before running. If a weight file is missing the engine will still start but will fall back to pure material evaluation and print a warning.

---

## Usage

### Web UI

```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser. The engine searches at depth 4 and responds with its best move via a REST endpoint called by the front-end.

### UCI Engine

```bash
python uci.py
```

Connect with any UCI-compatible GUI (Arena, Cute Chess, ChessBase, etc.). Supported commands: `uci`, `isready`, `ucinewgame`, `position [startpos | fen] [moves …]`, `go`, `quit`.

You can tune the neural blend ratio at startup:

```bash
ENGINE_EPSILON=0.3 python uci.py   # 30 % neural, 70 % material
```

### Training

```bash
# Legacy local training
jupyter notebook train/train.ipynb

# FT1 — optimised for Google Colab
jupyter notebook train_v2_TF1/train.ipynb
```

### Benchmarks

Pre-recorded benchmark results (NPS and inference latency) are in [`bench/benchmark_results.txt`](bench/benchmark_results.txt). Baseline: ~226 NPS on CPU at depth 4.

---

## Research Context

### Experiment pipeline

Experiment suites live in [`experiments/`](experiments/). Each suite targets a specific hypothesis (objective change, label cleaning, architecture tweak) and caches its outputs for reproducibility. The consolidated findings are in [`docs/research/experiment_journal.md`](docs/research/experiment_journal.md).

### Current takeaways

The strongest current evidence in this repo points to:

- **Failure A** being primarily **objective-side** (loss function and target scaling).
- **Failure B** being a combination of **center-label impurity** and **gradient interference**.
- Late short-horizon polish runs being unreliable as a primary fix for Failure B.
- Full training from epoch 0 with cleaner center supervision being the most plausible next direction.

---

## Documentation Map

| Document | Purpose |
|----------|---------|
| [`docs/README.md`](docs/README.md) | Documentation index |
| [`docs/research/experiment_journal.md`](docs/research/experiment_journal.md) | Consolidated experiment journal |
| [`docs/design/ft1_full_retrain_pipeline_spec.md`](docs/design/ft1_full_retrain_pipeline_spec.md) | FT1 training pipeline specification |
| [`docs/reports/project_report_vi.md`](docs/reports/project_report_vi.md) | Vietnamese project report |

---

## Contributing

- **Code style:** Python throughout. Core-engine code (board, move generator, search) contains Vietnamese inline comments — please preserve this style when editing those files.
- **Datasets and experiment outputs** are stored in-repo intentionally so that results remain reproducible and traceable without external dependencies.
- There are no strict branching conventions at present; feature branches off `main` with descriptive names are encouraged.
- When adding or modifying an experiment suite, update [`docs/research/experiment_journal.md`](docs/research/experiment_journal.md) with a summary of the results.

---

## License

This project does not currently include a license file. It is recommended to add an MIT License (or another permissive license) to clarify usage rights for contributors and users.
