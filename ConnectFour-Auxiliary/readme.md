# AlphaZero in Sparsely Rewarded Games: Limits and Auxiliary Supervision

This repository contains code, experiments, and analysis for the paper **“AlphaZero in Sparsely Rewarded Games: Limits and Auxiliary Supervision.”** The project studies the gap between **strong play** and **perfect play** in AlphaZero-style agents on two oracle-evaluable domains:

- **Chomp**
- **Connect Four**

The core comparison is between:

- **Vanilla AlphaZero**
- **Multi-frame AlphaZero**
- **AlphaZero Auxiliary Loss (AZAL)**

The paper shows that vanilla AlphaZero can achieve strong performance without consistently recovering perfect play, while **AZAL** substantially improves oracle alignment, reaching **perfect play in Chomp** and **near-perfect play in Connect Four**.

---

## Repository Purpose

This repo is organized around separate branches for different game/variant combinations, plus a `main` branch for graph generation and analysis.

In particular, this repository supports:

- training and evaluation for AlphaZero-based agents,
- branch-specific experiments for Chomp and Connect Four,
- oracle-based trace analysis,
- training-curve visualization,
- generation of trace graphs and trace tables.

The current GitHub repository view shows analysis-oriented files and folders such as `config/`, `figures/`, `games/`, `history/`, `generate_trace_graphs.py`, `generate_trace_tables.py`, `graphs.py`, and `history.py`.

---

## Branch Overview

The repository is split across the following branches:

### `main`
**Purpose:** graph making, trace generation, and analysis.

This branch is intended for:
- generating training curves,
- creating trace graphs,
- creating trace tables,
- storing experiment history and analysis outputs.

### Chomp branches

#### `Chomp-Vanilla-AlphaZero`
Vanilla AlphaZero experiments for **Chomp**.

#### `Chomp-Multiframe-AlphaZero`
Multi-frame AlphaZero experiments for **Chomp**.

#### `Chomp-AuxiliaryLoss-AlphaZero`
AZAL experiments for **Chomp**.

### Connect Four branches

#### `Connect-Four-Vanilla-AlphaZero`
Vanilla AlphaZero experiments for **Connect Four**.

#### `Connect-Four-AuxiliaryLoss-AlphaZero`
AZAL experiments for **Connect Four**.

---

## Research Summary

AlphaZero combines a neural network with Monte Carlo Tree Search (MCTS) and self-play. While this often leads to very strong policies, the paper argues that **superhuman play is not the same as perfect play**. In sparse or structure-heavy games, the standard search-learning loop may fail to preserve the exact trajectories required for optimal play.

This repository investigates that claim in two settings:

- **Connect Four**, where exact game-theoretic evaluation is available through a perfect solver.
- **Chomp**, where optimality is analyzed through **Grundy numbers** and the invariant of moving to `g = 0` states from winning positions.

To address this limitation, the project introduces **AlphaZero Auxiliary Loss (AZAL)**, which augments the standard AlphaZero objective with an oracle-derived auxiliary policy loss while leaving self-play, MCTS, and value targets unchanged.

---

## Expected Workflow

A typical workflow is:

1. Check out the branch for the experiment you want to run.
2. Train or evaluate the model for that branch’s game/variant.
3. Return to `main` for graph generation, trace analysis, and result visualization.

Example:

```bash
git clone https://github.com/Brentkong/AlphaZero-in-Sparsely-Rewarded-Games-Limits-and-Auxiliary-Supervision.git
cd AlphaZero-in-Sparsely-Rewarded-Games-Limits-and-Auxiliary-Supervision

# switch to a branch
git checkout Chomp-Vanilla-AlphaZero
````

To move back to analysis:

```bash
git checkout main
```

---

## Installation

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

If you are on Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## Repository Structure

On the `main` branch, the repository currently includes the following top-level items: `config/`, `figures/`, `games/`, `history/`, `.gitignore`, `generate_trace_graphs.py`, `generate_trace_tables.py`, `graphs.py`, `history.py`, and `requirements.txt`.

A high-level interpretation of these directories/files is:

* `config/` — experiment and plotting configuration
* `figures/` — generated visualizations and paper figures
* `games/` — game-specific data, traces, or outputs
* `history/` — logged training history / saved metrics
* `generate_trace_graphs.py` — trace-panel generation
* `generate_trace_tables.py` — LaTeX or analysis-ready trace tables
* `graphs.py` — plotting utilities
* `history.py` — history loading / processing utilities

---

## Methods Implemented

This project centers on three AlphaZero-style variants:

### 1. Vanilla AlphaZero

Standard self-play + MCTS + policy/value learning.

### 2. Multi-frame AlphaZero

Uses a short stack of recent states instead of a single board snapshot, testing whether richer input representation improves recovery of perfect play.

### 3. AlphaZero Auxiliary Loss (AZAL)

Adds an oracle-derived auxiliary policy loss to the usual AlphaZero objective:

```text
L = L_policy + L_value + λ_aux L_aux
```

This extra supervision biases the policy toward oracle-consistent actions while keeping the rest of the AlphaZero pipeline unchanged.

---

## Analysis and Outputs

This repository supports analysis such as:

* smoothed training-loss curves,
* move-by-move trace tables,
* deterministic greedy rollout trace panels,
* oracle-match metrics,
* longest oracle-consistent chain metrics.

These analyses are used to distinguish **strong empirical play** from **exact oracle-consistent play**.

---

## Branch Guide

Use this quick mapping to navigate the repository:

* `main` → graph making + analysis
* `Chomp-Vanilla-AlphaZero` → Chomp, vanilla AlphaZero
* `Chomp-Multiframe-AlphaZero` → Chomp, multi-frame AlphaZero
* `Chomp-AuxiliaryLoss-AlphaZero` → Chomp, AZAL
* `Connect-Four-Vanilla-AlphaZero` → Connect Four, vanilla AlphaZero
* `Connect-Four-AuxiliaryLoss-AlphaZero` → Connect Four, AZAL

---

## Paper

This repository accompanies the paper:

**AlphaZero in Sparsely Rewarded Games: Limits and Auxiliary Supervision**

Main paper theme:

* Vanilla AlphaZero can be strong without being perfect.
* Multi-frame inputs alone do not fully solve the problem.
* Stronger oracle-based supervision through AZAL materially improves optimality recovery.

---

## Citation

If you use this repository, please cite the associated paper.

```bibtex
@article{alphazero_sparse_auxiliary,
  title={AlphaZero in Sparsely Rewarded Games: Limits and Auxiliary Supervision},
  author={Brent Kong, Tony Yue Yu},
  journal={Preliminary work / under review},
  year={2026}
}
```

---

## Contact

For questions about the codebase or experiments, please open an issue in the repository.


