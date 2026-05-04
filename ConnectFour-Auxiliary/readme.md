# AlphaZero in Sparsely Rewarded Games: Limits and Auxiliary Supervision

This repository contains the training code, oracle tooling, traces, and plotting scripts for **"AlphaZero in Sparsely Rewarded Games: Limits and Auxiliary Supervision."** The project studies the gap between strong empirical play and exact oracle-consistent play in AlphaZero-style agents.

The current checkout is organized as a directory-based snapshot of the experiment variants, not as a branch-only layout.

## Current Layout

Top-level directories:

| Path | Contents |
| --- | --- |
| `Chomp-Vanilla/` | Vanilla AlphaZero for Chomp. Includes Chomp game logic, MCTS, ResNet, training loop, Grundy oracle wrapper, `grundy.cpp`, a checked-in macOS `libgrundy.dylib`, and saved trace figures. |
| `Chomp-Multiframe/` | Multi-frame Chomp AlphaZero. Similar to `Chomp-Vanilla/`, but the game/model path uses a stack of recent board states. |
| `Chomp-Auxiliary/` | Chomp AlphaZero with auxiliary oracle policy loss. Uses Grundy-derived best moves through `src/grundy_oracle.py`. |
| `ConnectFour-Vanilla/` | Vanilla AlphaZero for 6x7 Connect Four. Includes Python training code and a bundled `connect4-master/` perfect solver source/binary/opening book. |
| `ConnectFour-Auxiliary/` | Connect Four AlphaZero with auxiliary oracle policy loss. Uses `PerfectC4Oracle` and the bundled `connect4-master/` solver. |
| `Graph Creation/` | Analysis scripts, trace JSONs, training-history CSVs, generated graph PNGs, and generated LaTeX tables. |

Each experiment directory has:

- `src/main.py` - training entry point
- `src/config.py` - hyperparameters and solver/oracle settings
- `src/play.py` - rollout/evaluation script that records trace data
- `src/alphazero.py`, `src/mcts.py`, `src/resnet.py` - core AlphaZero implementation
- `figures/` - saved trace JSONs and plots for that variant
- `requirements.txt` - dependencies for that specific variant

The root also has `requirements.txt`. The current root and subproject requirements files list the same dependency set: `tqdm`, `matplotlib`, `wandb`, `torch`, `pandas`, `scipy`, and `tabulate`.

## Methods

The repository currently contains code for three AlphaZero-style variants:

- **Vanilla AlphaZero** - standard self-play, MCTS, policy loss, and value loss.
- **Multi-frame AlphaZero** - Chomp-only variant that encodes several recent states instead of a single board.
- **AlphaZero Auxiliary Loss (AZAL)** - adds an oracle-derived auxiliary policy loss:

```text
L = L_policy + L_value + lambda_aux L_aux
```

The oracle signal comes from Grundy-number analysis for Chomp and a perfect Connect Four solver for Connect Four.

## Installation

Create a virtual environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The subproject requirements files mirror the root dependency set right now, so installing the root file is enough for this checkout. You can still install from a specific folder if you are working with it independently:

```bash
pip install -r ConnectFour-Auxiliary/requirements.txt
pip install -r "Graph Creation/requirements.txt"
```

## Running Training

Training scripts are meant to be run from the variant's `src/` directory so local imports and native oracle files resolve cleanly.

Examples:

```bash
cd Chomp-Vanilla/src
python main.py
```

```bash
cd Chomp-Multiframe/src
python main.py
```

```bash
cd Chomp-Auxiliary/src
python main.py
```

```bash
cd ConnectFour-Vanilla/src
python main.py
```

```bash
cd ConnectFour-Auxiliary/src
python main.py
```

Training uses Weights & Biases through `wandb.init(...)` and writes model/optimizer checkpoints such as `model_<iteration>_<game>.pt` and `optimizer_<iteration>_<game>.pt` into the current working directory. Checkpoints and `wandb/` output are ignored by the root `.gitignore`.

Edit each variant's `src/config.py` to change board size, iteration counts, search counts, batch size, auxiliary-loss weight, or solver path.

## Oracle Notes

For Chomp, the oracle wrappers use `grundy.cpp` through a native library:

- macOS libraries are currently checked in as `src/libgrundy.dylib`.
- On Linux, build `libgrundy.so` from the relevant `src/grundy.cpp`:

```bash
g++ -O3 -std=c++17 -shared -fPIC grundy.cpp -o libgrundy.so
```

For Connect Four, each Connect Four directory includes `connect4-master/` with the solver source, a `c4solver` binary, and `7x6.book`. The current `src/config.py` files still point `solver_path` at an older absolute path under `/Users/brentkong/Documents/AlphaZero-Chomp/connect4-master`; update that value to the local solver directory if the old path is not present.

Example local values:

```python
'solver_path': '/Users/brentkong/Desktop/AlphaZero-Fresh/ConnectFour-Vanilla/connect4-master'
```

or:

```python
'solver_path': '/Users/brentkong/Desktop/AlphaZero-Fresh/ConnectFour-Auxiliary/connect4-master'
```

## Evaluation Scripts

The `src/play.py` scripts load a saved checkpoint, run AlphaZero-vs-AlphaZero, AlphaZero-vs-player, or AlphaZero-vs-oracle style rollouts depending on the `mode` variable, and write trace JSON/PNG outputs.

Current caveat: several `play.py` files contain hardcoded checkpoint and output paths under `/Users/brentkong/Documents/AlphaZero-Chomp/...`. Update those paths before using the scripts from this `AlphaZero-Fresh` checkout.

Trace JSONs use:

- `move_sequence`
- `best_moves`
- `grundy_numbers` for Chomp
- `score_state` for Connect Four

Those trace formats are consumed by the analysis scripts in `Graph Creation/`.

## Graph Creation

`Graph Creation/` contains the analysis side of the project:

- `history/` - W&B-exported training-history CSV files
- `games/` - checked-in trace JSONs by game, board size, and variant
- `figures/graphs/` - generated loss curves and trace panels
- `figures/tables/` - generated LaTeX metric tables
- `generate_trace_graphs.py` - builds trace-panel PNGs from trace JSON config files
- `generate_trace_tables.py` - builds compact LaTeX tables from trace JSON config files
- `graphs.py` - plots training losses from CSV histories
- `history.py` - exports a W&B run history to CSV

Example commands:

```bash
cd "Graph Creation"
python generate_trace_graphs.py --config config/chomp_9x10_config.json --outdir figures/graphs
python generate_trace_tables.py --config config/chomp_9x10_config.json --outdir figures/tables
```

The checked-in config JSON files currently reference older absolute trace paths under `/Users/brentkong/Documents/AlphaZero-Chomp/...`. To rerun them from this checkout alone, point the `traces` entries at the local files under `Graph Creation/games/`.

`graphs.py` and `history.py` also contain hardcoded local paths/W&B identifiers, so treat them as project scripts to edit for the run or machine you are using.

## Generated Results Currently Checked In

The repository includes generated assets, including:

- Chomp trace JSONs and Grundy-number plots under each Chomp variant's `figures/`
- Connect Four trace JSONs and score plots under each Connect Four variant's `figures/`
- Combined training-loss graphs under `Graph Creation/figures/graphs/`
- Trace panels for Chomp and Connect Four under `Graph Creation/figures/graphs/`
- LaTeX metric tables under `Graph Creation/figures/tables/`

## Citation

If you use this repository, please cite the associated paper:

```bibtex
@article{alphazero_sparse_auxiliary,
  title={AlphaZero in Sparsely Rewarded Games: Limits and Auxiliary Supervision},
  author={Brent Kong and Tony Yue Yu},
  journal={Preliminary work / under review},
  year={2026}
}
```
