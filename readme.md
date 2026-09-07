# AlphaZero in Sparsely Rewarded Games: Limits and Auxiliary Supervision

This repository contains the training code, oracle tooling, traces, and plotting scripts for **"AlphaZero in Sparsely Rewarded Games: Limits and Auxiliary Supervision."** The project studies the gap between strong empirical play and exact oracle-consistent play in AlphaZero-style agents.

## Paper

- arXiv: [arxiv.org/abs/2607.08984](https://arxiv.org/abs/2607.08984)
- OpenReview (TMLR): [openreview.net/forum?id=1z0CnFiJKg](https://openreview.net/forum?id=1z0CnFiJKg)

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
| `evaluation/` | Multi-seed experiment runner, sampled-state and moving-target diagnostics, trace aggregation, and result-compilation utilities. |
| `experiment_utils/` | Shared runtime helpers for deterministic seeding, CLI overrides, output paths, and configuration snapshots. |
| `eval-results/` | Published final checkpoints for every seed/configuration plus compact compiled and sampled-state evaluation results. |

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
- **Multi-frame AlphaZero** - Chomp-only variant that encodes several recent states instead of a single board. This checkout does not include a Connect Four multi-frame variant, so multi-frame claims should be scoped to Chomp.
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

Training entry points also accept runtime overrides:

```bash
python main.py --seed 1 --num-searches 200 --wandb-mode disabled
python main.py --seed 1 --rows 10 --cols 11 --outdir ../../results --wandb-mode offline
```

Checkpoints and `config_snapshot.json` are written under:

```text
results/{game}/{variant}/{board}/seed_{seed}/
```

The lowercase `results/` directory above is the default for newly launched
runs. It is ignored by Git. The uppercase `Results/` and
`Results-experiment/` directories, when present locally, contain the larger
raw experiment outputs used to build the curated, tracked `eval-results/`
release.

## Oracle Notes

For Chomp, the oracle wrappers use `grundy.cpp` through a native library:

- macOS libraries are currently checked in as `src/libgrundy.dylib`.
- On Linux, build `libgrundy.so` from the relevant `src/grundy.cpp`:

```bash
g++ -O3 -std=c++17 -shared -fPIC grundy.cpp -o libgrundy.so
```

For Connect Four, each Connect Four directory includes `connect4-master/` with the solver source, a `c4solver` binary, and `7x6.book`. The current `src/config.py` files resolve `solver_path` relative to the local variant directory.

Example local values:

```python
'solver_path': str(Path(__file__).resolve().parents[1] / "connect4-master")
```

or:

```python
'solver_path': '/path/to/repository/ConnectFour-Auxiliary/connect4-master'
```

## Evaluation Scripts

The `src/play.py` scripts load a saved checkpoint, run AlphaZero-vs-AlphaZero, AlphaZero-vs-player, or AlphaZero-vs-oracle style rollouts depending on the `mode` variable, and write trace JSON/PNG outputs.

The checkpoint and output paths are now command-line parameters. If `--checkpoint` is omitted, `play.py` uses the latest `model_*.pt` under the matching result directory.

```bash
python play.py --seed 1 --rows 9 --cols 10 \
  --checkpoint ../../eval-results/checkpoints/Chomp/Vanilla/9x10/seed_1/model_9_Chomp\(9x10\).pt \
  --eval-mode ava
```

The published checkpoints are organized as follows:

```text
eval-results/checkpoints/Chomp/{AZAL,Vanilla,MultiFrame}/{9x10,10x11}/seed_{0,1,2}/
eval-results/checkpoints/Connect-Four/{AZAL,Vanilla}/6x7/seed_{0,1,2}/
```

Chomp uses the final iteration-9 checkpoint; Connect Four uses the final
iteration-19 checkpoint. The matching hyperparameters are under
`eval-results/configs/`.

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

Additional reviewer-support utilities live in `evaluation/`:

```bash
python evaluation/run_multiseed.py --wandb-mode disabled
python evaluation/run_multiseed.py --sampled-states --moving-target --wandb-mode disabled
python evaluation/sampled_state_eval.py --game chomp --src-dir Chomp-Vanilla/src --checkpoint path/to/model.pt --rows 9 --cols 10
python evaluation/moving_target_diagnostic.py --game connect4 --src-dir ConnectFour-Vanilla/src --checkpoint path/to/model.pt
python evaluation/export_hyperparameters.py --seed-count 3
python evaluation/aggregate_traces.py --inputs results/**/game_trace_*_ava.json --outdir results/summary
```

To reproduce the checked-in sampled-state evaluation from the local raw
checkpoints under `Results/`, run:

```bash
bash run_sampled_state_experiment.sh
python evaluation/compile_results_json.py --results-dir Results --output Results/compiled_results.json
```

The first command evaluates five sampled states at depths `0`, `4`, `8`, and
`12` for each published seed/configuration and writes to
`Results-experiment/evaluations/sampled/`. These local raw outputs were copied
into `eval-results/evaluations/` for publication.

`run_multiseed.py` defaults to seeds `0 1 2` and `20` trace games per
checkpoint. With `--sampled-states`, it samples `32` states at each requested
depth (`0 4 8 12` by default). With `--moving-target`, it runs `8` self-play
diagnostic games per checkpoint.

The checked-in config JSON files use paths relative to the config file, pointing at the checked-in trace files under `Graph Creation/games/`.

`graphs.py` and `history.py` build local output paths with `pathlib.Path`. Before running `history.py`, set your W&B entity:

```bash
export WANDB_ENTITY="your-wandb-entity"
```

## Published Results

The compact, publication-ready result bundle is documented in
[`eval-results/README.md`](eval-results/README.md) and contains:

- 24 final model checkpoints: three seeds for each game/variant/board setup
- 24 matching `config_snapshot.json` files
- `eval-results/evaluations/compiled_results.json`, with metrics aggregated across seeds
- 24 sampled-state result sets in both JSON and CSV format

The repository also includes generated assets, including:

- Chomp trace JSONs and Grundy-number plots under each Chomp variant's `figures/`
- Connect Four trace JSONs and score plots under each Connect Four variant's `figures/`
- Combined training-loss graphs under `Graph Creation/figures/graphs/`
- Trace panels for Chomp and Connect Four under `Graph Creation/figures/graphs/`
- LaTeX metric tables under `Graph Creation/figures/tables/`

## Citation

If you use this repository, please cite the associated paper ([arXiv](https://arxiv.org/abs/2607.08984), [OpenReview](https://openreview.net/forum?id=1z0CnFiJKg)):

```bibtex
@article{kong2026alphazero,
  title={AlphaZero in Sparsely Rewarded Games: Limits and Auxiliary Supervision},
  author={Kong, Brent and Ram, Tejas and Yu, Tony Yue},
  journal={arXiv preprint arXiv:2607.08984},
  year={2026},
  eprint={2607.08984},
  archivePrefix={arXiv},
  url={https://arxiv.org/abs/2607.08984}
}
```
