# Evaluation Results

This directory contains the curated artifacts published with the repository.
The source training outputs remain in `Results/` and `Results-experiment/` and
are not modified by this collection.

## Contents

- `checkpoints/`: final model checkpoint for every game, variant, board size,
  and seed.
- `configs/`: the configuration snapshot associated with every checkpoint.
- `evaluations/compiled_results.json`: aggregate evaluation metrics.
- `evaluations/sampled/`: per-seed sampled-state evaluations in JSON and CSV.

Chomp checkpoints are from training iteration 9. Connect Four checkpoints are
from training iteration 19. Optimizer states, intermediate checkpoints, W&B
archives, debug logs, and per-game trace images are intentionally omitted.

PyTorch checkpoint files should only be loaded from trusted sources.
