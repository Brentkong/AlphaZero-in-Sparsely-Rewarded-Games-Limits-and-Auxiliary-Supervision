import os
import sys
from pathlib import Path

import torch
import wandb
from chomp import Chomp
from alphazero import AlphaZeroParallel
from resnet import ResNet
from config import args

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiment_utils.runtime import (
    prepare_training_run,
    wandb_init_kwargs,
)


if __name__ == "__main__":
    cli, output_dir = prepare_training_run(_REPO_ROOT, args)
    os.chdir(output_dir)

    wandb.init(
        project="chomp-alphazero",
        config=args,
        **wandb_init_kwargs(cli.wandb_mode),
    )

    game = Chomp(args['rows'], args['cols'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet(game, args['num_resBlocks'], args['num_hidden'], device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

    alphaZero = AlphaZeroParallel(model, optimizer, game, args)
    alphaZero.learn()
