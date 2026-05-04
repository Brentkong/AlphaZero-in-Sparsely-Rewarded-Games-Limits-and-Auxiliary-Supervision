import torch
import wandb
from chomp import Chomp
from alphazero import AlphaZeroParallel
from resnet import ResNet
from config import args
torch.manual_seed(0)


if __name__ == "__main__":
    wandb.init(
        project="chomp-alphazero",
        config=args
    )

    game = Chomp(args['rows'], args['cols'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet(game, args['num_resBlocks'], args['num_hidden'], device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

    alphaZero = AlphaZeroParallel(model, optimizer, game, args)
    alphaZero.learn()