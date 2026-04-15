import torch
import wandb
from connect_four import ConnectFour
from connect4_oracle import PerfectC4Oracle
from alphazero import AlphaZeroParallel
from resnet import ResNet
from config import args
torch.manual_seed(0)


if __name__ == "__main__":
    wandb.init(
        project="AlphaZero",
        config=args
    )

    game = ConnectFour()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet(game, args["num_resBlocks"], args["num_hidden"], device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])
    oracle = PerfectC4Oracle(game)
    alphaZero = AlphaZeroParallel(model, optimizer, game, args, oracle=oracle)
    alphaZero.learn()