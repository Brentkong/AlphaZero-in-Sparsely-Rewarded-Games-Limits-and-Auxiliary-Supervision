import json
from pathlib import Path

import torch
import numpy as np
from mcts import MCTS
from config import args
from resnet import ResNet
from connect4_oracle import *
import matplotlib.pyplot as plt
from connect_four import ConnectFour
torch.manual_seed(10)

BASE_DIR = Path.home() / "Documents" / "AlphaZero-Chomp"
FIGURES_DIR = BASE_DIR / "figures"

game = ConnectFour()
player = 1
mode = "avo"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet(game, 9, 128, device)
model.load_state_dict(torch.load(BASE_DIR / "weights" / "Connect_Four" / "Auxiliary" / "model_19_ConnectFour.pt", map_location=device))
model.eval()
mcts = MCTS(game, args, model)
state = game.get_initial_state()
oracle = PerfectC4Oracle(game)
move_sequence = []
score_state = []
best_moves = []

while True:
    print(f"Player {player} to move")
    print(state)
    best_actions, scores = oracle.best_move(state, move_sequence)
    print(f"Score: {scores}") 
    print()
    best_moves.append(best_actions)
    
    if mode == "ava":
        if player == 1:
            neutral_state = game.change_perspective(state, player)
            mcts_probs = mcts.search(neutral_state)
            action = np.argmax(mcts_probs)

        else:
            neutral_state = game.change_perspective(state, player)
            mcts_probs = mcts.search(neutral_state)
            action = np.argmax(mcts_probs)
    
    elif mode == "avo":
        if player == 1:
            neutral_state = game.change_perspective(state, player)
            mcts_probs = mcts.search(neutral_state)
            action = np.argmax(mcts_probs)

        else:
            neutral_state = game.change_perspective(state, player)
            actions, _ = oracle.best_move(state, move_sequence)
            action = actions[0]
    else:
        if player == 1:
            neutral_state = game.change_perspective(state, player)
            mcts_probs = mcts.search(neutral_state)
            action = np.argmax(mcts_probs)

        else:
            valid_moves = game.get_valid_moves(state)
            print("valid_moves", [i for i in range(game.action_size) if valid_moves[i] == 1])
            action = int(input(f"{player}:"))

            if valid_moves[action] == 0:
                print("\n")
                print("action not valid")
                print("\n")
                continue 

    move_sequence.append(int(action))
    score_state.append(scores[action])
    state = game.get_next_state(state, action, player)
    value, is_terminal = game.get_value_and_terminated(state, action)
    
    if is_terminal:
        print(state)
        if value == 1:
            print(player, "won")
        else:
            print("draw")
        break
        
    player = game.get_opponent(player)

print(f"Move sequence: {move_sequence}")
print(f"Scores: {score_state}")
print(f"Best moves: {best_moves}")


data = {
    "move_sequence": move_sequence,
    "score_state": score_state,
    "best_moves": best_moves
}

with open(FIGURES_DIR / f"game_trace_{args['row_count']*args['column_count']}_{mode}.json", "w") as f:
    json.dump(data, f, indent=2)


plt.figure(figsize=(8, 5))
plt.plot(
    list(range(len(score_state))),
    score_state,
    marker='o',
    linestyle='--'
)

plt.xlabel("Move Number")
plt.ylabel("Score")
plt.title("Score vs. Move Number")
plt.grid(True)
plt.savefig(FIGURES_DIR / f"score_{args['row_count']*args['column_count']}_{mode}.png", dpi=300, bbox_inches='tight')
plt.show()
