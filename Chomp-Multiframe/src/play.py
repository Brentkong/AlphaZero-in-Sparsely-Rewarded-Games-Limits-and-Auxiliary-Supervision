import time
import json
import sys
from pathlib import Path

import torch
import numpy as np
from config import args
from mcts import MCTS
from chomp import Chomp
from resnet import ResNet
import matplotlib.pyplot as plt
from grundy_oracle import grundy, fits_grundy_limit, find_best_moves

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiment_utils.runtime import (
    metadata,
    prepare_play_run,
)

cli, OUTPUT_DIR, checkpoint = prepare_play_run(_REPO_ROOT, args)

rows, cols = args['rows'], args['cols']
chomp = Chomp(rows, cols)
player = 1
mode = cli.eval_mode

state = chomp.get_initial_state()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet(chomp, args['num_resBlocks'], args['num_hidden'], device) 
model.load_state_dict(torch.load(checkpoint, map_location=device))
model.eval()
mcts = MCTS(chomp, args, model)

move_sequence = []
grundy_numbers = []
best_moves = []
start = time.time()
while True:
    print(state)
    if fits_grundy_limit(state, 130):
        grundy_number = grundy(state)
        best = find_best_moves(state)
        print(f"Grundy number: {grundy_number}") 
        grundy_numbers.append(grundy_number)
        best_moves.append([int(a) for a in best])
    else:
        grundy_numbers.append(None)
        best_moves.append(None)

    if mode == "ava":
        if player == 1:
            mcts_probs = mcts.search(state)
            action = np.argmax(mcts_probs)
            print(f"AlphaZero 1 Moves {action}")
                
        else:
            mcts_probs = mcts.search(state)
            action = np.argmax(mcts_probs)
            print(f"AlphaZero -1 Moves {action}")

    elif mode == "avp":
        if player == 1:
            mcts_probs = mcts.search(state)
            action = np.argmax(mcts_probs)
            print(f"AlphaZero 1 Moves {action}")

        else:
            valid_moves = chomp.get_valid_moves(state)
            print("valid_moves", [i for i in range(chomp.action_size) if valid_moves[i] == 1])
            action = int(input(f"{player}:"))

            if valid_moves[action] == 0:
                print("\n")
                print("action not valid")
                print("\n")
                continue 

    elif mode == "avo":
        if player == 1:
            mcts_probs = mcts.search(state)
            action = np.argmax(mcts_probs)
            print(f"AlphaZero -1 Moves {action}")
        else:
            action = find_best_moves(state)[0]
            print(f"Optimal 1 Moves {action}")
            
            
            
    state = chomp.get_next_state(state, action)
    move_sequence.append(int(action))
    value, is_terminal = chomp.get_value_and_terminated(state, action)
    
    if is_terminal:
        print(state)
        print(player, "lost")
        break
          
    player = chomp.get_opponent(player)

end = time.time()
print(f"Move sequence: {move_sequence}")
print(f"Grundy numbers: {grundy_numbers}")
print(f"Best moves: {best_moves}")
print(f"Elapsed time: {end - start:.2f} seconds")



data = {
    "metadata": metadata(args, checkpoint, mode=mode),
    "move_sequence": move_sequence,
    "grundy_numbers": grundy_numbers,
    "best_moves": best_moves
}

with open(OUTPUT_DIR / f"game_trace_{rows*cols}_{mode}.json", "w") as f:
    json.dump(data, f, indent=2)


plt.figure(figsize=(8, 5))
plt.plot(
    list(range(len(grundy_numbers))),
    grundy_numbers,
    marker='o',
    linestyle='--'
)

plt.xlabel("Move Number")
plt.ylabel("Grundy Number")
plt.title("Grundy Number vs. Move Number")
plt.grid(True)
plt.savefig(OUTPUT_DIR / f"grundy_{rows*cols}_{mode}.png", dpi=300, bbox_inches='tight')
if cli.show_plot:
    plt.show()
