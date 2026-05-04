import time
import json
import torch
import numpy as np
from config import args
from mcts import MCTS
from chomp import Chomp
from resnet import ResNet
import matplotlib.pyplot as plt
from grundy_oracle import grundy, fits_grundy_limit, find_best_moves
torch.manual_seed(10)

rows, cols = args['rows'], args['cols']
chomp = Chomp(rows, cols)
player = 1
mode = "avo"

state = chomp.get_initial_state()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet(chomp, args['num_resBlocks'], args['num_hidden'], device) 
model.load_state_dict(torch.load("/Users/brentkong/Documents/AlphaZero-Chomp/weights/Chomp/Auxilary/90/model_9_Chomp(9x10).pt", map_location=device))
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
            print(f"AlphaZero 1 Moves {action}")
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
    "move_sequence": move_sequence,
    "grundy_numbers": grundy_numbers,
    "best_moves": best_moves
}

with open(f"/Users/brentkong/Documents/AlphaZero-Chomp/figures/game_trace_{rows*cols}_{mode}.json", "w") as f:
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
plt.savefig(f'/Users/brentkong/Documents/AlphaZero-Chomp/figures/grundy_{rows*cols}_{mode}.png', dpi=300, bbox_inches='tight')
plt.show()