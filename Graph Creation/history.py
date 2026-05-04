import os
import wandb
import pandas as pd

ENTITY  = "brentkong-california-institute-of-technology-caltech"
PROJECT = "AlphaZero"
RUN_ID  = "cbgi8g5h"
NAME = "10x11_Auxiliary"

api = wandb.Api()
run = api.run(f"{ENTITY}/{PROJECT}/{RUN_ID}")

df = run.history(pandas = True) 
if "p_move_loss" in df.columns:
    print("Subtracting p_move_loss")
    df["total_loss"] -= df["p_move_loss"]
df.to_csv(f"/Users/brentkong/Documents/AlphaZero-Chomp/history/run_history_{NAME}.csv", index = False)


