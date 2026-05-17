import os
from pathlib import Path

import wandb
import pandas as pd

ENTITY  = os.environ.get("WANDB_ENTITY")
PROJECT = "AlphaZero"
RUN_ID  = "cbgi8g5h"
NAME = "10x11_Auxiliary"
BASE_DIR = Path.home() / "Documents" / "AlphaZero-Chomp"
HISTORY_DIR = BASE_DIR / "history"

if ENTITY is None:
    raise RuntimeError("Set WANDB_ENTITY to your Weights & Biases entity before running this script.")

api = wandb.Api()
run = api.run(f"{ENTITY}/{PROJECT}/{RUN_ID}")

df = run.history(pandas = True) 
if "p_move_loss" in df.columns:
    print("Subtracting p_move_loss")
    df["total_loss"] -= df["p_move_loss"]
df.to_csv(HISTORY_DIR / f"run_history_{NAME}.csv", index = False)
