from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path.home() / "Documents" / "AlphaZero-Chomp"
csv_dir = BASE_DIR / "history"
out_dir = BASE_DIR / "figures"

out_dir.mkdir(parents=True, exist_ok=True)

LOSS_COLS = ["total_loss", "value_loss", "policy_loss"]
P_LOSS_COLS = ["p_move_loss"]
XCOL = "_step"

SMOOTH_WINDOW = 7
TRUNCATE_WITHIN_GROUP = True 

def maybe_smooth(series, window):
    if window is None or window <= 1:
        return series
    return series.rolling(window=window, min_periods=1).mean()

def load_run(path):
    path = Path(path)
    df = pd.read_csv(path)
    if XCOL not in df.columns:
        raise ValueError(f"{path.name} missing {XCOL}")
    for c in LOSS_COLS:
        if c not in df.columns:
            raise ValueError(f"{path.name} missing {c}")
    df = df.sort_values(XCOL).dropna(subset=[XCOL])
    return df

runs = {
    "9x9 Vanilla":    csv_dir / "run_history_9x9_Working.csv",
    "9x10 Vanilla":   csv_dir / "run_history_9x10_Working.csv",
    "9x10 Multi-Frame":   csv_dir / "run_history_9x10_Upgrade.csv",
    "9x10 Auxiliary-Loss":   csv_dir / "run_history_9x10_Auxiliary.csv",

    "10x10 Vanilla":  csv_dir / "run_history_10x10_Working.csv",
    "10x11 Vanilla":  csv_dir / "run_history_10x11_Working.csv",
    "10x11 Multi-Frame":  csv_dir / "run_history_10x11_Upgrade.csv",
    "10x11 Auxiliary-Loss":   csv_dir / "run_history_10x11_Auxiliary.csv",

    "Connect Four Vanilla": csv_dir / "run_history_6x7_C4_Working.csv",
    "Connect Four Auxiliary": csv_dir / "run_history_6x7_C4_Auxiliary.csv"
}

groups = {
    "9x9-9x10": ["9x9 Vanilla", "9x10 Vanilla", "9x10 Multi-Frame", "9x10 Auxiliary-Loss"],
    "10x10-10x11": ["10x10 Vanilla", "10x11 Vanilla", "10x11 Multi-Frame","10x11 Auxiliary-Loss"],
    "Connect Four": ["Connect Four Vanilla", "Connect Four Auxiliary"],
}
p_groups = {
    "9x10": ["9x10 Auxiliary-Loss"],
    "10x11": ["10x11 Auxiliary-Loss"],
    "Connect Four": ["Connect Four Auxiliary"]
}

dfs = {label: load_run(path) for label, path in runs.items()}

def plot_group(group_name, labels, loss_cols):
    if TRUNCATE_WITHIN_GROUP:
        common_max_step = min(dfs[l][XCOL].max() for l in labels)
    else:
        common_max_step = None

    for loss_col in loss_cols:
        plt.figure()

        for label in labels:
            df = dfs[label]
            plot_df = df
            if common_max_step is not None:
                plot_df = plot_df[plot_df[XCOL] <= common_max_step]

            y = maybe_smooth(plot_df[loss_col], SMOOTH_WINDOW)
            plt.plot(plot_df[XCOL], y, label=label)

        plt.xlabel("step (_step)")
        plt.ylabel(loss_col)
        plt.grid(True, alpha=0.25)

        title = f"{group_name}: {loss_col} vs step"
        if common_max_step is not None:
            title += f" (truncated @ {int(common_max_step)})"
        plt.title(title)

        plt.legend()

        out_path = out_dir / f"{group_name}__{loss_col}_vs_step.png"
        plt.savefig(out_path, dpi=200)
        plt.close()

for gname, labels in groups.items():
    plot_group(gname, labels, LOSS_COLS)

for gname, labels in p_groups.items():
    plot_group(gname, labels, P_LOSS_COLS)

print(f"Saved 6 merged plots to: {out_dir}")
