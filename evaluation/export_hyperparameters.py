from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.common import EXPERIMENTS, REPO_ROOT


def load_config(path: Path) -> Dict[str, Any]:
    spec = importlib.util.spec_from_file_location(f"config_{abs(hash(path))}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load config from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return dict(module.args)


def board_label(config: Dict[str, Any]) -> str:
    if "rows" in config and "cols" in config:
        return f"{config['rows']}x{config['cols']}"
    return f"{config['row_count']}x{config['column_count']}"


def row_for_experiment(
    name: str,
    seed_count: int,
    checkpoint_rule: str | None,
) -> Dict[str, Any]:
    exp = EXPERIMENTS[name]
    config = load_config(REPO_ROOT / str(exp["src"]) / "config.py")
    if exp["game"] == "chomp":
        config["rows"] = exp["rows"]
        config["cols"] = exp["cols"]
    aux_weight = config.get("p_move_lambda", config.get("aux_weight", "N/A"))
    games_per_iteration = config.get("num_selfPlay_iterations")
    if exp["game"] == "connect4":
        input_desc = "3-plane"
    elif "num_frames" in config:
        input_desc = f"{config['num_frames']}-frame"
    else:
        input_desc = "1-plane"
    return {
        "Experiment": name,
        "Game": config.get("game_name", "Chomp" if exp["game"] == "chomp" else "Connect Four"),
        "Variant": config.get("variant", exp["variant"]),
        "Board": board_label(config),
        "Architecture": f"Dual-head Conv-ResNet ({input_desc} input)",
        "Residual blocks/channels": f"{config.get('num_resBlocks')}/{config.get('num_hidden')}",
        "MCTS simulations": config.get("num_searches"),
        "Training iterations": config.get("num_iterations"),
        "Games per iteration": games_per_iteration,
        "Replay buffer size": f"All states from {games_per_iteration} games (no cap)",
        "Batch size": config.get("batch_size"),
        "Learning rate": config.get("lr"),
        "Auxiliary-loss weight": aux_weight,
        "Temperature schedule": f"constant {config.get('temperature')}",
        "Dirichlet noise": f"epsilon={config.get('dirichlet_epsilon')}, alpha={config.get('dirichlet_alpha')}",
        "Checkpoint rule": checkpoint_rule or config.get("checkpoint_selection_rule", "last iteration checkpoint"),
        "Seed count": seed_count,
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_tex(path: Path, rows: List[Dict[str, Any]]) -> None:
    headers = list(rows[0])
    columns = "l" * len(headers)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Reproducibility appendix hyperparameters exported from checked-in config files.}",
        r"\label{tab:reproducibility_hyperparameters}",
        f"\\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(str(row[h]) for h in headers) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a hyperparameter table from variant config.py files."
    )
    parser.add_argument("--outdir", type=Path, default=REPO_ROOT / "Graph Creation" / "figures" / "tables")
    parser.add_argument("--seed-count", type=int, default=3)
    parser.add_argument("--checkpoint-rule", default=None)
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=sorted(EXPERIMENTS),
        default=sorted(EXPERIMENTS),
    )
    args = parser.parse_args()

    rows = [
        row_for_experiment(name, args.seed_count, args.checkpoint_rule)
        for name in args.experiments
    ]

    args.outdir.mkdir(parents=True, exist_ok=True)
    write_csv(args.outdir / "hyperparameters.csv", rows)
    write_tex(args.outdir / "hyperparameters.tex", rows)
    print(f"Wrote hyperparameter table to {args.outdir}")


if __name__ == "__main__":
    main()
