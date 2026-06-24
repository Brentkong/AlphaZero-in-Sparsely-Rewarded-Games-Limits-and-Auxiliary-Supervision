from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.common import (
    REPO_ROOT,
    chomp_oracle_funcs,
    connect4_oracle,
    load_experiment,
)
from evaluation.sampled_state_eval import choose_action
from experiment_utils.runtime import evaluation_dir, metadata


def class_from_value(value: Optional[float]) -> str:
    if value is None or math.isnan(float(value)):
        return "unknown"
    if value > 0:
        return "winning"
    if value < 0:
        return "losing"
    return "draw"


def final_outcome(record_player: int, terminal_player: int, terminal_value: float, game: Any) -> float:
    return (
        terminal_value
        if record_player == terminal_player
        else game.get_opponent_value(terminal_value)
    )


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    labeled = [r for r in records if r["oracle_class"] in {"winning", "losing"}]
    losing_to_win = [
        r for r in labeled if r["oracle_class"] == "losing" and r["final_outcome"] > 0
    ]
    winning_failed = [
        r for r in labeled if r["oracle_class"] == "winning" and r["final_outcome"] <= 0
    ]
    return {
        "visited_states": len(records),
        "labeled_states": len(labeled),
        "oracle_losing_but_selfplay_won": len(losing_to_win),
        "oracle_winning_but_failed_to_convert": len(winning_failed),
    }


def chomp_oracle_label(state: np.ndarray, oracle_size: int) -> Tuple[Optional[int], str]:
    grundy, fits_grundy_limit, _ = chomp_oracle_funcs()
    if not fits_grundy_limit(state, oracle_size):
        return None, "unknown"
    value = int(grundy(state))
    return value, "winning" if value != 0 else "losing"


def eval_chomp(args: argparse.Namespace) -> Dict[str, Any]:
    exp = load_experiment(
        args.src_dir,
        args.checkpoint,
        game_name="chomp",
        seed=args.seed,
        rows=args.rows,
        cols=args.cols,
        num_searches=args.num_searches,
    )
    exp.config["game_name"] = "Chomp"
    records: List[Dict[str, Any]] = []

    for episode in range(args.episodes):
        state = exp.game.get_initial_state()
        player = 1
        trajectory: List[Dict[str, Any]] = []
        terminal_value = 0.0
        terminal_player = player

        for ply in range(args.max_plies):
            oracle_value, oracle_class = chomp_oracle_label(state, args.oracle_size)
            trajectory.append(
                {
                    "episode": episode,
                    "ply": ply,
                    "player": player,
                    "oracle_value": oracle_value,
                    "oracle_class": oracle_class,
                }
            )
            action = choose_action(exp, state)
            state = exp.game.get_next_state(state, action)
            value, terminal = exp.game.get_value_and_terminated(state, action)
            if terminal:
                terminal_value = float(value)
                terminal_player = player
                break
            player = exp.game.get_opponent(player)

        for row in trajectory:
            outcome = final_outcome(row["player"], terminal_player, terminal_value, exp.game)
            row["final_outcome"] = outcome
            records.append(row)

    return {"metadata": metadata(exp.config, args.checkpoint), "records": records, "summary": summarize(records)}


def connect4_oracle_label(oracle: Any, game: Any, state: np.ndarray, history: List[int]) -> Tuple[Optional[int], str]:
    result = oracle.best_move(state, history)
    if result is None:
        return None, "unknown"
    _, scores = result
    valid = game.get_valid_moves(state).astype(bool)
    masked = [scores[i] if valid[i] else -10**9 for i in range(game.action_size)]
    best = int(max(masked))
    return best, class_from_value(best)


def eval_connect4(args: argparse.Namespace) -> Dict[str, Any]:
    exp = load_experiment(
        args.src_dir,
        args.checkpoint,
        game_name="connect4",
        seed=args.seed,
        num_searches=args.num_searches,
    )
    exp.config["game_name"] = "Connect Four"
    oracle = connect4_oracle(exp.game)
    records: List[Dict[str, Any]] = []

    try:
        for episode in range(args.episodes):
            state = exp.game.get_initial_state()
            history: List[int] = []
            player = 1
            trajectory: List[Dict[str, Any]] = []
            terminal_value = 0.0
            terminal_player = player

            for ply in range(args.max_plies):
                oracle_value, oracle_class = connect4_oracle_label(oracle, exp.game, state, history)
                trajectory.append(
                    {
                        "episode": episode,
                        "ply": ply,
                        "player": player,
                        "oracle_value": oracle_value,
                        "oracle_class": oracle_class,
                    }
                )
                action = choose_action(exp, state, player)
                history.append(action)
                state = exp.game.get_next_state(state, action, player)
                value, terminal = exp.game.get_value_and_terminated(state, action)
                if terminal:
                    terminal_value = float(value)
                    terminal_player = player
                    break
                player = exp.game.get_opponent(player)

            for row in trajectory:
                outcome = final_outcome(row["player"], terminal_player, terminal_value, exp.game)
                row["final_outcome"] = outcome
                records.append(row)
    finally:
        oracle.close()

    return {"metadata": metadata(exp.config, args.checkpoint), "records": records, "summary": summarize(records)}


def write_csv(path: Path, records: List[Dict[str, Any]]) -> None:
    fields = [
        "episode",
        "ply",
        "player",
        "oracle_value",
        "oracle_class",
        "final_outcome",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({field: row.get(field) for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose moving-target self-play labels against oracle classes."
    )
    parser.add_argument("--game", choices=("chomp", "connect4"), required=True)
    parser.add_argument("--src-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--cols", type=int, default=None)
    parser.add_argument("--num-searches", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--max-plies", type=int, default=80)
    parser.add_argument("--oracle-size", type=int, default=130)
    args = parser.parse_args()

    result = eval_chomp(args) if args.game == "chomp" else eval_connect4(args)

    args.outdir = evaluation_dir(REPO_ROOT, result["metadata"], "diagnostics", args.outdir)
    args.outdir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.game}_moving_target_seed_{args.seed}"
    with (args.outdir / f"{stem}.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    write_csv(args.outdir / f"{stem}.csv", result["records"])
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
