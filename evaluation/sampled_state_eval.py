from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
from experiment_utils.runtime import metadata, parse_int_list


def valid_actions(game: Any, state: np.ndarray) -> np.ndarray:
    return np.flatnonzero(game.get_valid_moves(state))


def chomp_sample(game: Any, depth: int, rng: np.random.Generator) -> Optional[np.ndarray]:
    state = game.get_initial_state()
    for _ in range(depth):
        actions = valid_actions(game, state)
        if len(actions) == 0:
            return None
        action = int(rng.choice(actions))
        state = game.get_next_state(state, action)
        _, terminal = game.get_value_and_terminated(state, action)
        if terminal:
            return None
    return state


def connect4_sample(
    game: Any,
    depth: int,
    rng: np.random.Generator,
) -> Optional[Tuple[np.ndarray, List[int], int]]:
    state = game.get_initial_state()
    history: List[int] = []
    player = 1
    for _ in range(depth):
        actions = valid_actions(game, state)
        if len(actions) == 0:
            return None
        action = int(rng.choice(actions))
        history.append(action)
        state = game.get_next_state(state, action, player)
        _, terminal = game.get_value_and_terminated(state, action)
        if terminal:
            return None
        player = game.get_opponent(player)
    return state, history, player


def choose_action(exp: Any, state: np.ndarray, player: int = 1) -> int:
    if hasattr(exp.game, "change_perspective"):
        state = exp.game.change_perspective(state, player)
    probs = exp.mcts.search(state)
    return int(np.argmax(probs))


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    labeled = [r for r in records if r["labeled"]]
    matches = sum(1 for r in labeled if r["match"])
    failures = [r["failure_depth"] for r in records if r["failure_depth"] is not None]
    chains = [r["chain_length"] for r in records]
    return {
        "sampled_state_match_rate": matches / len(labeled) if labeled else math.nan,
        "sampled_state_matches": matches,
        "sampled_state_labeled": len(labeled),
        "mean_chain_length": float(np.mean(chains)) if chains else math.nan,
        "mean_failure_depth": float(np.mean(failures)) if failures else math.nan,
        "failure_count": len(failures),
        "sample_count": len(records),
    }


def chomp_chain(
    exp: Any,
    start_state: np.ndarray,
    start_depth: int,
    chain_steps: int,
    oracle_size: int,
) -> Tuple[int, Optional[int]]:
    grundy, fits_grundy_limit, find_best_moves = chomp_oracle_funcs()
    state = start_state.copy()
    chain = 0
    for step in range(chain_steps):
        if not fits_grundy_limit(state, oracle_size) or grundy(state) == 0:
            return chain, None
        best = find_best_moves(state)
        action = choose_action(exp, state)
        if action not in best:
            return chain, start_depth + step
        chain += 1
        state = exp.game.get_next_state(state, action)
        _, terminal = exp.game.get_value_and_terminated(state, action)
        if terminal:
            return chain, None
    return chain, None


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
    rng = np.random.default_rng(args.seed)
    grundy, fits_grundy_limit, find_best_moves = chomp_oracle_funcs()

    records: List[Dict[str, Any]] = []
    for depth in args.depths:
        for _ in range(args.samples):
            state = chomp_sample(exp.game, depth, rng)
            if state is None:
                continue
            labeled = fits_grundy_limit(state, args.oracle_size) and grundy(state) != 0
            best: Sequence[int] = find_best_moves(state) if labeled else []
            action = choose_action(exp, state) if labeled else None
            chain, failure_depth = chomp_chain(
                exp, state, depth, args.chain_steps, args.oracle_size
            )
            records.append(
                {
                    "depth": int(depth),
                    "labeled": bool(labeled),
                    "action": action,
                    "best_moves": [int(x) for x in best],
                    "match": bool(action in best) if labeled else None,
                    "chain_length": int(chain),
                    "failure_depth": failure_depth,
                }
            )
    return {"metadata": metadata(exp.config, args.checkpoint), "records": records, "summary": summarize(records)}


def connect4_chain(
    exp: Any,
    oracle: Any,
    start_state: np.ndarray,
    start_history: List[int],
    start_player: int,
    start_depth: int,
    chain_steps: int,
) -> Tuple[int, Optional[int]]:
    state = start_state.copy()
    history = list(start_history)
    player = start_player
    chain = 0
    for step in range(chain_steps):
        result = oracle.best_move(state, history)
        if result is None:
            return chain, None
        best, _ = result
        action = choose_action(exp, state, player)
        if action not in best:
            return chain, start_depth + step
        chain += 1
        history.append(action)
        state = exp.game.get_next_state(state, action, player)
        _, terminal = exp.game.get_value_and_terminated(state, action)
        if terminal:
            return chain, None
        player = exp.game.get_opponent(player)
    return chain, None


def eval_connect4(args: argparse.Namespace) -> Dict[str, Any]:
    exp = load_experiment(
        args.src_dir,
        args.checkpoint,
        game_name="connect4",
        seed=args.seed,
        num_searches=args.num_searches,
    )
    exp.config["game_name"] = "Connect Four"
    rng = np.random.default_rng(args.seed)
    oracle = connect4_oracle(exp.game)

    records: List[Dict[str, Any]] = []
    try:
        for depth in args.depths:
            samples = []
            for _ in range(args.samples):
                sample = connect4_sample(exp.game, depth, rng)
                if sample is not None:
                    samples.append(sample)
            if not samples:
                continue

            states = np.stack([s[0] for s in samples])
            histories = [s[1] for s in samples]
            players = [s[2] for s in samples]
            oracle_results = oracle.best_move_batch(states, histories)

            for (state, history, player), oracle_result in zip(samples, oracle_results):
                if oracle_result is None:
                    continue
                best, scores = oracle_result
                action = choose_action(exp, state, player)
                chain, failure_depth = connect4_chain(
                    exp, oracle, state, history, player, depth, args.chain_steps
                )
                records.append(
                    {
                        "depth": int(depth),
                        "labeled": True,
                        "action": int(action),
                        "best_moves": [int(x) for x in best],
                        "scores": [int(x) for x in scores],
                        "match": bool(action in best),
                        "chain_length": int(chain),
                        "failure_depth": failure_depth,
                    }
                )
    finally:
        oracle.close()

    return {"metadata": metadata(exp.config, args.checkpoint), "records": records, "summary": summarize(records)}


def write_csv(path: Path, records: List[Dict[str, Any]]) -> None:
    fields = ["depth", "labeled", "action", "best_moves", "match", "chain_length", "failure_depth"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field) for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate oracle consistency on sampled states.")
    parser.add_argument("--game", choices=("chomp", "connect4"), required=True)
    parser.add_argument("--src-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=REPO_ROOT / "results" / "sampled")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--cols", type=int, default=None)
    parser.add_argument("--num-searches", type=int, default=None)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--depths", nargs="+", default=["0", "4", "8", "12"])
    parser.add_argument("--chain-steps", type=int, default=8)
    parser.add_argument("--oracle-size", type=int, default=130)
    args = parser.parse_args()
    args.depths = parse_int_list(args.depths)

    result = eval_chomp(args) if args.game == "chomp" else eval_connect4(args)

    args.outdir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.game}_sampled_seed_{args.seed}"
    with (args.outdir / f"{stem}.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    write_csv(args.outdir / f"{stem}.csv", result["records"])
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
