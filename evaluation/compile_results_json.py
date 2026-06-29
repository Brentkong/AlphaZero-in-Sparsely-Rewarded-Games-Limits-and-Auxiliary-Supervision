from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiment_utils.runtime import mean_std


TRACE_RE = re.compile(r"game_trace_(?P<size>\d+)_ava_trace_(?P<trace>\d+)\.json$")


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_game(value: str) -> str:
    return {"Connect-Four": "Connect Four"}.get(value, value)


def normalize_variant(value: str) -> str:
    return {"MultiFrame": "Multi-Frame"}.get(value, value)


def run_parts(path: Path, results_dir: Path) -> Dict[str, Any]:
    rel = path.relative_to(results_dir)
    parts = rel.parts
    data: Dict[str, Any] = {
        "game_dir": parts[0] if len(parts) > 0 else "",
        "variant_dir": parts[1] if len(parts) > 1 else "",
        "board_dir": parts[2] if len(parts) > 2 else "",
        "seed_dir": parts[3] if len(parts) > 3 else "",
    }
    seed_dir = str(data["seed_dir"])
    data["seed"] = int(seed_dir.removeprefix("seed_")) if seed_dir.startswith("seed_") else None
    return data


def trace_index(path: Path) -> Optional[int]:
    match = TRACE_RE.match(path.name)
    return int(match.group("trace")) if match else None


def config_metadata(path: Path, config: Dict[str, Any], results_dir: Path) -> Dict[str, Any]:
    parts = run_parts(path, results_dir)
    game = normalize_game(str(config.get("game_name", parts["game_dir"])))
    variant = normalize_variant(str(config.get("variant", parts["variant_dir"])))
    board = str(config.get("board", parts["board_dir"]))
    if "rows" in config and "cols" in config:
        board = f"{config['rows']}x{config['cols']}"
    elif "row_count" in config and "column_count" in config:
        board = f"{config['row_count']}x{config['column_count']}"
    return {
        "game": game,
        "variant": variant,
        "board": board,
        "seed": config.get("seed", parts["seed"]),
        "source": str(path),
    }


def trace_metadata(path: Path, trace: Dict[str, Any], results_dir: Path) -> Dict[str, Any]:
    parts = run_parts(path, results_dir)
    meta = dict(trace.get("metadata", {}))
    return {
        "game": normalize_game(str(meta.get("game", parts["game_dir"]))),
        "variant": normalize_variant(str(meta.get("variant", parts["variant_dir"]))),
        "board": str(meta.get("board", parts["board_dir"])),
        "seed": meta.get("seed", parts["seed"]),
        "checkpoint": meta.get("checkpoint", ""),
        "trace_index": meta.get("trace_index", trace_index(path)),
        "source": str(path),
    }


def oracle_match_flags(trace: Dict[str, Any]) -> List[Optional[bool]]:
    moves = trace.get("move_sequence", [])
    best_moves = trace.get("best_moves", [])
    flags: List[Optional[bool]] = []
    for move, best in zip(moves, best_moves):
        if best is None or len(best) == 0:
            flags.append(None)
        else:
            flags.append(move in best)
    return flags


def player_flags(flags: List[Optional[bool]], player: Optional[int]) -> List[Optional[bool]]:
    if player is None:
        return flags
    return [flags[idx] for idx in range(player, len(flags), 2)]


def count_flags(flags: Iterable[Optional[bool]]) -> Dict[str, Any]:
    labeled = [flag for flag in flags if flag is not None]
    matches = sum(1 for flag in labeled if flag is True)
    total = len(labeled)
    rate = matches / total if total else math.nan
    return {"matches": matches, "labeled": total, "rate": rate}


def longest_true_run(flags: Iterable[Optional[bool]]) -> Optional[int]:
    seen_labeled = False
    best = 0
    current = 0
    for flag in flags:
        if flag is True:
            seen_labeled = True
            current += 1
            best = max(best, current)
        elif flag is False:
            seen_labeled = True
            current = 0
    return best if seen_labeled else None


def true_prefix_len(flags: Iterable[Optional[bool]]) -> Optional[int]:
    seen_labeled = False
    prefix = 0
    for flag in flags:
        if flag is None:
            continue
        seen_labeled = True
        if flag is True:
            prefix += 1
        else:
            return prefix
    return prefix if seen_labeled else None


def first_failure_ply(flags: List[Optional[bool]], player: Optional[int]) -> Optional[int]:
    if player is None:
        for idx, flag in enumerate(flags):
            if flag is False:
                return idx
        return None
    for local_idx, flag in enumerate(player_flags(flags, player)):
        if flag is False:
            return player + 2 * local_idx
    return None


def chomp_preserve_counts(trace: Dict[str, Any], player: Optional[int]) -> Dict[str, Any]:
    grundy = trace.get("grundy_numbers")
    if not isinstance(grundy, list):
        return {"preserved": 0, "winning_turns": 0, "rate": math.nan}

    players = [0, 1] if player is None else [player]
    preserved = 0
    winning_turns = 0
    for current_player in players:
        for idx in range(current_player, len(grundy) - 1, 2):
            if grundy[idx] != 0:
                winning_turns += 1
                if grundy[idx + 1] == 0:
                    preserved += 1
    rate = preserved / winning_turns if winning_turns else math.nan
    return {"preserved": preserved, "winning_turns": winning_turns, "rate": rate}


def metric_block(trace: Dict[str, Any], flags: List[Optional[bool]], player: Optional[int]) -> Dict[str, Any]:
    selected = player_flags(flags, player)
    counts = count_flags(selected)
    preserve = chomp_preserve_counts(trace, player)
    return {
        "oracle_match": counts,
        "longest_oracle_chain": longest_true_run(selected),
        "oracle_consistent_prefix": true_prefix_len(selected),
        "first_non_oracle_ply": first_failure_ply(flags, player),
        "preserve_to_zero": preserve,
    }


def summarize_trace(path: Path, results_dir: Path) -> Dict[str, Any]:
    trace = load_json(path)
    flags = oracle_match_flags(trace)
    meta = trace_metadata(path, trace, results_dir)
    moves = trace.get("move_sequence", [])
    return {
        **meta,
        "move_count": len(moves),
        "sampled_selfplay": sampled_selfplay_counts(trace),
        "players": {
            "all": metric_block(trace, flags, None),
            "p1": metric_block(trace, flags, 0),
            "p2": metric_block(trace, flags, 1),
        },
    }


def finite(value: Any) -> bool:
    return value is not None and not (isinstance(value, float) and math.isnan(value))


def stat_block(values: Iterable[Any], digits: int = 3) -> Dict[str, Any]:
    mean, std, n = mean_std([float(value) for value in values if finite(value)])
    formatted = "N/A" if n == 0 else f"{mean:.{digits}f} +/- {std:.{digits}f}"
    return {"mean": mean, "std": std, "n": n, "formatted": formatted}


def rate_count_block(matches: int, labeled: int, digits: int = 3) -> Dict[str, Any]:
    rate = matches / labeled if labeled else math.nan
    formatted = "N/A" if labeled == 0 else f"{rate:.{digits}f} ({matches}/{labeled})"
    return {
        "rate": rate,
        "matches": matches,
        "labeled": labeled,
        "formatted": formatted,
    }


def row_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
    return row["game"], row["variant"], row["board"]


def sorted_seeds(rows: List[Dict[str, Any]]) -> List[int]:
    seeds = [row.get("seed") for row in rows]
    return sorted(int(seed) for seed in seeds if seed is not None and seed != "")


def aggregate_trace_rows(rows: List[Dict[str, Any]], player_key: str) -> Dict[str, Any]:
    matches = 0
    labeled = 0
    preserve_hits = 0
    preserve_total = 0
    chains: List[float] = []
    prefixes: List[float] = []
    failures: List[float] = []
    move_counts: List[float] = []

    for row in rows:
        metrics = row["players"][player_key]
        match = metrics["oracle_match"]
        matches += int(match["matches"])
        labeled += int(match["labeled"])
        preserve = metrics["preserve_to_zero"]
        preserve_hits += int(preserve["preserved"])
        preserve_total += int(preserve["winning_turns"])
        if finite(metrics["longest_oracle_chain"]):
            chains.append(float(metrics["longest_oracle_chain"]))
        if finite(metrics["oracle_consistent_prefix"]):
            prefixes.append(float(metrics["oracle_consistent_prefix"]))
        if finite(metrics["first_non_oracle_ply"]):
            failures.append(float(metrics["first_non_oracle_ply"]))
        move_counts.append(float(row["move_count"]))

    chain_mean, chain_std, chain_n = mean_std(chains)
    prefix_mean, prefix_std, prefix_n = mean_std(prefixes)
    failure_mean, failure_std, failure_n = mean_std(failures)
    move_mean, move_std, move_n = mean_std(move_counts)

    return {
        "trace_count": len(rows),
        "move_count": {"mean": move_mean, "std": move_std, "n": move_n},
        "oracle_match": {
            "matches": matches,
            "labeled": labeled,
            "rate": matches / labeled if labeled else math.nan,
        },
        "longest_oracle_chain": {"mean": chain_mean, "std": chain_std, "n": chain_n},
        "oracle_consistent_prefix": {"mean": prefix_mean, "std": prefix_std, "n": prefix_n},
        "first_non_oracle_ply": {"mean": failure_mean, "std": failure_std, "n": failure_n},
        "preserve_to_zero": {
            "preserved": preserve_hits,
            "winning_turns": preserve_total,
            "rate": preserve_hits / preserve_total if preserve_total else math.nan,
        },
    }


def aggregate_by(
    trace_rows: List[Dict[str, Any]],
    key_fields: Tuple[str, ...],
) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in trace_rows:
        groups[tuple(row.get(field) for field in key_fields)].append(row)

    out: List[Dict[str, Any]] = []
    for key, rows in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        data = {field: value for field, value in zip(key_fields, key)}
        data["trace_count"] = len(rows)
        data["players"] = {
            "all": aggregate_trace_rows(rows, "all"),
            "p1": aggregate_trace_rows(rows, "p1"),
            "p2": aggregate_trace_rows(rows, "p2"),
        }
        out.append(data)
    return out


def seed_row_map(per_seed_rows: List[Dict[str, Any]]) -> Dict[Tuple[Any, ...], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in per_seed_rows:
        grouped[(row["game"], row["variant"], row["board"])].append(row)
    return grouped


def robustness_rows(per_seed_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for (game, variant, board), rows in sorted(seed_row_map(per_seed_rows).items()):
        for player in ("all", "p1", "p2"):
            rates = [r["players"][player]["oracle_match"]["rate"] for r in rows]
            chains = [r["players"][player]["longest_oracle_chain"]["mean"] for r in rows]
            prefixes = [r["players"][player]["oracle_consistent_prefix"]["mean"] for r in rows]
            failures = [r["players"][player]["first_non_oracle_ply"]["mean"] for r in rows]
            rate_mean, rate_std, rate_n = mean_std(rates)
            chain_mean, chain_std, chain_n = mean_std(chains)
            prefix_mean, prefix_std, prefix_n = mean_std(prefixes)
            failure_mean, failure_std, failure_n = mean_std(failures)
            out.append(
                {
                    "game": game,
                    "variant": variant,
                    "board": board,
                    "player": player,
                    "seed_count": len(rows),
                    "oracle_match_rate": {"mean": rate_mean, "std": rate_std, "n": rate_n},
                    "longest_oracle_chain": {"mean": chain_mean, "std": chain_std, "n": chain_n},
                    "oracle_consistent_prefix": {"mean": prefix_mean, "std": prefix_std, "n": prefix_n},
                    "first_non_oracle_ply": {"mean": failure_mean, "std": failure_std, "n": failure_n},
                }
            )
    return out


def hyperparameter_rows(results_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    fields = [
        "num_resBlocks",
        "num_hidden",
        "num_searches",
        "num_iterations",
        "num_selfPlay_iterations",
        "num_parallel_games",
        "num_epochs",
        "batch_size",
        "lr",
        "weight_decay",
        "dirichlet_epsilon",
        "dirichlet_alpha",
        "temperature",
        "p_move_lambda",
        "aux_weight",
        "num_frames",
        "checkpoint_selection_rule",
    ]
    for path in sorted(results_dir.rglob("config_snapshot.json")):
        config = load_json(path)
        meta = config_metadata(path, config, results_dir)
        row: Dict[str, Any] = {**meta}
        for field in fields:
            if field in config:
                row[field] = config[field]
        rows.append(row)
    return rows


def sampled_rows(results_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(results_dir.rglob("*.json")):
        if path.name.startswith("game_trace_") or path.name == "config_snapshot.json":
            continue
        try:
            data = load_json(path)
        except (json.JSONDecodeError, OSError):
            continue
        if "summary" not in data or "records" not in data:
            continue
        meta = data.get("metadata", {})
        summary = data.get("summary", {})
        if "sampled_state_match_rate" not in summary:
            continue
        rows.append(
            {
                "game": normalize_game(str(meta.get("game", ""))),
                "variant": normalize_variant(str(meta.get("variant", ""))),
                "board": str(meta.get("board", "")),
                "seed": meta.get("seed", ""),
                "checkpoint": meta.get("checkpoint", ""),
                "source": str(path),
                **summary,
            }
        )
    return rows


def moving_target_rows(results_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(results_dir.rglob("*.json")):
        if path.name.startswith("game_trace_") or path.name == "config_snapshot.json":
            continue
        try:
            data = load_json(path)
        except (json.JSONDecodeError, OSError):
            continue
        if "summary" not in data or "records" not in data:
            continue
        meta = data.get("metadata", {})
        summary = data.get("summary", {})
        if "oracle_losing_but_selfplay_won" not in summary:
            continue
        records = data.get("records", [])
        oracle_losing = sum(1 for record in records if record.get("oracle_class") == "losing")
        oracle_winning = sum(1 for record in records if record.get("oracle_class") == "winning")
        rows.append(
            {
                "game": normalize_game(str(meta.get("game", ""))),
                "variant": normalize_variant(str(meta.get("variant", ""))),
                "board": str(meta.get("board", "")),
                "seed": meta.get("seed", ""),
                "checkpoint": meta.get("checkpoint", ""),
                "source": str(path),
                "oracle_losing_states": oracle_losing,
                "oracle_winning_states": oracle_winning,
                **summary,
            }
        )
    return rows


def grouped_by_experiment(rows: List[Dict[str, Any]]) -> Dict[Tuple[Any, ...], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row_key(row)].append(row)
    return grouped


def main_trace_robustness_table(
    per_seed_rows: List[Dict[str, Any]],
    aggregate_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    aggregate_by_key = {row_key(row): row for row in aggregate_rows}
    rows: List[Dict[str, Any]] = []
    for key, seed_rows in sorted(grouped_by_experiment(per_seed_rows).items()):
        game, variant, board = key
        aggregate = aggregate_by_key[key]
        aggregate_metrics = aggregate["players"]["all"]
        match = aggregate_metrics["oracle_match"]
        rows.append(
            {
                "game": game,
                "board": board,
                "model": variant,
                "runs": len(seed_rows),
                "seeds": sorted_seeds(seed_rows),
                "traces": aggregate["trace_count"],
                "oracle_match_rate_over_runs": stat_block(
                    row["players"]["all"]["oracle_match"]["rate"] for row in seed_rows
                ),
                "oracle_match_pooled_denominator": rate_count_block(
                    int(match["matches"]), int(match["labeled"])
                ),
                "longest_oracle_consistent_chain_over_runs": stat_block(
                    row["players"]["all"]["longest_oracle_chain"]["mean"] for row in seed_rows
                ),
                "first_non_oracle_ply_over_runs": stat_block(
                    row["players"]["all"]["first_non_oracle_ply"]["mean"] for row in seed_rows
                ),
                "mean_trace_length_over_runs": stat_block(
                    row["players"]["all"]["move_count"]["mean"] for row in seed_rows
                ),
            }
        )
    return rows


def player_split_trace_table(
    per_seed_rows: List[Dict[str, Any]],
    aggregate_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    aggregate_by_key = {row_key(row): row for row in aggregate_rows}
    rows: List[Dict[str, Any]] = []
    for key, seed_rows in sorted(grouped_by_experiment(per_seed_rows).items()):
        game, variant, board = key
        aggregate = aggregate_by_key[key]
        for player_key, label in (("p1", "P1"), ("p2", "P2")):
            metrics = aggregate["players"][player_key]
            match = metrics["oracle_match"]
            preserve = metrics["preserve_to_zero"]
            rows.append(
                {
                    "game": game,
                    "board": board,
                    "model": variant,
                    "player": label,
                    "runs": len(seed_rows),
                    "seeds": sorted_seeds(seed_rows),
                    "traces": aggregate["trace_count"],
                    "oracle_match_rate_over_runs": stat_block(
                        row["players"][player_key]["oracle_match"]["rate"] for row in seed_rows
                    ),
                    "oracle_match_pooled_denominator": rate_count_block(
                        int(match["matches"]), int(match["labeled"])
                    ),
                    "longest_oracle_consistent_chain_over_runs": stat_block(
                        row["players"][player_key]["longest_oracle_chain"]["mean"] for row in seed_rows
                    ),
                    "oracle_consistent_prefix_over_runs": stat_block(
                        row["players"][player_key]["oracle_consistent_prefix"]["mean"] for row in seed_rows
                    ),
                    "first_non_oracle_ply_over_runs": stat_block(
                        row["players"][player_key]["first_non_oracle_ply"]["mean"] for row in seed_rows
                    ),
                    "chomp_preserve_to_g0_denominator": rate_count_block(
                        int(preserve["preserved"]), int(preserve["winning_turns"])
                    ),
                }
            )
    return rows


def model_macro_average_table(main_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in main_rows:
        grouped[str(row["model"])].append(row)

    rows: List[Dict[str, Any]] = []
    for model, model_rows in sorted(grouped.items()):
        rows.append(
            {
                "model": model,
                "experiment_count": len(model_rows),
                "experiments": [
                    {"game": row["game"], "board": row["board"]} for row in model_rows
                ],
                "oracle_match_rate_macro_average": stat_block(
                    row["oracle_match_rate_over_runs"]["mean"] for row in model_rows
                ),
                "longest_oracle_consistent_chain_macro_average": stat_block(
                    row["longest_oracle_consistent_chain_over_runs"]["mean"]
                    for row in model_rows
                ),
            }
        )
    return rows


def hyperparameter_table(config_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ignore = {"game", "variant", "board", "seed", "source"}
    grouped = grouped_by_experiment(config_rows)
    rows: List[Dict[str, Any]] = []
    for (game, variant, board), group in sorted(grouped.items()):
        row: Dict[str, Any] = {
            "game": game,
            "board": board,
            "model": variant,
            "seed_count": len(group),
            "seeds": sorted_seeds(group),
        }
        fields = sorted({key for item in group for key in item if key not in ignore})
        for field in fields:
            values = [item[field] for item in group if field in item]
            unique = []
            seen = set()
            for value in values:
                marker = json.dumps(value, sort_keys=True, default=str)
                if marker not in seen:
                    seen.add(marker)
                    unique.append(value)
            row[field] = unique[0] if len(unique) == 1 else unique
        rows.append(row)
    return rows


def sampled_state_table(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for (game, variant, board), group in sorted(grouped_by_experiment(rows).items()):
        matches = sum(int(row.get("sampled_state_matches", 0)) for row in group)
        labeled = sum(int(row.get("sampled_state_labeled", 0)) for row in group)
        out.append(
            {
                "game": game,
                "board": board,
                "model": variant,
                "runs": len(group),
                "seeds": sorted_seeds(group),
                "sample_count": sum(int(row.get("sample_count", 0)) for row in group),
                "oracle_match_rate_over_runs": stat_block(
                    row.get("sampled_state_match_rate") for row in group
                ),
                "oracle_match_pooled_denominator": rate_count_block(matches, labeled),
                "mean_chain_length_over_runs": stat_block(
                    row.get("mean_chain_length") for row in group
                ),
                "mean_failure_depth_over_runs": stat_block(
                    row.get("mean_failure_depth") for row in group
                ),
            }
        )
    return out


def moving_target_table(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for (game, variant, board), group in sorted(grouped_by_experiment(rows).items()):
        losing_won = sum(int(row.get("oracle_losing_but_selfplay_won", 0)) for row in group)
        losing_total = sum(int(row.get("oracle_losing_states", 0)) for row in group)
        winning_failed = sum(int(row.get("oracle_winning_but_failed_to_convert", 0)) for row in group)
        winning_total = sum(int(row.get("oracle_winning_states", 0)) for row in group)
        out.append(
            {
                "game": game,
                "board": board,
                "model": variant,
                "runs": len(group),
                "seeds": sorted_seeds(group),
                "visited_states": sum(int(row.get("visited_states", 0)) for row in group),
                "labeled_states": sum(int(row.get("labeled_states", 0)) for row in group),
                "oracle_losing_but_selfplay_won": rate_count_block(losing_won, losing_total),
                "oracle_winning_but_failed_to_convert": rate_count_block(
                    winning_failed, winning_total
                ),
            }
        )
    return out


def maybe_float(value: Any) -> Optional[float]:
    return float(value) if finite(value) else None


def flat_mean_std(values: Iterable[Any]) -> Tuple[Optional[float], Optional[float], int]:
    mean, std, n = mean_std([float(value) for value in values if finite(value)])
    if n == 0:
        return None, None, 0
    return mean, std, n


def flat_rate(matches: int, labeled: int) -> Optional[float]:
    return matches / labeled if labeled else None


def unique_run_count(rows: List[Dict[str, Any]]) -> int:
    return len({(row["board"], row["seed"]) for row in rows})


def trace_values(rows: List[Dict[str, Any]], player: str, field: str) -> List[Any]:
    return [row["players"][player][field] for row in rows]


def pooled_match(rows: List[Dict[str, Any]], player: str) -> Tuple[int, int, Optional[float]]:
    matches = sum(int(row["players"][player]["oracle_match"]["matches"]) for row in rows)
    labeled = sum(int(row["players"][player]["oracle_match"]["labeled"]) for row in rows)
    return matches, labeled, flat_rate(matches, labeled)


def pooled_preserve(rows: List[Dict[str, Any]]) -> Tuple[int, int, Optional[float]]:
    preserved = sum(int(row["players"]["all"]["preserve_to_zero"]["preserved"]) for row in rows)
    labeled = sum(int(row["players"]["all"]["preserve_to_zero"]["winning_turns"]) for row in rows)
    return preserved, labeled, flat_rate(preserved, labeled)


def add_mean_std_fields(
    row: Dict[str, Any],
    prefix: str,
    values: Iterable[Any],
    *,
    include_n: bool = True,
) -> None:
    mean, std, n = flat_mean_std(values)
    row[f"{prefix}_mean"] = mean
    row[f"{prefix}_std"] = std
    if include_n:
        row[f"{prefix}_n"] = n


def connect4_winner(moves: List[int]) -> Optional[int]:
    rows, cols = 6, 7
    board = [[0 for _ in range(cols)] for _ in range(rows)]

    def won(row: int, col: int, player: int) -> bool:
        for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
            total = 1
            for sign in (1, -1):
                r, c = row + sign * dr, col + sign * dc
                while 0 <= r < rows and 0 <= c < cols and board[r][c] == player:
                    total += 1
                    r += sign * dr
                    c += sign * dc
            if total >= 4:
                return True
        return False

    for idx, action in enumerate(moves):
        if not isinstance(action, int) or action < 0 or action >= cols:
            return None
        player = 1 if idx % 2 == 0 else -1
        open_rows = [r for r in range(rows - 1, -1, -1) if board[r][action] == 0]
        if not open_rows:
            return None
        row = open_rows[0]
        board[row][action] = player
        if won(row, action, player):
            return player
    return 0 if len(moves) == rows * cols else None


def final_outcomes(trace: Dict[str, Any], count: int) -> List[Optional[int]]:
    moves = trace.get("move_sequence", [])
    if not isinstance(moves, list) or not moves:
        return [None] * count
    if isinstance(trace.get("grundy_numbers"), list):
        loser = 1 if (len(moves) - 1) % 2 == 0 else -1
        return [-1 if (1 if idx % 2 == 0 else -1) == loser else 1 for idx in range(count)]
    winner = connect4_winner([int(move) for move in moves])
    if winner is None:
        return [None] * count
    if winner == 0:
        return [0] * count
    return [1 if (1 if idx % 2 == 0 else -1) == winner else -1 for idx in range(count)]


def sampled_selfplay_counts(trace: Dict[str, Any]) -> Dict[str, int]:
    values = trace.get("grundy_numbers")
    chomp = isinstance(values, list)
    if not chomp:
        values = trace.get("score_state")
    if not isinstance(values, list):
        return {"positions": 0, "oracle_losing": 0, "oracle_losing_but_rollout_won": 0}

    positions = 0
    oracle_losing = 0
    losing_but_won = 0
    for value, outcome in zip(values, final_outcomes(trace, len(values))):
        if value is None or outcome is None:
            continue
        positions += 1
        losing = int(value) == 0 if chomp else float(value) < 0
        if losing:
            oracle_losing += 1
            if outcome > 0:
                losing_but_won += 1
    return {
        "positions": positions,
        "oracle_losing": oracle_losing,
        "oracle_losing_but_rollout_won": losing_but_won,
    }


def variation_summary_rows(trace_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in trace_rows:
        grouped[(str(row["game"]), str(row["variant"]), str(row["board"]))].append(row)

    out: List[Dict[str, Any]] = []
    for (game, variant, board), rows in sorted(grouped.items()):
        matches, labeled, match_rate = pooled_match(rows, "all")
        p1_matches, p1_labeled, p1_rate = pooled_match(rows, "p1")
        p2_matches, p2_labeled, p2_rate = pooled_match(rows, "p2")
        losing_won = sum(row["sampled_selfplay"]["oracle_losing_but_rollout_won"] for row in rows)
        oracle_losing = sum(row["sampled_selfplay"]["oracle_losing"] for row in rows)
        perfect_traces = sum(
            1 for row in rows if row["players"]["all"]["first_non_oracle_ply"] is None
        )

        result: Dict[str, Any] = {
            "variation": f"{game} {variant} {board}",
            "game": game,
            "model": variant,
            "board": board,
            "seeds": sorted({int(row["seed"]) for row in rows if row["seed"] is not None}),
            "runs": unique_run_count(rows),
            "games": len(rows),
            "oracle_match_rate": match_rate,
            "oracle_match_matches": matches,
            "oracle_match_labeled": labeled,
            "p1_oracle_match_rate": p1_rate,
            "p1_oracle_match_matches": p1_matches,
            "p1_oracle_match_labeled": p1_labeled,
            "p2_oracle_match_rate": p2_rate,
            "p2_oracle_match_matches": p2_matches,
            "p2_oracle_match_labeled": p2_labeled,
            "perfect_trace_rate": perfect_traces / len(rows) if rows else None,
            "perfect_traces": perfect_traces,
            "sampled_selfplay_positions": sum(
                row["sampled_selfplay"]["positions"] for row in rows
            ),
            "sampled_selfplay_oracle_losing_but_rollout_won": rate_count_block(
                losing_won, oracle_losing
            ),
        }
        add_mean_std_fields(
            result,
            "oracle_match_rate_by_game",
            (row["players"]["all"]["oracle_match"]["rate"] for row in rows),
            include_n=False,
        )
        add_mean_std_fields(
            result,
            "longest_oracle_chain",
            trace_values(rows, "all", "longest_oracle_chain"),
            include_n=False,
        )
        add_mean_std_fields(
            result,
            "oracle_consistent_prefix",
            trace_values(rows, "all", "oracle_consistent_prefix"),
            include_n=False,
        )
        add_mean_std_fields(
            result,
            "first_non_oracle_ply",
            trace_values(rows, "all", "first_non_oracle_ply"),
        )
        add_mean_std_fields(
            result,
            "game_length",
            (row["move_count"] for row in rows),
            include_n=False,
        )
        out.append(result)

    return out


def find_row(
    rows: List[Dict[str, Any]],
    *,
    game: str,
    board: str,
    model: str,
    player: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    for row in rows:
        if row.get("game") != game or row.get("board") != board or row.get("model") != model:
            continue
        if player is not None and row.get("player") != player:
            continue
        return row
    return None


def reviewer_claim_checks(
    main_rows: List[Dict[str, Any]],
    player_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    cf_azal_p1 = find_row(
        player_rows, game="Connect Four", board="6x7", model="AZAL", player="P1"
    )
    cf_vanilla_p1 = find_row(
        player_rows, game="Connect Four", board="6x7", model="Vanilla", player="P1"
    )
    has_connect4_multiframe = any(
        row["game"] == "Connect Four" and row["model"] == "Multi-Frame"
        for row in main_rows
    )
    return {
        "connect_four_p1_azal_vs_vanilla": {
            "azal_oracle_match": None
            if cf_azal_p1 is None
            else cf_azal_p1["oracle_match_rate_over_runs"],
            "vanilla_oracle_match": None
            if cf_vanilla_p1 is None
            else cf_vanilla_p1["oracle_match_rate_over_runs"],
            "azal_longest_chain": None
            if cf_azal_p1 is None
            else cf_azal_p1["longest_oracle_consistent_chain_over_runs"],
            "vanilla_longest_chain": None
            if cf_vanilla_p1 is None
            else cf_vanilla_p1["longest_oracle_consistent_chain_over_runs"],
        },
        "multi_frame_scope": {
            "connect_four_multiframe_present": has_connect4_multiframe,
            "paper_scope_note": "Multi-Frame comparisons are Chomp-only unless a Connect Four Multi-Frame row is added.",
        },
    }


def compile_results(results_dir: Path) -> List[Dict[str, Any]]:
    trace_paths = sorted(results_dir.rglob("game_trace_*_ava_trace_*.json"))
    trace_rows = [summarize_trace(path, results_dir) for path in trace_paths]
    return variation_summary_rows(trace_rows)


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, data: Any, indent: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(data), f, indent=indent, sort_keys=True, allow_nan=False)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize saved experiment results into paper-ready JSON tables."
    )
    parser.add_argument("--results-dir", type=Path, default=_REPO_ROOT / "results")
    parser.add_argument(
        "--output",
        type=Path,
        default=_REPO_ROOT / "Results" / "compiled_results.json",
    )
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    data = compile_results(results_dir)
    write_json(args.output.resolve(), data, args.indent)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "rows": len(data),
                "variations": [row["variation"] for row in data],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
