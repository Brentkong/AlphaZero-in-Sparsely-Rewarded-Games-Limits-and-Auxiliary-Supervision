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


def metric_block(trace: Dict[str, Any], flags: List[Optional[bool]], player: Optional[int]) -> Dict[str, Any]:
    selected = player_flags(flags, player)
    counts = count_flags(selected)
    return {
        "oracle_match": counts,
        "longest_oracle_chain": longest_true_run(selected),
        "oracle_consistent_prefix": true_prefix_len(selected),
        "first_non_oracle_ply": first_failure_ply(flags, player),
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


def stat_block(values: Iterable[Any], digits: int = 3, *, include_n: bool = True) -> Dict[str, Any]:
    mean, std, n = mean_std([float(value) for value in values if finite(value)])
    formatted = "N/A" if n == 0 else f"{mean:.{digits}f} +/- {std:.{digits}f}"
    block = {"mean": mean, "std": std, "formatted": formatted}
    if include_n:
        block["n"] = n
    return block


def rate_count_block(matches: int, labeled: int, digits: int = 3) -> Dict[str, Any]:
    rate = matches / labeled if labeled else math.nan
    formatted = "N/A" if labeled == 0 else f"{rate:.{digits}f} ({matches}/{labeled})"
    return {
        "rate": rate,
        "matches": matches,
        "labeled": labeled,
        "formatted": formatted,
    }


def rate_block(count: int, total: int, digits: int = 3) -> Dict[str, Any]:
    rate = count / total if total else math.nan
    formatted = "N/A" if total == 0 else f"{rate:.{digits}f} ({count}/{total})"
    return {"rate": rate, "count": count, "total": total, "formatted": formatted}


def clean_rate_count(block: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "rate": block.get("rate"),
        "count": block.get("matches", 0),
        "total": block.get("labeled", 0),
        "formatted": block.get("formatted", "N/A"),
    }


def row_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
    return row["game"], row["variant"], row["board"]


def sorted_seeds(rows: List[Dict[str, Any]]) -> List[int]:
    seeds = [row.get("seed") for row in rows]
    return sorted(int(seed) for seed in seeds if seed is not None and seed != "")


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


def grouped_by_experiment(rows: List[Dict[str, Any]]) -> Dict[Tuple[Any, ...], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row_key(row)].append(row)
    return grouped


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
                "seeds": sorted_seeds(group),
                "sample_count": sum(int(row.get("sample_count", 0)) for row in group),
                "oracle_match_rate_over_runs": stat_block(
                    (row.get("sampled_state_match_rate") for row in group),
                    include_n=False,
                ),
                "oracle_match_pooled_denominator": rate_count_block(matches, labeled),
            }
        )
    return out


def add_random_start_sampled_metrics(rows: List[Dict[str, Any]]) -> None:
    sampled_dir = _REPO_ROOT / "Results-experiment" / "evaluations" / "sampled"
    sampled_by_key = {
        (row["game"], row["model"], row["board"]): row
        for row in sampled_state_table(sampled_rows(sampled_dir))
    }
    empty = {
        "seeds": [],
        "sample_count": 0,
        "oracle_match_rate_over_runs": stat_block([], include_n=False),
        "oracle_match_pooled_denominator": rate_count_block(0, 0),
    }
    for row in rows:
        sampled = sampled_by_key.get((row["game"], row["model"], row["board"]), empty)
        row["random_start_sampled_state"] = {
            "seeds": sampled["seeds"],
            "positions": sampled["sample_count"],
            "oracle_move_match": clean_rate_count(sampled["oracle_match_pooled_denominator"]),
            "oracle_match_over_runs": sampled["oracle_match_rate_over_runs"],
        }


def trace_values(rows: List[Dict[str, Any]], player: str, field: str) -> List[Any]:
    return [row["players"][player][field] for row in rows]


def pooled_match(rows: List[Dict[str, Any]], player: str) -> Tuple[int, int]:
    matches = sum(int(row["players"][player]["oracle_match"]["matches"]) for row in rows)
    labeled = sum(int(row["players"][player]["oracle_match"]["labeled"]) for row in rows)
    return matches, labeled


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
        matches, labeled = pooled_match(rows, "all")
        p1_matches, p1_labeled = pooled_match(rows, "p1")
        p2_matches, p2_labeled = pooled_match(rows, "p2")
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
            "full_game_self_play": {
                "games": len(rows),
                "game_length": stat_block(
                    (row["move_count"] for row in rows),
                    include_n=False,
                ),
                "perfect_traces": rate_block(perfect_traces, len(rows)),
            },
            "oracle_move_match": {
                "all": rate_block(matches, labeled),
                "p1": rate_block(p1_matches, p1_labeled),
                "p2": rate_block(p2_matches, p2_labeled),
                "per_game": stat_block(
                    (row["players"]["all"]["oracle_match"]["rate"] for row in rows),
                    include_n=False,
                ),
            },
            "oracle_chain": {
                "longest": stat_block(
                    trace_values(rows, "all", "longest_oracle_chain"),
                    include_n=False,
                ),
                "consistent_prefix": stat_block(
                    trace_values(rows, "all", "oracle_consistent_prefix"),
                    include_n=False,
                ),
                "first_non_oracle_ply": stat_block(
                    trace_values(rows, "all", "first_non_oracle_ply"),
                    include_n=False,
                ),
            },
            "oracle_value_vs_rollout": {
                "positions": sum(row["sampled_selfplay"]["positions"] for row in rows),
                "oracle_losing_but_rollout_won": rate_block(losing_won, oracle_losing),
            },
        }
        out.append(result)

    return out


def compile_results(results_dir: Path) -> List[Dict[str, Any]]:
    trace_paths = sorted(results_dir.rglob("game_trace_*_ava_trace_*.json"))
    trace_rows = [summarize_trace(path, results_dir) for path in trace_paths]
    rows = variation_summary_rows(trace_rows)
    add_random_start_sampled_metrics(rows)
    return rows


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
        json.dump(json_safe(data), f, indent=indent, allow_nan=False)
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
