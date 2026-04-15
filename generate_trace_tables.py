from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


SAMPLE_CONFIG: Dict[str, Any] = {
    "games": {
        "chomp": {
            "title": "Chomp 10x11 Self-Play",
            "traces": {
                "Vanilla": "/mnt/data/chomp_vanilla_10x11_selfplay.json",
                "Multi-Frame": "/mnt/data/chomp_multiframe_10x11_selfplay.json",
                "AZAL": "/mnt/data/chomp_azal_10x11_selfplay.json",
            },
        },
        "connect4": {
            "title": "Connect Four Self-Play",
            "traces": {
                "Vanilla": "/mnt/data/connect4_vanilla_selfplay.json",
                "AZAL": "/mnt/data/connect4_azal_selfplay.json",
            },
        },
    }
}


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_game(trace: Dict[str, Any]) -> str:
    if "grundy_numbers" in trace:
        return "chomp"
    if "score_state" in trace:
        return "connect4"
    raise ValueError("Trace must contain either 'grundy_numbers' or 'score_state'.")


def oracle_match_flags(trace: Dict[str, Any]) -> List[Optional[bool]]:
    moves = trace["move_sequence"]
    best_moves = trace["best_moves"]
    if len(moves) != len(best_moves):
        raise ValueError("'move_sequence' and 'best_moves' must have the same length.")

    out: List[Optional[bool]] = []
    for move, best in zip(moves, best_moves):
        if best is None or len(best) == 0:
            out.append(None)
        else:
            out.append(move in best)
    return out


def safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else math.nan


def longest_true_run(flags: List[Optional[bool]]) -> int:
    best = 0
    current = 0
    for flag in flags:
        if flag is True:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best

def first_false_ply(flags: List[Optional[bool]], player_index: int) -> Optional[int]:
    for idx, flag in enumerate(flags):
        if flag is False:
            return player_index + 2 * idx
    return None

def player_flags(flags: List[Optional[bool]], player_index: int) -> List[Optional[bool]]:
    # player_index = 0 for P1 (even plies), 1 for P2 (odd plies)
    return [flags[t] for t in range(player_index, len(flags), 2)]


def chomp_preserve_rate(trace: Dict[str, Any], player_index: int) -> float:
    g = trace["grundy_numbers"]

    winning_turns = 0
    preserved = 0

    # Player 1: t = 0,2,4,... and check g[t] -> g[t+1]
    # Player 2: t = 1,3,5,... and check g[t] -> g[t+1]
    for t in range(player_index, len(g) - 1, 2):
        if g[t] != 0:
            winning_turns += 1
            if g[t + 1] == 0:
                preserved += 1

    return safe_rate(preserved, winning_turns)


def summarize_player(flags: List[Optional[bool]]) -> Dict[str, Any]:
    labeled = [f for f in flags if f is not None]
    match_count = sum(1 for f in labeled if f is True)
    labeled_count = len(labeled)

    return {
        "Oracle-match rate": safe_rate(match_count, labeled_count),
        "Longest oracle-consistent chain": longest_true_run(flags),
    }


def summarize_chomp_both_players(model: str, trace: Dict[str, Any]) -> Dict[str, Any]:
    all_flags = oracle_match_flags(trace)

    p1_flags = player_flags(all_flags, 0)
    p2_flags = player_flags(all_flags, 1)

    p1 = summarize_player(p1_flags)
    p2 = summarize_player(p2_flags)

    return {
        "Model": model,
        "Oracle-match rate (P1)": p1["Oracle-match rate"],
        "Longest oracle-consistent chain (P1)": p1["Longest oracle-consistent chain"],
        "Preserve-to-$g=0$ rate (P1)": chomp_preserve_rate(trace, 0),
        "Oracle-match rate (P2)": p2["Oracle-match rate"],
        "Longest oracle-consistent chain (P2)": p2["Longest oracle-consistent chain"],
        "Preserve-to-$g=0$ rate (P2)": chomp_preserve_rate(trace, 1),
    }


def summarize_connect4_both_players(model: str, trace: Dict[str, Any]) -> Dict[str, Any]:
    all_flags = oracle_match_flags(trace)

    p1_flags = player_flags(all_flags, 0)
    p2_flags = player_flags(all_flags, 1)

    p1 = summarize_player(p1_flags)
    p2 = summarize_player(p2_flags)

    return {
        "Model": model,
        "Oracle-match rate (P1)": p1["Oracle-match rate"],
        "Longest oracle-consistent chain (P1)": p1["Longest oracle-consistent chain"],
        "First non-oracle ply (P1)": first_false_ply(p1_flags, 0),
        "Oracle-match rate (P2)": p2["Oracle-match rate"],
        "Longest oracle-consistent chain (P2)": p2["Longest oracle-consistent chain"],
        "First non-oracle ply (P2)": first_false_ply(p2_flags, 1),
    }


def format_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    return out


def write_latex_table(
    df: pd.DataFrame,
    path: Path,
    caption: str,
    label: str,
) -> None:
    formatted = format_dataframe(df)

    tabular = formatted.to_latex(
        index=False,
        escape=False,
        na_rep="",
    ).strip()

    latex = "\n".join([
        r"\begin{table*}[t]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        tabular,
        r"\end{table*}",
        "",
    ])

    with path.open("w", encoding="utf-8") as f:
        f.write(latex)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate compact trace metrics tables as LaTeX files."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a JSON config file describing the trace files for each game.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("trace_table_outputs"),
        help="Directory where the TEX tables will be written.",
    )
    parser.add_argument(
        "--write-sample-config",
        type=Path,
        default=None,
        help="Write a sample config JSON to this path and exit.",
    )
    args = parser.parse_args()

    if args.write_sample_config is not None:
        args.write_sample_config.parent.mkdir(parents=True, exist_ok=True)
        with args.write_sample_config.open("w", encoding="utf-8") as f:
            json.dump(SAMPLE_CONFIG, f, indent=2)
        print(f"Wrote sample config to {args.write_sample_config}")
        return

    if args.config is None:
        raise SystemExit("Please provide --config, or use --write-sample-config first.")

    config = load_json(args.config)
    games = config.get("games", {})
    if not games:
        raise SystemExit("Config file must contain a top-level 'games' object.")

    args.outdir.mkdir(parents=True, exist_ok=True)

    if "chomp" in games:
        rows: List[Dict[str, Any]] = []
        for model, trace_path in games["chomp"].get("traces", {}).items():
            trace = load_json(Path(trace_path))
            if infer_game(trace) != "chomp":
                raise ValueError(f"Trace '{trace_path}' is not a Chomp trace.")
            rows.append(summarize_chomp_both_players(model, trace))
        if rows:
            df = pd.DataFrame(rows)
            write_latex_table(
                df,
                args.outdir / "chomp_metrics.tex",
                caption="Chomp trace-level metrics across model variants, reported separately for first-player and second-player turns.",
                label="tab:chomp_trace_metrics",
            )

    if "connect4" in games:
        rows: List[Dict[str, Any]] = []
        for model, trace_path in games["connect4"].get("traces", {}).items():
            trace = load_json(Path(trace_path))
            if infer_game(trace) != "connect4":
                raise ValueError(f"Trace '{trace_path}' is not a Connect Four trace.")
            rows.append(summarize_connect4_both_players(model, trace))
        if rows:
            df = pd.DataFrame(rows)
            write_latex_table(
                df,
                args.outdir / "connect4_metrics.tex",
                caption="Connect Four trace-level metrics across model variants, reported separately for first-player and second-player turns.",
                label="tab:connect4_trace_metrics",
            )

    print(f"Wrote LaTeX table files to {args.outdir}")


if __name__ == "__main__":
    main()