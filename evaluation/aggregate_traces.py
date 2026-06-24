from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiment_utils.runtime import mean_std


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def oracle_match_flags(trace: Dict[str, Any]) -> List[Optional[bool]]:
    flags: List[Optional[bool]] = []
    for move, best in zip(trace["move_sequence"], trace["best_moves"]):
        if best is None or len(best) == 0:
            flags.append(None)
        else:
            flags.append(move in best)
    return flags


def longest_true_run(flags: Iterable[Optional[bool]]) -> int:
    best = 0
    current = 0
    for flag in flags:
        if flag is True:
            current += 1
            best = max(best, current)
        elif flag is False:
            current = 0
    return best


def first_false_ply(flags: List[Optional[bool]]) -> Optional[int]:
    for idx, flag in enumerate(flags):
        if flag is False:
            return idx
    return None


def rate(flags: Iterable[Optional[bool]]) -> float:
    labeled = [flag for flag in flags if flag is not None]
    if not labeled:
        return math.nan
    return sum(1 for flag in labeled if flag is True) / len(labeled)


def legacy_metadata(path: Path, trace: Dict[str, Any]) -> Dict[str, str]:
    parts = list(path.parts)
    if "Chomp" in parts:
        idx = parts.index("Chomp")
        return {
            "game": "Chomp",
            "board": parts[idx + 1] if idx + 1 < len(parts) else "",
            "variant": parts[idx + 2] if idx + 2 < len(parts) else path.parent.name,
        }
    if "Connect Four" in parts:
        idx = parts.index("Connect Four")
        return {
            "game": "Connect Four",
            "board": "6x7",
            "variant": parts[idx + 1] if idx + 1 < len(parts) else path.parent.name,
        }
    return {
        "game": "Chomp" if "grundy_numbers" in trace else "Connect Four",
        "board": "",
        "variant": path.parent.name,
    }


def summarize_trace(path: Path) -> Dict[str, Any]:
    trace = load_json(path)
    fallback = legacy_metadata(path, trace)
    meta = {**fallback, **trace.get("metadata", {})}
    flags = oracle_match_flags(trace)
    failure = first_false_ply(flags)
    return {
        "source": str(path),
        "kind": "trace",
        "game": meta.get("game"),
        "variant": meta.get("variant"),
        "board": meta.get("board"),
        "seed": meta.get("seed", ""),
        "oracle_match_rate": rate(flags),
        "longest_oracle_chain": longest_true_run(flags),
        "first_non_oracle_ply": failure if failure is not None else math.nan,
        "sampled_state_match_rate": math.nan,
    }


def summarize_sample(path: Path) -> Dict[str, Any]:
    data = load_json(path)
    meta = data.get("metadata", {})
    summary = data.get("summary", {})
    return {
        "source": str(path),
        "kind": "sampled",
        "game": meta.get("game", ""),
        "variant": meta.get("variant", ""),
        "board": meta.get("board", ""),
        "seed": meta.get("seed", ""),
        "oracle_match_rate": math.nan,
        "longest_oracle_chain": summary.get("mean_chain_length", math.nan),
        "first_non_oracle_ply": summary.get("mean_failure_depth", math.nan),
        "sampled_state_match_rate": summary.get("sampled_state_match_rate", math.nan),
    }


def group_key(row: Dict[str, Any]) -> tuple[Any, ...]:
    return row["game"], row["variant"], row["board"], row["kind"]


def fmt(value: float) -> str:
    if value is None or math.isnan(float(value)):
        return "N/A"
    return f"{float(value):.3f}"


def fmt_mean_std(values: List[float]) -> str:
    mean, std, n = mean_std(values)
    if n == 0:
        return "N/A"
    return f"{mean:.3f} +/- {std:.3f} (n={n})"


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "game",
        "variant",
        "board",
        "kind",
        "oracle_match_rate",
        "longest_oracle_chain",
        "first_non_oracle_ply",
        "sampled_state_match_rate",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_tex(path: Path, rows: List[Dict[str, Any]]) -> None:
    headers = [
        "Game",
        "Variant",
        "Board",
        "Kind",
        "Oracle match",
        "Longest chain",
        "First failure",
        "Sampled match",
    ]
    lines = [
        r"\begin{tabular}{llllllll}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    str(row["game"]),
                    str(row["variant"]),
                    str(row["board"]),
                    str(row["kind"]),
                    str(row["oracle_match_rate"]),
                    str(row["longest_oracle_chain"]),
                    str(row["first_non_oracle_ply"]),
                    str(row["sampled_state_match_rate"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate per-seed trace and sampled-state metrics."
    )
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--outdir", type=Path, default=Path("results/summary"))
    args = parser.parse_args()

    per_seed: List[Dict[str, Any]] = []
    for path in args.inputs:
        data = load_json(path)
        if "move_sequence" in data and "best_moves" in data:
            per_seed.append(summarize_trace(path))
        elif "summary" in data:
            per_seed.append(summarize_sample(path))
        else:
            raise ValueError(f"Unsupported metrics file: {path}")

    grouped: Dict[tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in per_seed:
        grouped.setdefault(group_key(row), []).append(row)

    aggregate_rows: List[Dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        game, variant, board, kind = key
        aggregate_rows.append(
            {
                "game": game,
                "variant": variant,
                "board": board,
                "kind": kind,
                "oracle_match_rate": fmt_mean_std([r["oracle_match_rate"] for r in rows]),
                "longest_oracle_chain": fmt_mean_std([r["longest_oracle_chain"] for r in rows]),
                "first_non_oracle_ply": fmt_mean_std([r["first_non_oracle_ply"] for r in rows]),
                "sampled_state_match_rate": fmt_mean_std(
                    [r["sampled_state_match_rate"] for r in rows]
                ),
            }
        )

    args.outdir.mkdir(parents=True, exist_ok=True)
    write_csv(args.outdir / "per_seed_metrics.csv", per_seed)
    write_csv(args.outdir / "aggregate_metrics.csv", aggregate_rows)
    write_tex(args.outdir / "aggregate_metrics.tex", aggregate_rows)
    print(f"Wrote aggregate metrics to {args.outdir}")


if __name__ == "__main__":
    main()
