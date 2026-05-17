from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt


SAMPLE_CONFIG: Dict[str, Any] = {
    "games": {
        "chomp": {
            "title": "Chomp 10x11 Self-Play",
            "ylabel": "Grundy number g(s_t)",
            "traces": {
                "Vanilla": "/mnt/data/chomp_vanilla_10x11_selfplay.json",
                "Multi-Frame": "/mnt/data/chomp_multiframe_10x11_selfplay.json",
                "AZAL": "/mnt/data/chomp_azal_10x11_selfplay.json",
            },
        },
        "connect4": {
            "title": "Connect Four Self-Play",
            "ylabel": "Oracle score of current state",
            "traces": {
                "Vanilla": "/mnt/data/connect4_vanilla_selfplay.json",
                "AZAL": "/mnt/data/connect4_azal_selfplay.json",
            },
        },
    }
}
MODEL_COLORS = {
    "Vanilla": "tab:blue",
    "Multi-Frame": "tab:orange",
    "AZAL": "tab:green",
}


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_trace_path(trace_path: str, config_dir: Path) -> Path:
    path = Path(trace_path)
    if path.is_absolute():
        return path
    return config_dir / path


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

    flags: List[Optional[bool]] = []
    for move, best in zip(moves, best_moves):
        if best is None or len(best) == 0:
            flags.append(None)
        else:
            flags.append(move in best)
    return flags


def oracle_series(trace: Dict[str, Any]) -> List[float]:
    game = infer_game(trace)
    if game == "chomp":
        return list(trace["grundy_numbers"])
    return list(trace["score_state"])


def sanitized_name(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")


def plot_game(game_name: str, spec: Dict[str, Any], outdir: Path, config_dir: Path) -> Path:
    traces = spec.get("traces", {})
    if not traces:
        raise ValueError(f"No traces provided for game '{game_name}'.")

    model_items = list(traces.items())
    n_models = len(model_items)

    fig_width_panels = max(3, n_models)
    fig = plt.figure(figsize=(5.0 * fig_width_panels, 6.5))
    gs = fig.add_gridspec(2, n_models, height_ratios=[3.0, 1.2], hspace=0.12, wspace=0.18)

    top_axes = []
    for i in range(n_models):
        if i == 0:
            ax = fig.add_subplot(gs[0, i])
        else:
            ax = fig.add_subplot(gs[0, i], sharey=top_axes[0])
        top_axes.append(ax)

    ax_bottom = fig.add_subplot(gs[1, :])

    title = spec.get("title", game_name.title())
    ylabel = spec.get("ylabel", "Oracle quantity")
    model_names = [name for name, _ in model_items]

    for row_idx, ((model_name, trace_path), ax_top) in enumerate(zip(model_items, top_axes)):
        trace_file = resolve_trace_path(trace_path, config_dir)
        trace = load_json(trace_file)
        inferred_game = infer_game(trace)
        if inferred_game != game_name:
            raise ValueError(
                f"Trace '{trace_file}' looks like '{inferred_game}', not '{game_name}'."
            )

        y = oracle_series(trace)
        x = list(range(len(y)))
        flags = oracle_match_flags(trace)

        color = MODEL_COLORS.get(model_name, None)
        line = ax_top.plot(
            x,
            y,
            marker="o",
            linewidth=2.0,
            label=model_name,
            color=color,
        )[0]
        color = line.get_color()

        optimal_x = [i for i, f in enumerate(flags) if f is True]
        optimal_y = [y[i] for i in optimal_x]
        nonoptimal_x = [i for i, f in enumerate(flags) if f is False]
        nonoptimal_y = [y[i] for i in nonoptimal_x]
        unlabeled_x = [i for i, f in enumerate(flags) if f is None]
        unlabeled_y = [y[i] for i in unlabeled_x]

        if optimal_x:
            ax_top.scatter(
                optimal_x, optimal_y,
                s=42, facecolors=color, edgecolors=color, zorder=3
            )
        if nonoptimal_x:
            ax_top.scatter(
                nonoptimal_x, nonoptimal_y,
                s=62, facecolors="none", edgecolors=color, linewidths=1.8, zorder=4
            )
        if unlabeled_x:
            ax_top.scatter(
                unlabeled_x, unlabeled_y,
                s=24, facecolors="none", edgecolors="0.6", linewidths=1.0, zorder=2
            )

        ax_top.axhline(0, linewidth=1.0)
        ax_top.set_title(model_name)
        ax_top.set_xlabel("Ply")
        ax_top.grid(True, alpha=0.3)

        if row_idx == 0:
            ax_top.set_ylabel(ylabel)
        else:
            ax_top.tick_params(axis="y", labelleft=False)

        labeled_x = [i for i, f in enumerate(flags) if f is not None]
        optimal_strip_x = [i for i, f in enumerate(flags) if f is True]
        nonoptimal_strip_x = [i for i, f in enumerate(flags) if f is False]

        if labeled_x:
            ax_bottom.hlines(
                row_idx,
                xmin=min(labeled_x),
                xmax=max(labeled_x),
                linewidth=0.8,
                alpha=0.25,
                color=color,
            )
        if optimal_strip_x:
            ax_bottom.scatter(
                optimal_strip_x,
                [row_idx] * len(optimal_strip_x),
                s=36,
                facecolors=color,
                edgecolors=color,
                zorder=3
            )
        if nonoptimal_strip_x:
            ax_bottom.scatter(
                nonoptimal_strip_x,
                [row_idx] * len(nonoptimal_strip_x),
                s=52,
                facecolors="none",
                edgecolors=color,
                linewidths=1.8,
                zorder=4
            )

    ax_bottom.set_yticks(list(range(n_models)))
    ax_bottom.set_yticklabels(model_names)
    ax_bottom.set_xlabel("Ply")
    ax_bottom.set_ylabel("Optimal?")
    ax_bottom.grid(True, axis="x", alpha=0.2)
    ax_bottom.set_ylim(-0.75, n_models - 0.25)
    ax_bottom.invert_yaxis()


    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    outpath = outdir / f"{sanitized_name(game_name)}_trace_panel.png"
    fig.savefig(outpath, dpi=250)
    plt.close(fig)
    return outpath


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the two paper-style trace graphs: one for Chomp and one for Connect Four. "
            "Each figure has a top panel for oracle quantity per ply and a bottom strip for oracle-optimal moves."
        )
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
        default=Path("trace_graph_outputs"),
        help="Directory where PNG figures will be written.",
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

    config_path = args.config.resolve()
    config_dir = config_path.parent
    config = load_json(config_path)
    games = config.get("games", {})
    if not games:
        raise SystemExit("Config file must contain a top-level 'games' object.")

    args.outdir.mkdir(parents=True, exist_ok=True)

    generated_paths: List[Path] = []
    for game_name in ("chomp", "connect4"):
        if game_name not in games:
            continue
        outpath = plot_game(game_name, games[game_name], args.outdir, config_dir)
        generated_paths.append(outpath)

    if not generated_paths:
        raise SystemExit("No supported games found in config. Expected 'chomp' and/or 'connect4'.")

    print("Generated figures:")
    for path in generated_paths:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
