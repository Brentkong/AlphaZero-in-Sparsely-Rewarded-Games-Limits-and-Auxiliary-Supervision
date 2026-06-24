from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.common import EXPERIMENTS, REPO_ROOT
from experiment_utils.runtime import parse_int_list


def board_label(exp: Dict[str, object]) -> str:
    if exp["game"] == "chomp":
        return f"{exp['rows']}x{exp['cols']}"
    return "6x7"


def result_dir(base: Path, exp: Dict[str, object], seed: int) -> Path:
    game = "Chomp" if exp["game"] == "chomp" else "Connect Four"
    return base / game / str(exp["variant"]) / board_label(exp) / f"seed_{seed}"


def trace_path(
    base: Path,
    exp: Dict[str, object],
    seed: int,
    eval_mode: str,
    trace_index: int,
) -> Path:
    cells = int(exp["rows"]) * int(exp["cols"]) if exp["game"] == "chomp" else 42
    return result_dir(base, exp, seed) / f"game_trace_{cells}_{eval_mode}_trace_{trace_index:03d}.json"


def latest_checkpoint(path: Path) -> Path:
    candidates = sorted(path.glob("model_*.pt"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No model_*.pt checkpoint found in {path}")
    return candidates[-1]


def run(cmd: List[str], cwd: Path, dry_run: bool) -> None:
    shown = " ".join(cmd)
    print(f"[{cwd}] {shown}")
    if not dry_run:
        subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run configured experiment variants across multiple seeds."
    )
    parser.add_argument("--seeds", nargs="+", default=["0", "1", "2"])
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=sorted(EXPERIMENTS),
        default=sorted(EXPERIMENTS),
    )
    parser.add_argument("--outdir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--python", default="python3")
    parser.add_argument("--num-searches", type=int, default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="disabled")
    parser.add_argument("--eval-mode", choices=("ava", "avp", "avo"), default="ava")
    parser.add_argument("--trace-games", type=int, default=20)
    parser.add_argument("--sampled-states", action="store_true")
    parser.add_argument("--moving-target", action="store_true")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--depths", nargs="+", default=["0", "4", "8", "12"])
    parser.add_argument("--diagnostic-episodes", type=int, default=8)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.outdir = args.outdir.resolve()
    if args.trace_games < 0:
        raise ValueError("--trace-games must be non-negative")

    seeds = parse_int_list(args.seeds)
    traces: List[Path] = []
    sampled_results: List[Path] = []

    for exp_name in args.experiments:
        exp = EXPERIMENTS[exp_name]
        src_dir = REPO_ROOT / str(exp["src"])
        for seed in seeds:
            common = ["--seed", str(seed), "--outdir", str(args.outdir)]
            if args.num_searches is not None:
                common += ["--num-searches", str(args.num_searches)]
            if exp["game"] == "chomp":
                common += ["--rows", str(exp["rows"]), "--cols", str(exp["cols"])]

            if not args.skip_training:
                run(
                    [
                        args.python,
                        "main.py",
                        *common,
                        "--wandb-mode",
                        args.wandb_mode,
                    ],
                    src_dir,
                    args.dry_run,
                )

            run_path = result_dir(args.outdir, exp, seed)
            if not args.skip_eval:
                checkpoint = latest_checkpoint(run_path) if not args.dry_run else run_path / "model_LAST.pt"
                for trace_index in range(args.trace_games):
                    run(
                        [
                            args.python,
                            "play.py",
                            *common,
                            "--checkpoint",
                            str(checkpoint),
                            "--eval-mode",
                            args.eval_mode,
                            "--trace-index",
                            str(trace_index),
                        ],
                        src_dir,
                        args.dry_run,
                    )
                    traces.append(trace_path(args.outdir, exp, seed, args.eval_mode, trace_index))

            if args.sampled_states or args.moving_target:
                checkpoint = latest_checkpoint(run_path) if not args.dry_run else run_path / "model_LAST.pt"
                game = str(exp["game"])
                eval_common = [
                    "--game",
                    game,
                    "--src-dir",
                    str(src_dir),
                    "--checkpoint",
                    str(checkpoint),
                    "--seed",
                    str(seed),
                ]
                if args.num_searches is not None:
                    eval_common += ["--num-searches", str(args.num_searches)]
                if game == "chomp":
                    eval_common += ["--rows", str(exp["rows"]), "--cols", str(exp["cols"])]

                if args.sampled_states:
                    run(
                        [
                            args.python,
                            str(REPO_ROOT / "evaluation" / "sampled_state_eval.py"),
                            *eval_common,
                            "--outdir",
                            str(args.outdir / "evaluations" / "sampled"),
                            "--samples",
                            str(args.samples),
                            "--depths",
                            *args.depths,
                        ],
                        src_dir,
                        args.dry_run,
                    )
                    if not args.dry_run:
                        pattern = (
                            args.outdir
                            / "evaluations"
                            / "sampled"
                            / ("Chomp" if game == "chomp" else "Connect Four")
                            / str(exp["variant"])
                            / board_label(exp)
                            / f"seed_{seed}"
                            / f"{game}_sampled_seed_{seed}.json"
                        )
                        sampled_results.append(pattern)

                if args.moving_target:
                    run(
                        [
                            args.python,
                            str(REPO_ROOT / "evaluation" / "moving_target_diagnostic.py"),
                            *eval_common,
                            "--outdir",
                            str(args.outdir / "evaluations" / "diagnostics"),
                            "--episodes",
                            str(args.diagnostic_episodes),
                        ],
                        src_dir,
                        args.dry_run,
                    )

    if traces:
        run(
            [
                args.python,
                str(REPO_ROOT / "evaluation" / "aggregate_traces.py"),
                "--outdir",
                str(args.outdir / "summary"),
                "--inputs",
                *[str(path) for path in traces],
            ],
            REPO_ROOT,
            args.dry_run,
        )

    if sampled_results:
        run(
            [
                args.python,
                str(REPO_ROOT / "evaluation" / "aggregate_traces.py"),
                "--outdir",
                str(args.outdir / "summary" / "sampled"),
                "--inputs",
                *[str(path) for path in sampled_results],
            ],
            REPO_ROOT,
            args.dry_run,
        )


if __name__ == "__main__":
    main()
