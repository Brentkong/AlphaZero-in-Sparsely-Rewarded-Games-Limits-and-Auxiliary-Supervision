from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import torch


WANDB_MODES = ("online", "offline", "disabled")


def seed_everything(seed: int) -> None:
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def add_training_cli(parser: argparse.ArgumentParser, config: Dict[str, Any]) -> None:
    parser.add_argument("--seed", type=int, default=config.get("seed", 0))
    if "rows" in config and "cols" in config:
        parser.add_argument("--rows", type=int, default=config["rows"])
        parser.add_argument("--cols", type=int, default=config["cols"])
    parser.add_argument("--num-searches", type=int, default=config.get("num_searches"))
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--wandb-mode", choices=WANDB_MODES, default=None)


def add_play_cli(parser: argparse.ArgumentParser, config: Dict[str, Any]) -> None:
    parser.add_argument("--seed", type=int, default=config.get("seed", 0))
    if "rows" in config and "cols" in config:
        parser.add_argument("--rows", type=int, default=config["rows"])
        parser.add_argument("--cols", type=int, default=config["cols"])
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--num-searches", type=int, default=config.get("num_searches"))
    parser.add_argument("--eval-mode", choices=("ava", "avp", "avo"), default="avo")


def apply_cli_overrides(config: Dict[str, Any], cli: argparse.Namespace) -> None:
    if hasattr(cli, "seed") and cli.seed is not None:
        config["seed"] = int(cli.seed)
    if hasattr(cli, "rows") and cli.rows is not None:
        config["rows"] = int(cli.rows)
    if hasattr(cli, "cols") and cli.cols is not None:
        config["cols"] = int(cli.cols)
    if hasattr(cli, "num_searches") and cli.num_searches is not None:
        config["num_searches"] = int(cli.num_searches)


def board_label(config: Dict[str, Any]) -> str:
    if "board" in config:
        return str(config["board"])
    if "rows" in config and "cols" in config:
        return f"{config['rows']}x{config['cols']}"
    return f"{config['row_count']}x{config['column_count']}"


def run_dir(
    repo_root: Path,
    config: Dict[str, Any],
    outdir: Optional[Path] = None,
) -> Path:
    base = Path(outdir) if outdir is not None else repo_root / "results"
    return (
        base
        / str(config.get("game_name", "game"))
        / str(config.get("variant", "variant"))
        / board_label(config)
        / f"seed_{int(config.get('seed', 0))}"
    )


def evaluation_dir(
    repo_root: Path,
    config: Dict[str, Any],
    kind: str,
    outdir: Optional[Path] = None,
) -> Path:
    base = Path(outdir) if outdir is not None else repo_root / "results" / "evaluations" / kind
    return (
        base
        / str(config.get("game_name", config.get("game", "game")))
        / str(config.get("variant", "variant"))
        / board_label(config)
        / f"seed_{int(config.get('seed', 0))}"
    )


def snapshot_config(config: Dict[str, Any], outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "config_snapshot.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True, default=str)
    return path


def wandb_init_kwargs(mode: Optional[str]) -> Dict[str, str]:
    return {} if mode is None else {"mode": mode}


def latest_checkpoint(path: Path) -> Path:
    candidates = sorted(Path(path).glob("model_*.pt"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(
            f"No model_*.pt checkpoint found in {path}. Pass --checkpoint explicitly."
        )
    return candidates[-1]


def prepare_training_run(
    repo_root: Path,
    config: Dict[str, Any],
) -> tuple[argparse.Namespace, Path]:
    parser = argparse.ArgumentParser()
    add_training_cli(parser, config)
    cli = parser.parse_args()
    apply_cli_overrides(config, cli)
    seed_everything(config["seed"])
    output_dir = run_dir(repo_root, config, cli.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_config(config, output_dir)
    return cli, output_dir


def prepare_play_run(
    repo_root: Path,
    config: Dict[str, Any],
) -> tuple[argparse.Namespace, Path, Path]:
    parser = argparse.ArgumentParser()
    add_play_cli(parser, config)
    parser.add_argument("--show-plot", action="store_true")
    cli = parser.parse_args()
    apply_cli_overrides(config, cli)
    seed_everything(config["seed"])
    output_dir = run_dir(repo_root, config, cli.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_config(config, output_dir)
    checkpoint = cli.checkpoint if cli.checkpoint is not None else latest_checkpoint(output_dir)
    return cli, output_dir, checkpoint


def metadata(config: Dict[str, Any], checkpoint: Optional[Path] = None, **extra: Any) -> Dict[str, Any]:
    data = {
        "game": config.get("game_name"),
        "variant": config.get("variant"),
        "board": board_label(config),
        "seed": int(config.get("seed", 0)),
    }
    if checkpoint is not None:
        data["checkpoint"] = str(checkpoint)
    data.update(extra)
    return data


def parse_int_list(values: Sequence[str]) -> list[int]:
    out: list[int] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                out.append(int(part))
    return out


def mean_std(values: Iterable[float]) -> tuple[float, float, int]:
    vals = [float(v) for v in values if v is not None and not np.isnan(float(v))]
    if not vals:
        return float("nan"), float("nan"), 0
    return float(np.mean(vals)), float(np.std(vals, ddof=0)), len(vals)
