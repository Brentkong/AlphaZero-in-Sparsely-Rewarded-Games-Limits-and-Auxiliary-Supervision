from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

VARIANT_MODULES = [
    "config",
    "chomp",
    "connect_four",
    "resnet",
    "mcts",
    "grundy_oracle",
    "connect4_oracle",
]


EXPERIMENTS = {
    "chomp-vanilla-9x10": {
        "src": "Chomp-Vanilla/src",
        "game": "chomp",
        "variant": "Vanilla",
        "rows": 9,
        "cols": 10,
    },
    "chomp-multiframe-9x10": {
        "src": "Chomp-Multiframe/src",
        "game": "chomp",
        "variant": "Multi-Frame",
        "rows": 9,
        "cols": 10,
    },
    "chomp-azal-9x10": {
        "src": "Chomp-Auxiliary/src",
        "game": "chomp",
        "variant": "AZAL",
        "rows": 9,
        "cols": 10,
    },
    "chomp-vanilla-10x11": {
        "src": "Chomp-Vanilla/src",
        "game": "chomp",
        "variant": "Vanilla",
        "rows": 10,
        "cols": 11,
    },
    "chomp-multiframe-10x11": {
        "src": "Chomp-Multiframe/src",
        "game": "chomp",
        "variant": "Multi-Frame",
        "rows": 10,
        "cols": 11,
    },
    "chomp-azal-10x11": {
        "src": "Chomp-Auxiliary/src",
        "game": "chomp",
        "variant": "AZAL",
        "rows": 10,
        "cols": 11,
    },
    "connect4-vanilla": {
        "src": "ConnectFour-Vanilla/src",
        "game": "connect4",
        "variant": "Vanilla",
    },
    "connect4-azal": {
        "src": "ConnectFour-Auxiliary/src",
        "game": "connect4",
        "variant": "AZAL",
    },
}


@dataclass
class LoadedExperiment:
    src_dir: Path
    config: Dict[str, Any]
    game: Any
    model: Any
    mcts: Any
    device: torch.device


def prepare_src_imports(src_dir: Path) -> None:
    src_dir = src_dir.resolve()
    for name in VARIANT_MODULES:
        sys.modules.pop(name, None)
    if str(src_dir) in sys.path:
        sys.path.remove(str(src_dir))
    sys.path.insert(0, str(src_dir))


def load_experiment(
    src_dir: Path,
    checkpoint: Path,
    *,
    game_name: str,
    seed: int,
    rows: Optional[int] = None,
    cols: Optional[int] = None,
    num_searches: Optional[int] = None,
) -> LoadedExperiment:
    from experiment_utils.runtime import seed_everything

    prepare_src_imports(src_dir)
    config_mod = importlib.import_module("config")
    config = dict(config_mod.args)
    config["seed"] = int(seed)
    if rows is not None:
        config["rows"] = int(rows)
    if cols is not None:
        config["cols"] = int(cols)
    if num_searches is not None:
        config["num_searches"] = int(num_searches)

    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if game_name == "chomp":
        game_mod = importlib.import_module("chomp")
        game = game_mod.Chomp(config["rows"], config["cols"])
    elif game_name == "connect4":
        game_mod = importlib.import_module("connect_four")
        game = game_mod.ConnectFour()
    else:
        raise ValueError(f"Unsupported game: {game_name}")

    resnet_mod = importlib.import_module("resnet")
    mcts_mod = importlib.import_module("mcts")
    model = resnet_mod.ResNet(game, config["num_resBlocks"], config["num_hidden"], device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    mcts = mcts_mod.MCTS(game, config, model)

    return LoadedExperiment(
        src_dir=src_dir,
        config=config,
        game=game,
        model=model,
        mcts=mcts,
        device=device,
    )


def connect4_oracle(game: Any) -> Any:
    oracle_mod = importlib.import_module("connect4_oracle")
    return oracle_mod.PerfectC4Oracle(game)


def chomp_oracle_funcs() -> tuple[Any, Any, Any]:
    oracle_mod = importlib.import_module("grundy_oracle")
    return oracle_mod.grundy, oracle_mod.fits_grundy_limit, oracle_mod.find_best_moves

