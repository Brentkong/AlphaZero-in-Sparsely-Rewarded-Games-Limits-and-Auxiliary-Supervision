from pathlib import Path

args = {
        'row_count': 6,
        'column_count': 7,
        'num_resBlocks': 9, 
        'num_hidden': 128,

        'lr': 0.001,
        'weight_decay': 0.0001,

        'C': 2,
        'num_searches': 1000,
        'num_iterations': 20,
        'num_selfPlay_iterations': 700,
        'num_parallel_games': 100,
        'num_epochs': 4,
        'batch_size': 128,
        'temperature': 1.25,
        'dirichlet_epsilon': 0.25,
        'dirichlet_alpha': 0.3,

        'p_move_lambda': 1,
        'p_move_eps': 1e-3,
        'aux_weight': 1.0,
        'oracle_batch_size': 2048,
        "require_oracle": True,

        'log_freq': 100,
        'solver_path': str(Path(__file__).resolve().parents[1] / "connect4-master"),
    }