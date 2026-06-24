args = {
        'seed': 0,
        'game_name': 'Chomp',
        'variant': 'AZAL',
        'checkpoint_selection_rule': 'last iteration checkpoint',
        'rows': 9,
        'cols': 10,
        'num_resBlocks': 9, 
        'num_hidden': 128,

        'lr': 0.001,
        'weight_decay': 0.0001,

        'C': 2,
        'num_searches': 800,
        'num_iterations': 10,
        'num_selfPlay_iterations': 700,
        'num_parallel_games': 100,
        'num_epochs': 4,
        'batch_size': 128,
        'temperature': 1.25,
        'dirichlet_epsilon': 0.25,
        'dirichlet_alpha': 0.3,

        'p_move_lambda': 1,
        'p_move_oracle_size': 110,
        'p_move_eps': 1e-3,

        'log_freq': 100
    }
