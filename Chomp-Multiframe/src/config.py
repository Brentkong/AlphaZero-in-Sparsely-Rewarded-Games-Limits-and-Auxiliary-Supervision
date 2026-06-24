args = {
        'seed': 0,
        'game_name': 'Chomp',
        'variant': 'Multi-Frame',
        'checkpoint_selection_rule': 'last iteration checkpoint',
        'rows': 10,
        'cols': 11,
        'num_resBlocks': 9, 
        'num_hidden': 128,

        'lr': 0.001,
        'weight_decay': 0.0001,

        'C': 2,
        'max_rollout_steps': 200,
        'num_frames': 3,
        'num_searches': 800,
        'num_iterations': 10,
        'num_selfPlay_iterations': 700,
        'num_parallel_games': 100,
        'num_epochs': 4,
        'batch_size': 128,
        'temperature': 1.25,
        'dirichlet_epsilon': 0.25,
        'dirichlet_alpha': 0.3,

        'log_freq': 100
    }
