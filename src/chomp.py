import numpy as np

class Chomp:
    def __init__(self, rows, columns):
        self.row_count = rows
        self.column_count = columns
        self.action_size = self.row_count * self.column_count
        
    def __repr__(self):
        return f"Chomp({self.row_count}x{self.column_count})"
        
    def get_initial_state(self):
        return np.ones((self.row_count, self.column_count))
    
    def get_next_state(self, state, action):
        row = action // self.column_count
        column = action % self.column_count
        new_state = state.copy()  
        new_state[row:, column:] = 0
        return new_state
    
    def get_valid_moves(self, state):
        return (state.reshape(-1) == 1).astype(np.uint8) 
    
    def check_win(self, state):  
        return state[0, 0] == 0
    
    def get_value_and_terminated(self, state, action):
        next_state = self.get_next_state(state, action)
        if self.check_win(next_state):
            return -1, True
        return 0, False
    
    def get_opponent(self, player):
        return -player
    
    def get_opponent_value(self, value):
        return -value
   
    def get_encoded_state(self, histories):
        if len(histories) > 0 and isinstance(histories[0], np.ndarray):
            hist = histories
            if len(hist) < 3:
                hist = [hist[0]] * (3 - len(hist)) + hist

            frames = [(s == 1).astype(np.float32) for s in hist[-3:]]
            return np.stack(frames, axis=0)

        encoded_batch = []
        for hist in histories:
            if len(hist) < 3:
                hist = [hist[0]] * (3 - len(hist)) + hist

            frames = [(s == 1).astype(np.float32) for s in hist[-3:]]
            encoded_batch.append(np.stack(frames, axis=0))

        return np.stack(encoded_batch, axis=0)
    


    