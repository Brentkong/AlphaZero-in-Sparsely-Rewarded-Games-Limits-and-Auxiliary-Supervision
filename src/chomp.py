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
    
    def get_value_and_terminated(self, state, action=None):
        if self.check_win(state):
            return -1, True
        return 0, False
    
    def get_opponent(self, player):
        return -player
    
    def get_opponent_value(self, value):
        return -value
   
    def get_encoded_state(self, state):
        encoded_state = np.stack(
            (state == 1,),  
            axis=0
        ).astype(np.float32)

        if len(state.shape) == 3:
            encoded_state = np.swapaxes(encoded_state, 0, 1)

        return encoded_state

