import numpy as np
import math
import torch
import random
torch.manual_seed(0)

class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0, visit_count=0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior
        self.history = []
        
        self.children = []
        
        self.visit_count = visit_count
        self.value_sum = 0
        
    def is_fully_expanded(self):
        return len(self.children) > 0
    
    def select(self):
        best_child = None
        best_ucb = -np.inf
        
        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb
                
        return best_child
    
    def get_ucb(self, child):
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.args['C'] * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior
    
    def expand(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.copy()
                child_state = self.game.get_next_state(child_state, action)

                child = Node(self.game, self.args, child_state, self, action, prob)
                self.children.append(child)
                
        return child
            
    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1
        
        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)          


class MCTS:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model

    def _shape_preserving_rollout(self, start_state, prev_state):
        prev = np.array(prev_state, copy = True)
        start = np.array(start_state, copy = True)

        target_mask = (prev == 1) & (start == 0)

        if not target_mask.any():
            return None
        
        cur_state = start.copy()
        last_action = None

        max_steps = int(self.args.get('max_rollout_steps', 200))
        steps = 0

        while True:
            steps += 1
            if steps > max_steps:
                return None
            
            valid = self.game.get_valid_moves(cur_state)
            possible_actions = np.nonzero(valid)[0]
            if len(possible_actions) == 0:
                try: 
                    val, term = self.game.get_value_and_terminated(cur_state, last_action)
                    if term:
                        return float(val)
                except Exception:
                    return None
                return None
            
            q_action = int(random.choice(possible_actions))
            q_state = self.game.get_next_state(cur_state, q_action)
            last_action = q_action

            q_val, q_term = self.game.get_value_and_terminated(q_state, q_action)
            if q_term:
                return float(q_val)
            
            valid_from_q = self.game.get_valid_moves(q_state)
            possible_r_actions = np.nonzero(valid_from_q)[0]

            found_r = False
            for r_action in possible_r_actions:
                r_state = self.game.get_next_state(q_state, int(r_action))
                removed_mask = (q_state == 1) & (r_state == 0)

                if removed_mask.shape == target_mask.shape and np.array_equal(removed_mask, target_mask):
                    found_r = True
                    last_action = r_action
                    cur_state = r_state
                    break
            
            if not found_r:
                return None
            
            r_val, r_term = self.game.get_value_and_terminated(cur_state, last_action)
            if r_term:
                return float(r_val)
        
    @torch.no_grad()
    def search(self, state):
        root = Node(self.game, self.args, state, visit_count=1)
        root.history = [state.copy()]
        
        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_state(root.history[-3:]), device=self.model.device).unsqueeze(0)
        )
        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
            * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size)
        
        valid_moves = self.game.get_valid_moves(state)
        policy *= valid_moves
        policy /= np.sum(policy)
        root.expand(policy)

        for child in root.children:
            child.history = root.history + [child.state.copy()]

        for search in range(self.args['num_searches']):
            node = root
            while node.is_fully_expanded():
                node = node.select()
                
            value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
            value = self.game.get_opponent_value(value)
            
            rollout_value = None

            if not is_terminal:
                policy, value_pred = self.model(
                    torch.tensor(self.game.get_encoded_state(node.history[-3:]), device=self.model.device).unsqueeze(0)
                )
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                valid_moves = self.game.get_valid_moves(node.state)
                policy *= valid_moves
                policy /= np.sum(policy)
                
                value_pred = value_pred.item()
                
                node.expand(policy)

                for child in node.children:
                    child.history = node.history + [child.state.copy()]

                if len(node.history) >= 2:
                    prev_state = node.history[-2]
                    start_state = node.state
                    rollout_value = self._shape_preserving_rollout(start_state, prev_state)

            if rollout_value is not None:
                node.backpropagate(rollout_value)
            else:
                node.backpropagate(value if is_terminal else value_pred)    
            
            
        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs


class MCTSParallel:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model
        
    def _shape_preserving_rollout(self, start_state, prev_state):
        prev = np.array(prev_state, copy = True)
        start = np.array(start_state, copy = True)

        target_mask = (prev == 1) & (start == 0)

        if not target_mask.any():
            return None
        
        cur_state = start.copy()
        last_action = None

        max_steps = int(self.args.get('max_rollout_steps', 200))
        steps = 0

        while True:
            steps += 1
            if steps > max_steps:
                return None
            
            valid = self.game.get_valid_moves(cur_state)
            possible_actions = np.nonzero(valid)[0]
            if len(possible_actions) == 0:
                try: 
                    val, term = self.game.get_value_and_terminated(cur_state, last_action)
                    if term:
                        return float(val)
                except Exception:
                    return None
                return None
            
            q_action = int(random.choice(possible_actions))
            q_state = self.game.get_next_state(cur_state, q_action)
            last_action = q_action

            q_val, q_term = self.game.get_value_and_terminated(q_state, q_action)
            if q_term:
                return float(q_val)
            
            valid_from_q = self.game.get_valid_moves(q_state)
            possible_r_actions = np.nonzero(valid_from_q)[0]

            found_r = False
            for r_action in possible_r_actions:
                r_state = self.game.get_next_state(q_state, int(r_action))
                removed_mask = (q_state == 1) & (r_state == 0)

                if removed_mask.shape == target_mask.shape and np.array_equal(removed_mask, target_mask):
                    found_r = True
                    last_action = r_action
                    cur_state = r_state
                    break
            
            if not found_r:
                return None
            
            r_val, r_term = self.game.get_value_and_terminated(cur_state, last_action)
            if r_term:
                return float(r_val)

    @torch.no_grad()
    def search(self, states, spGames):
        policy, _ = self.model(
            torch.tensor(
                self.game.get_encoded_state([spg.history[-3:] for spg in spGames]),
                device=self.model.device
            )
        )
        policy = torch.softmax(policy, axis=1).cpu().numpy()
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
            * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size, size=policy.shape[0])
        
        for i, spg in enumerate(spGames):
            spg_policy = policy[i]
            valid_moves = self.game.get_valid_moves(states[i])
            spg_policy *= valid_moves
            spg_policy /= np.sum(spg_policy)

            spg.root = Node(self.game, self.args, states[i], visit_count=1)
            spg.root.expand(spg_policy)
        
        for search in range(self.args['num_searches']):
            for spg in spGames:
                spg.node = None
                node = spg.root

                while node.is_fully_expanded():
                    node = node.select()

                value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
                value = self.game.get_opponent_value(value)
                
                if is_terminal:
                    node.backpropagate(value)
                    
                else:
                    spg.node = node
                    
            expandable_spGames = [mappingIdx for mappingIdx in range(len(spGames)) if spGames[mappingIdx].node is not None]
                    
            if len(expandable_spGames) > 0:
                histories = [
                    spGames[mappingIdx].history[-3:]
                    for mappingIdx in expandable_spGames
                ]

                policy, value = self.model(
                    torch.tensor(
                        self.game.get_encoded_state(histories),
                        device=self.model.device
                    )
                )

                policy = torch.softmax(policy, axis=1).cpu().numpy()
                value = value.cpu().numpy()
                
            for i, mappingIdx in enumerate(expandable_spGames):
                node = spGames[mappingIdx].node
                spg_policy, spg_value = policy[i], value[i]
                
                valid_moves = self.game.get_valid_moves(node.state)
                spg_policy *= valid_moves
                spg_policy /= np.sum(spg_policy)

                node.expand(spg_policy)

                rollout_value = None
                spg = spGames[mappingIdx]

                if len(spg.history) >= 2:
                    prev_state = spg.history[-2]
                    start_state = node.state
                    rollout_value = self._shape_preserving_rollout(start_state, prev_state)
                
                if rollout_value is not None:
                    final_value = max(rollout_value, float(spg_value))
                    
                else:
                    final_value = float(spg_value)
                
                node.backpropagate(final_value)