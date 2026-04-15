import numpy as np
import torch
import torch.nn.functional as F
import wandb
from tqdm.notebook import trange
import random
from mcts import MCTSParallel
torch.manual_seed(0)

class AlphaZeroParallel:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTSParallel(game, args, model)
        self._pmove_cache = {}
        self._pmove_oracle_available = None  
        
    def selfPlay(self):
        return_memory = []
        player = 1
        spGames = [SPG(self.game) for spg in range(self.args['num_parallel_games'])]
        
        while len(spGames) > 0:
            states = np.stack([spg.state for spg in spGames])
            
            self.mcts.search(states, spGames)
            
            for i in range(len(spGames))[::-1]:
                spg = spGames[i]
                
                action_probs = np.zeros(self.game.action_size)
                for child in spg.root.children:
                    action_probs[child.action_taken] = child.visit_count
                action_probs /= np.sum(action_probs)

                spg.memory.append((spg.root.state, action_probs, player))

                temperature_action_probs = action_probs ** (1 / self.args['temperature'])
                temperature_action_probs /= temperature_action_probs.sum()
                action = np.random.choice(self.game.action_size, p=temperature_action_probs) 

                spg.state = self.game.get_next_state(spg.state, action)

                value, is_terminal = self.game.get_value_and_terminated(spg.state, action)

                if is_terminal:
                    for hist_neutral_state, hist_action_probs, hist_player in spg.memory:
                        hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                        return_memory.append((
                            self.game.get_encoded_state(hist_neutral_state),
                            hist_action_probs,
                            hist_outcome
                        ))
                    del spGames[i]
                    
            player = self.game.get_opponent(player)
            
        return return_memory
                
    def _pmove_targets_np(self, state_batch_np):
            lam = float(self.args.get('p_move_lambda', 0.0))
            oracle_size = int(self.args.get('p_move_oracle_size', 0))
            eps = float(self.args.get('p_move_eps', 1e-3))
            if lam <= 0.0 or oracle_size <= 0:
                return None, None

            if self._pmove_oracle_available is None:
                try:
                    from grundy_oracle import fits_grundy_limit, find_best_moves  
                    self._pmove_oracle_available = True
                except Exception:
                    self._pmove_oracle_available = False

            if not self._pmove_oracle_available:
                return None, None

            from grundy_oracle import fits_grundy_limit, find_best_moves

            N = state_batch_np.shape[0]
            q_targets = np.zeros((N, self.game.action_size), dtype=np.float32)
            mask = np.zeros((N,), dtype=np.float32)

            for i in range(N):
                s = (state_batch_np[i, 0].cpu().numpy() > 0.5).astype(np.int32)

                if hasattr(self.game, "check_win") and self.game.check_win(s):
                    continue

                if not fits_grundy_limit(s, oracle_size):
                    continue

                key = s.tobytes()
                best_moves = self._pmove_cache.get(key)
                if best_moves is None:
                    try:
                        best_moves = find_best_moves(s)
                    except Exception:
                        best_moves = []
                    self._pmove_cache[key] = best_moves

                if not best_moves:
                    continue

                valid = self.game.get_valid_moves(s).astype(np.float32)
                z = np.zeros(self.game.action_size, dtype=np.float32)
                z[best_moves] = 1.0

                q = (z + eps) * valid
                qsum = float(q.sum())
                if qsum <= 0:
                    continue

                q_targets[i] = q / qsum
                mask[i] = 1.0

            return q_targets, mask

    def train(self, memory):
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx : batchIdx + self.args['batch_size']]

            state, policy_targets, value_targets = zip(*sample)
            
            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)
            
            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)
            
            out_policy, out_value = self.model(state)
            
            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)

            p_move_loss = torch.tensor(0.0, device=self.model.device)
            p_labeled_frac = 0.0
            q_targets_np, q_mask_np = self._pmove_targets_np(state)
            if q_targets_np is not None:
                q_targets = torch.tensor(q_targets_np, dtype=torch.float32, device=self.model.device)
                q_mask = torch.tensor(q_mask_np, dtype=torch.float32, device=self.model.device)

                log_probs = F.log_softmax(out_policy, dim=1)
                per_sample_ce = -(q_targets * log_probs).sum(dim=1)  

                denom = q_mask.sum() + 1e-8
                p_move_loss = (per_sample_ce * q_mask).sum() / denom
                p_labeled_frac = float(q_mask.mean().item())

            lam = float(self.args.get('p_move_lambda', 0.0))
            loss = policy_loss + value_loss + lam * p_move_loss
            
            if wandb is not None and batchIdx % self.args['log_freq'] == 0:
                wandb.log({
                    "policy_loss": policy_loss.item(),
                    "value_loss": value_loss.item(),
                    "p_move_loss": float(p_move_loss.item()),
                    "p_move_labeled_frac": float(p_labeled_frac),
                    "total_loss": loss.item()
                })

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def learn(self):
        for iteration in range(self.args['num_iterations']):
            memory = []
            
            self.model.eval()
            for selfPlay_iteration in trange(self.args['num_selfPlay_iterations'] // self.args['num_parallel_games']):
                memory += self.selfPlay()
            
            self.model.train()
            for epoch in trange(self.args['num_epochs']):
                self.train(memory)
            
            torch.save(self.model.state_dict(), f"model_{iteration}_{self.game}.pt")
            torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}_{self.game}.pt")
            
class SPG:
    def __init__(self, game):
        self.state = game.get_initial_state()
        self.memory = []
        self.root = None
        self.node = None