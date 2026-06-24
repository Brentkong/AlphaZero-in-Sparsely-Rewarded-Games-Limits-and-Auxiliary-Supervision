import numpy as np
import torch
import torch.nn.functional as F
import wandb
from tqdm import trange
import random
from mcts import MCTSParallel


class AlphaZeroParallel:
    def __init__(self, model, optimizer, game, args, oracle=None):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTSParallel(game, args, model)

        self.oracle = oracle

        self.p_move_lambda = float(self.args.get("p_move_lambda", self.args.get("aux_weight", 0.0)))
        self.p_move_eps = float(self.args.get("p_move_eps", 1e-3))

        self.require_oracle = bool(self.args.get("require_oracle", self.p_move_lambda > 0.0))

        self._oracle_used_this_iteration = False
        self._oracle_total_positions = 0
        self._oracle_unique_positions = 0

        if self.require_oracle:
            if self.oracle is None:
                raise ValueError(
                    "Perfect solver is required but `oracle=None`. "
                    "Construct with: AlphaZeroParallel(..., oracle=PerfectC4Oracle(game))"
                )
            if self.p_move_lambda <= 0.0:
                raise ValueError(
                    "Perfect solver is required, but p_move_lambda<=0 so it would never be used. "
                    "Set args['p_move_lambda'] > 0 (or set require_oracle=False)."
                )
            if not hasattr(self.oracle, "analyze_batch"):
                raise TypeError(
                    "oracle must implement analyze_batch(seqs). "
                    "(Expected connect4_oracle.PerfectC4Oracle)"
                )

            try:
                test_scores = self.oracle.analyze_batch([""])[0]
            except Exception as e:
                raise RuntimeError(
                    "Perfect solver healthcheck failed. "
                    "Verify solver_path / c4solver binary / 7x6.book and that the solver runs."
                ) from e

            if len(test_scores) != int(self.game.action_size):
                raise RuntimeError(
                    f"Perfect solver returned {len(test_scores)} scores, but game.action_size={self.game.action_size}."
                )

    def selfPlay(self):
        return_memory = []
        player = 1
        spGames = [SPG(self.game) for _ in range(self.args["num_parallel_games"])]

        while len(spGames) > 0:
            states = np.stack([spg.state for spg in spGames])
            neutral_states = self.game.change_perspective(states, player)

            with torch.inference_mode():
                self.mcts.search(neutral_states, spGames)

            for i in range(len(spGames))[::-1]:
                spg = spGames[i]

                action_probs = np.zeros(self.game.action_size, dtype=np.float32)
                for child in spg.root.children:
                    action_probs[child.action_taken] = child.visit_count
                s = float(action_probs.sum())
                if s > 0:
                    action_probs /= s

                if self.oracle is None:
                    spg.memory.append((spg.root.state, action_probs, player))
                else:
                    spg.memory.append((spg.root.state, action_probs, player, spg.seq))

                temperature_action_probs = action_probs ** (1 / self.args["temperature"])
                temperature_action_probs /= temperature_action_probs.sum()
                action = np.random.choice(self.game.action_size, p=temperature_action_probs)

                if self.oracle is not None:
                    spg.seq += str(int(action) + 1)

                spg.state = self.game.get_next_state(spg.state, action, player)

                value, is_terminal = self.game.get_value_and_terminated(spg.state, action)

                if is_terminal:
                    if self.oracle is None:
                        for hist_neutral_state, hist_action_probs, hist_player in spg.memory:
                            hist_outcome = (
                                value
                                if hist_player == player
                                else self.game.get_opponent_value(value)
                            )
                            return_memory.append(
                                (
                                    self.game.get_encoded_state(hist_neutral_state),
                                    hist_action_probs,
                                    hist_outcome,
                                )
                            )
                    else:
                        for (
                            hist_neutral_state,
                            hist_action_probs,
                            hist_player,
                            hist_seq,
                        ) in spg.memory:
                            hist_outcome = (
                                value
                                if hist_player == player
                                else self.game.get_opponent_value(value)
                            )
                            return_memory.append(
                                (
                                    self.game.get_encoded_state(hist_neutral_state),
                                    hist_action_probs,
                                    hist_outcome,
                                    hist_seq,
                                )
                            )
                    del spGames[i]

            player = self.game.get_opponent(player)

        return return_memory

    def _attach_oracle_targets(self, memory):
        if len(memory) == 0:
            return memory

        if self.oracle is None:
            if self.require_oracle:
                raise RuntimeError(
                    "Perfect solver is required, but `oracle=None` so it cannot be used."
                )
            return memory

        if len(memory[0]) == 4 and not isinstance(memory[0][3], str):
            self._oracle_used_this_iteration = True
            return memory

        if len(memory[0]) != 4 or not isinstance(memory[0][3], str):
            raise RuntimeError(
                "Expected memory entries (encoded_state, policy, value, seq_string) when oracle is enabled. "
                "Did you forget to pass oracle=PerfectC4Oracle(game) into AlphaZeroParallel?"
            )

        seqs = [m[3] for m in memory]
        uniq = list(dict.fromkeys(seqs))

        batch_size = int(self.args.get("oracle_batch_size", 2048))
        uniq_scores = []
        for i in range(0, len(uniq), batch_size):
            uniq_scores.extend(self.oracle.analyze_batch(uniq[i : i + batch_size]))

        if len(uniq_scores) != len(uniq):
            raise RuntimeError(
                "Perfect solver returned an unexpected number of results. "
                f"Expected {len(uniq)}, got {len(uniq_scores)}."
            )

        seq_to_scores = dict(zip(uniq, uniq_scores))

        out = []
        for enc_state, pi, z, seq in memory:
            scores = np.asarray(seq_to_scores[seq], dtype=np.float32)
            if scores.shape != (self.game.action_size,):
                raise RuntimeError(
                    f"Perfect solver returned scores shape {scores.shape}, expected ({self.game.action_size},)."
                )
            out.append((enc_state, pi, z, scores))

        self._oracle_used_this_iteration = True
        self._oracle_total_positions = len(seqs)
        self._oracle_unique_positions = len(uniq)

        return out

    def train(self, memory):
        random.shuffle(memory)

        for batchIdx in range(0, len(memory), self.args["batch_size"]):
            sample = memory[batchIdx : batchIdx + self.args["batch_size"]]

            oracle_scores = None
            if len(sample[0]) == 3:
                state, policy_targets, value_targets = zip(*sample)
            elif len(sample[0]) == 4:
                state, policy_targets, value_targets, oracle_scores = zip(*sample)
                if isinstance(oracle_scores[0], str):
                    raise RuntimeError(
                        "Training memory contains seq strings, not oracle scores. "
                        "Did you forget to call _attach_oracle_targets() in learn()?"
                    )
            else:
                raise RuntimeError(f"Unexpected memory tuple length: {len(sample[0])}")

            state = np.array(state, dtype=np.float32)
            policy_targets = np.array(policy_targets, dtype=np.float32)
            value_targets = np.array(value_targets, dtype=np.float32).reshape(-1, 1)

            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(
                policy_targets, dtype=torch.float32, device=self.model.device
            )
            value_targets = torch.tensor(
                value_targets, dtype=torch.float32, device=self.model.device
            )

            out_policy, out_value = self.model(state)

            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)

            p_move_loss = torch.tensor(0.0, device=self.model.device)
            p_labeled_frac = 0.0

            if self.p_move_lambda > 0.0:
                if oracle_scores is None:
                    raise RuntimeError(
                        "p_move_lambda>0 but no oracle scores are present in `memory`. "
                        "This means the perfect solver is NOT being used."
                    )

                oracle_scores_t = torch.tensor(
                    np.stack(oracle_scores).astype(np.float32, copy=False),
                    dtype=torch.float32,
                    device=self.model.device,
                )

                valid = (state[:, 1, :, :].sum(dim=1) > 0.5).to(torch.float32)

                masked_scores = oracle_scores_t + (valid - 1.0) * 1e9
                best = masked_scores.max(dim=1, keepdim=True).values

                z = ((masked_scores == best) & (valid > 0.0)).to(torch.float32)
                q = (z + self.p_move_eps) * valid
                qsum = q.sum(dim=1, keepdim=True)

                q_mask = (qsum.squeeze(1) > 0.0).to(torch.float32)
                q_targets = q / (qsum + 1e-8)

                log_probs = F.log_softmax(out_policy, dim=1)
                per_sample_ce = -(q_targets * log_probs).sum(dim=1)

                denom = q_mask.sum() + 1e-8
                p_move_loss = (per_sample_ce * q_mask).sum() / denom
                p_labeled_frac = float(q_mask.mean().item())

                if self.require_oracle and p_labeled_frac <= 0.0:
                    raise RuntimeError(
                        "Perfect solver is required, but this batch has 0 labeled examples. "
                        "(This should never happen for Connect Four.)"
                    )

            loss = policy_loss + value_loss + self.p_move_lambda * p_move_loss

            if wandb is not None and batchIdx % self.args["log_freq"] == 0:
                wandb.log(
                    {
                        "policy_loss": float(policy_loss.item()),
                        "value_loss": float(value_loss.item()),
                        "p_move_loss": float(p_move_loss.item()),
                        "p_move_labeled_frac": float(p_labeled_frac),
                        "aux_loss": float(p_move_loss.item()),
                        "total_loss": float(loss.item()),
                    }
                )

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

    def learn(self):
        for iteration in range(self.args["num_iterations"]):
            memory = []

            self._oracle_used_this_iteration = False
            self._oracle_total_positions = 0
            self._oracle_unique_positions = 0

            self.model.eval()
            for _ in trange(
                self.args["num_selfPlay_iterations"] // self.args["num_parallel_games"]
            ):
                memory += self.selfPlay()

            if self.oracle is not None or self.require_oracle or self.p_move_lambda > 0.0:
                memory = self._attach_oracle_targets(memory)

            if self.require_oracle and not self._oracle_used_this_iteration:
                raise RuntimeError(
                    "Perfect solver is required, but it was never queried. "
                    "Double-check that you passed oracle=PerfectC4Oracle(game) into AlphaZeroParallel."
                )

            if wandb is not None and self._oracle_used_this_iteration:
                wandb.log(
                    {
                        "oracle_total_positions": int(self._oracle_total_positions),
                        "oracle_unique_positions": int(self._oracle_unique_positions),
                        "p_move_lambda": float(self.p_move_lambda),
                    }
                )

            self.model.train()
            for _ in trange(self.args["num_epochs"]):
                self.train(memory)

            torch.save(self.model.state_dict(), f"model_{iteration}_{self.game}.pt")
            torch.save(
                self.optimizer.state_dict(), f"optimizer_{iteration}_{self.game}.pt"
            )


class SPG:
    def __init__(self, game):
        self.state = game.get_initial_state()
        self.memory = []
        self.root = None
        self.node = None
        self.seq = ""
