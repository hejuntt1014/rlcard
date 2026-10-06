"""Optional A3 reward, rule-opponent and native batching adapters."""
import random
import importlib
import numpy as np

from rlcard.agents.dmc_agent.collector import RLCardAdapter, choose_action_indices


class A3Adapter(RLCardAdapter):
    def reset(self, seed=None):
        if seed is not None:
            self.env.seed(seed)
            if hasattr(self.env, 'game'):
                self.env.game.np_random = random.Random(seed)
        return self.env.reset()

    def rule_action(self, state, player):
        if hasattr(self.env, '_engine'):
            e = self.env._engine
            return ((True, e.get_rule_agent_action()) if e.is_rule_agent_seat(player)
                    else (False, None))
        if player in self.env._seat_agents:
            from .env import _hand_key
            hand = self.env._seat_agents[player].step(state['raw_obs'])
            return True, hand if isinstance(hand, str) else _hand_key(hand)
        return False, None

    def episode_data(self):
        e = self.env
        rewards = ([list(e._engine.get_step_rewards(p)) for p in range(e.num_players)]
                   if hasattr(e, '_engine') else e._step_rewards)
        if getattr(e, 'reward_mode', 'shaped') == 'game':
            rewards = [[0.] * len(r) for r in rewards]
        return e.get_training_payoffs(), rewards, None

    def decision_labels(self, player):
        return np.asarray(self.env.get_aux_targets()[player], dtype=np.int64)


class NativePool:
    def __init__(self, env, count, seed, max_episode_steps):
        module = importlib.import_module(getattr(env, 'native_module', 'a3dizhu_cpp'))
        self.vec = module.VectorizedEngine(count)
        self.context_features = getattr(env, 'name', '') == 'a3dizhu-v12'
        self.rules = getattr(env, 'rules', None)
        self.reward_mode = getattr(env, 'reward_mode', 'shaped')
        self.vec.seed(seed)
        self.vec.set_greedy_ratio(env.greedy_ratio)
        self.vec.set_random_ratio(env.random_ratio)
        self.count = count
        self.steps = [[[] for _ in range(4)] for _ in range(count)]
        self.max_episode_steps = max_episode_steps
        for i in range(count):
            self.reset(i)

    def reset(self, i):
        self.vec.reset(i)
        if self.rules is not None:
            from .v12 import RULE_KEYS
            self.vec.set_rules(i, *(self.rules[k] for k in RULE_KEYS))

    def round(self, model, epsilon, max_actions, model_lock):
        pending, finished = [], []
        num_steps = 0
        for i in range(self.count):
            if self.context_features:
                done, obs, pids, actions, auxiliary = self.vec.advance_to_decision_with_data(i)
            else:
                done, obs, pids, actions = self.vec.advance_to_decision_with_actions(i)
                # The V6 compatibility API does not expose pre-action rule labels.
                auxiliary = np.full((len(pids), 3), -1, dtype=np.int64)
            for row, p in enumerate(pids):
                self.steps[i][p].append((obs[row].copy(), actions[row].copy(), auxiliary[row].copy()))
            num_steps += len(pids)
            if sum(map(len, self.steps[i])) > self.max_episode_steps:
                raise RuntimeError('Episode exceeded max_episode_steps')
            if done:
                rewards = [list(self.vec.get_step_rewards(i, p)) for p in range(4)]
                if self.reward_mode == 'game':
                    payoffs = self.vec.get_payoffs(i)
                    rewards = [[0.] * len(r) for r in rewards]
                else:
                    payoffs = self.vec.get_training_payoffs(i)
                finished.append((self.steps[i], payoffs, rewards, None))
                self.steps[i] = [[] for _ in range(4)]
                self.reset(i)
            else:
                pending.append(i)
        if pending:
            # Group by policy, preserving independent role networks as needed.
            groups = {}
            for i in pending:
                p = self.vec.get_player_id(i)
                agent = model.get_agent(p)
                groups.setdefault(id(agent), (agent, []))[1].append(i)
            for agent, indices in groups.values():
                if self.context_features:
                    raw, actions, offsets, keys = self.vec.prepare_batch(indices)
                else:
                    _, actions, raw, offsets, keys = self.vec.prepare_batch(indices)
                choices = choose_action_indices(agent, raw, actions, offsets, epsilon, max_actions, model_lock)
                for row, i in enumerate(indices):
                    start, end = offsets[row:row + 2]
                    a = choices[row]
                    p = self.vec.get_player_id(i)
                    labels = (self.vec.get_aux_targets_for_player(i, p) if self.context_features
                              else self.vec.get_aux_targets(i)[p])
                    self.steps[i][p].append((raw[row].copy(), actions[start + a].copy(), np.asarray(labels).copy()))
                    self.vec.step(i, keys[row][a])
                    num_steps += 1
        return finished, num_steps
