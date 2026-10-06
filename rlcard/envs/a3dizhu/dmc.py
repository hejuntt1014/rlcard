"""Optional A3 reward, rule-opponent and native batching adapters."""
import random
import numpy as np

from rlcard.agents.dmc_agent.collector import RLCardAdapter, score_actions


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
        return e.get_training_payoffs(), rewards, e.get_aux_targets()


class NativePool:
    def __init__(self, env, count, seed, max_episode_steps):
        from a3dizhu_cpp import VectorizedEngine
        self.vec = VectorizedEngine(count)
        self.vec.seed(seed)
        self.vec.set_greedy_ratio(env.greedy_ratio)
        self.vec.set_random_ratio(env.random_ratio)
        self.count = count
        self.steps = [[[] for _ in range(4)] for _ in range(count)]
        self.max_episode_steps = max_episode_steps
        for i in range(count):
            self.vec.reset(i)

    def round(self, model, epsilon, max_actions, model_lock):
        pending, finished = [], []
        num_steps = 0
        for i in range(self.count):
            done, obs, pids, actions = self.vec.advance_to_decision_with_actions(i)
            for row, p in enumerate(pids):
                self.steps[i][p].append((obs[row].copy(), actions[row].copy()))
            num_steps += len(pids)
            if sum(map(len, self.steps[i])) > self.max_episode_steps:
                raise RuntimeError('Episode exceeded max_episode_steps')
            if done:
                finished.append((self.steps[i], self.vec.get_training_payoffs(i),
                                 [list(self.vec.get_step_rewards(i, p)) for p in range(4)],
                                 self.vec.get_aux_targets(i)))
                self.steps[i] = [[] for _ in range(4)]
                self.vec.reset(i)
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
                obs, actions, raw, offsets, keys = self.vec.prepare_batch(indices)
                values = score_actions(agent, obs, actions, max_actions, model_lock)
                for row, i in enumerate(indices):
                    start, end = offsets[row:row + 2]
                    a = (np.random.randint(end - start) if np.random.random() < epsilon
                         else int(values[start:end].argmax()))
                    p = self.vec.get_player_id(i)
                    self.steps[i][p].append((raw[row].copy(), actions[start + a].copy()))
                    self.vec.step(i, keys[row][a])
                    num_steps += 1
        return finished, num_steps
