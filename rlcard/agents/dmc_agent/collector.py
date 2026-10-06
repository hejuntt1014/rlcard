"""Batched decision inference and interruptible rollout workers."""
import copy
import random
import traceback
from collections import defaultdict

import numpy as np
import torch

from .utils import TrajectoryWriter


class EnvironmentFactory:
    """Picklable construction recipe, without live game state or module objects."""
    def __init__(self, entry_point, config):
        self.entry_point, self.config = entry_point, config

    def __call__(self):
        return self.entry_point(copy.deepcopy(self.config))


def environment_source(env):
    if hasattr(env, '_creation_config'):
        return EnvironmentFactory(env._creation_entry_point, env._creation_config)
    return env


def action_features(state, shape):
    keys = list(state['legal_actions'])
    features = []
    if not keys:
        raise ValueError('Nonterminal decision has no legal actions')
    for key in keys:
        value = state['legal_actions'][key]
        if value is None:
            value = np.zeros(shape, dtype=np.float32)
            value.reshape(-1)[int(key)] = 1
        features.append(np.asarray(value))
    return keys, np.stack(features)


def score_actions(agent, observations, actions, max_actions, model_lock):
    outputs = []
    # Protect readers against a partially copied parameter snapshot.
    with model_lock, torch.no_grad():
        for start in range(0, len(actions), max_actions):
            end = start + max_actions
            obs = torch.as_tensor(observations[start:end], device=agent.device).float()
            act = torch.as_tensor(actions[start:end], device=agent.device).float()
            outputs.append(agent.forward(obs, act))
        return torch.cat(outputs).cpu().numpy()


class RLCardAdapter:
    """Terminal-return DMC adapter; override episode_data for reward extensions."""
    def __init__(self, env):
        self.env = env
        self.num_players = env.num_players

    def reset(self, seed=None):
        if seed is not None:
            self.env.seed(seed)
        return self.env.reset()

    def is_over(self):
        return self.env.is_over()

    def step(self, action):
        return self.env.step(action)

    def rule_action(self, state, player):
        return False, None

    def episode_data(self):
        return self.env.get_payoffs(), None, None


class PettingZooAdapter:
    """AEC adapter using accumulated per-action rewards, including dead turns."""
    def __init__(self, env):
        self.env = env
        self.names = list(env.possible_agents)
        self.num_players = len(self.names)

    def reset(self, seed=None):
        self.env.reset(seed=seed)
        self.rewards = [[] for _ in self.names]
        return self._advance()

    def _advance(self):
        from rlcard.utils.pettingzoo_utils import wrap_state
        while self.env.agents:
            p = self.names.index(self.env.agent_selection)
            obs, reward, terminated, truncated, _ = self.env.last()
            if self.rewards[p]:
                self.rewards[p][-1] += float(reward)
            if terminated or truncated:
                self.env.step(None)
            else:
                return wrap_state(obs), p
        return None, None

    def is_over(self):
        return not self.env.agents

    def step(self, action):
        p = self.names.index(self.env.agent_selection)
        self.rewards[p].append(0.)
        self.env.step(action)
        return self._advance()

    def rule_action(self, state, player):
        return False, None

    def episode_data(self):
        return [0.] * self.num_players, self.rewards, None


class PythonPool:
    def __init__(self, env, count, seed, adapter_class, max_episode_steps):
        source = environment_source(env)
        self.envs = [adapter_class(source() if callable(source) else copy.deepcopy(source))
                     for _ in range(count)]
        self.players = self.envs[0].num_players
        self.states = [e.reset(seed + i) for i, e in enumerate(self.envs)]
        self.steps = [[[] for _ in range(self.players)] for _ in self.envs]
        self.lengths = [0] * count
        self.max_episode_steps = max_episode_steps

    def round(self, model, epsilon, max_actions, model_lock):
        groups = defaultdict(list)
        finished = []
        num_steps = 0
        for i, env in enumerate(self.envs):
            if env.is_over():
                finished.append((self.steps[i], *env.episode_data()))
                self.steps[i] = [[] for _ in range(self.players)]
                self.lengths[i] = 0
                self.states[i] = env.reset()
                continue
            state, p = self.states[i]
            is_rule, action = env.rule_action(state, p)
            keys, features = action_features(state, model.get_agent(p).action_shape)
            if is_rule:
                self._step(i, p, state['obs'], action, features[keys.index(action)])
                num_steps += 1
            else:
                agent = model.get_agent(p)
                groups[id(agent)].append((i, p, state['obs'], keys, features))

        for entries in groups.values():
            agent = model.get_agent(entries[0][1])
            observations = np.concatenate([np.repeat(obs[None], len(keys), axis=0)
                                           for _, _, obs, keys, _ in entries])
            actions = np.concatenate([features for _, _, _, _, features in entries])
            values = score_actions(agent, observations, actions, max_actions, model_lock)
            offset = 0
            for i, p, obs, keys, features in entries:
                count = len(keys)
                choice = (np.random.randint(count) if np.random.random() < epsilon
                          else int(values[offset:offset + count].argmax()))
                self._step(i, p, obs, keys[choice], features[choice])
                offset += count
                num_steps += 1

        for i, env in enumerate(self.envs):
            if env.is_over():
                finished.append((self.steps[i], *env.episode_data()))
                self.steps[i] = [[] for _ in range(self.players)]
                self.lengths[i] = 0
                self.states[i] = env.reset()
        return finished, num_steps

    def _step(self, i, p, obs, action, feature):
        self.steps[i][p].append((np.asarray(obs).copy(), feature.copy()))
        self.states[i] = self.envs[i].step(action)
        self.lengths[i] += 1
        if self.lengths[i] > self.max_episode_steps:
            raise RuntimeError('Episode exceeded max_episode_steps; no truncated targets were emitted')


def actor_worker(actor_id, seed, env, model, model_lock, buffers,
                 free_queues, full_queues, stop, epsilon, errors, counters,
                 T, count, backend, adapter_class, max_actions, max_episode_steps):
    try:
        torch.set_num_threads(1)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if backend == 'cpp':
            from rlcard.envs.a3dizhu.dmc import NativePool
            pool = NativePool(env() if callable(env) else env, count, seed, max_episode_steps)
        else:
            pool = PythonPool(env, count, seed, adapter_class, max_episode_steps)
        writer = TrajectoryWriter(T, free_queues, full_queues, buffers)
        while not stop.is_set():
            writer.flush()
            if writer.congested():
                stop.wait(.005)
                continue
            episodes, steps = pool.round(model, epsilon.value, max_actions, model_lock)
            for episode in episodes:
                writer.add_episode(*episode)
            with counters.get_lock():
                counters[0] += steps
                counters[1] += len(episodes)
    except BaseException:
        errors.put((actor_id, traceback.format_exc()))
        raise
    finally:
        # Shutdown may leave published slots unread. Do not wait for feeder drain.
        for q in full_queues:
            q.cancel_join_thread()
