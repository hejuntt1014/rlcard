"""Batched decision inference and interruptible rollout workers."""
import copy
import contextlib
import random
import traceback
from collections import defaultdict

import numpy as np
import torch

from .utils import TrajectoryWriter, ColumnTrajectoryWriter


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
        return torch.cat(outputs).float().cpu().numpy()


def score_action_groups(agent, observations, actions, offsets, max_actions, model_lock):
    """Encode unique states once; bound candidate inputs/activations per forward."""
    offsets = np.asarray(offsets, dtype=np.int64)
    counts = np.diff(offsets)
    if len(offsets) != len(observations) + 1 or offsets[0] != 0 or offsets[-1] != len(actions) or np.any(counts <= 0):
        raise ValueError('Offsets must cover one nonempty candidate group per state')
    mapping = np.repeat(np.arange(len(observations)), counts)
    net = getattr(agent, 'inference_runner', agent.net)
    outputs = []
    dtype = getattr(agent, 'inference_dtype', None)
    amp = torch.autocast('cuda', dtype=dtype) if dtype is not None else contextlib.nullcontext()
    with model_lock, torch.no_grad(), amp:
        obs = torch.as_tensor(observations, device=agent.device).float()
        # Keep ordinary batches on-device once, but do not turn the forward
        # chunk limit into an unbounded allocation for large action spaces.
        all_actions = (torch.as_tensor(actions, device=agent.device).float()
                       if len(actions) <= max_actions else None)
        all_indices = (torch.as_tensor(mapping, device=agent.device, dtype=torch.long)
                       if all_actions is not None else None)
        reusable = hasattr(net, 'encode_state')
        encoded = net.encode_state(obs) if reusable else None
        # Known phase routing avoids CUDA nonzero synchronization in actor forwards.
        phase_column = getattr(net, 'declaration_flag_index', None)
        declaration = np.asarray(observations)[:, phase_column] > .5 if reusable and phase_column is not None else None
        for start in range(0, len(actions), max_actions):
            end = min(start + max_actions, len(actions))
            indices = mapping[start:end]
            index = (all_indices[start:end] if all_indices is not None else
                     torch.as_tensor(indices, device=agent.device, dtype=torch.long))
            act = (all_actions[start:end] if all_actions is not None else
                   torch.as_tensor(actions[start:end], device=agent.device).float())
            if not reusable:
                output = agent.forward(obs.index_select(0, index), act)
            elif declaration is None:
                output = net.score_encoded(encoded, act, index)
            elif not declaration[indices].any():
                output = net.score_encoded(encoded, act, index, phase='play')
            elif declaration[indices].all():
                output = net.score_encoded(encoded, act, index, phase='declare')
            else:
                output = torch.empty(end - start, device=agent.device, dtype=obs.dtype)
                for is_declare, phase in ((False, 'play'), (True, 'declare')):
                    rows = np.flatnonzero(declaration[indices] == is_declare)
                    if not len(rows):
                        continue
                    selected = torch.as_tensor(rows, device=agent.device, dtype=torch.long)
                    values = net.score_encoded(encoded, act.index_select(0, selected),
                                               index.index_select(0, selected), phase=phase)
                    output.index_copy_(0, selected, values.to(output.dtype))
            # Graph runners reuse output storage on the next chunk replay.
            outputs.append(output.clone() if hasattr(net, 'cache') and len(actions) > max_actions else output)
        return torch.cat(outputs).float().cpu().numpy()


def segmented_argmax(values, offsets):
    """First maximizing index in each nonempty group, matching NumPy NaNs."""
    groups = len(offsets) - 1
    # Tiny batches favor direct reductions; very wide action groups should not
    # materialize several candidate-sized arrays just to avoid a few Python calls.
    if groups < 32 or len(values) > groups * 64:
        return np.asarray([values[left:right].argmax()
                           for left, right in zip(offsets[:-1], offsets[1:])], dtype=np.int64)
    maxima = np.maximum.reduceat(values, offsets[:-1])
    expanded = np.repeat(maxima, np.diff(offsets))
    matches = (values == expanded) | (np.isnan(values) & np.isnan(expanded))
    candidates = np.where(matches, np.arange(len(values)), len(values))
    return np.minimum.reduceat(candidates, offsets[:-1]) - offsets[:-1]


def choose_action_indices(agent, observations, actions, offsets, epsilon, max_actions, model_lock):
    """Forced and exploratory decisions do not need a value-network forward."""
    offsets = np.asarray(offsets, dtype=np.int64)
    counts = np.diff(offsets)
    if len(offsets) != len(observations) + 1 or offsets[0] != 0 or offsets[-1] != len(actions):
        raise ValueError('Offsets do not match observations and actions')
    choices = np.empty(len(counts), dtype=np.int64)
    pending = []
    for i, count in enumerate(counts):
        if count <= 0:
            raise ValueError('Each decision must have a legal action')
        if count == 1:
            choices[i] = 0
        elif np.random.random() < epsilon:
            choices[i] = np.random.randint(count)
        else:
            pending.append(i)
    if pending:
        if len(pending) == len(counts):
            selected_actions, selected_offsets = actions, offsets
            selected_observations = observations
        else:
            selected_actions = np.concatenate([actions[offsets[i]:offsets[i + 1]] for i in pending])
            selected_offsets = np.cumsum([0] + [int(counts[i]) for i in pending])
            selected_observations = np.asarray(observations)[pending]
        values = score_action_groups(agent, selected_observations, selected_actions,
                                     selected_offsets, max_actions, model_lock)
        choices[pending] = segmented_argmax(values, selected_offsets)
    return choices


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

    def decision_labels(self, player):
        return None


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

    def decision_labels(self, player):
        return None


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
            observations = np.stack([obs for _, _, obs, _, _ in entries])
            actions = np.concatenate([features for _, _, _, _, features in entries])
            offsets = np.cumsum([0] + [len(keys) for _, _, _, keys, _ in entries])
            choices = choose_action_indices(agent, observations, actions, offsets, epsilon, max_actions, model_lock)
            for (i, p, obs, keys, features), choice in zip(entries, choices):
                self._step(i, p, obs, keys[choice], features[choice])
                num_steps += 1

        for i, env in enumerate(self.envs):
            if env.is_over():
                finished.append((self.steps[i], *env.episode_data()))
                self.steps[i] = [[] for _ in range(self.players)]
                self.lengths[i] = 0
                self.states[i] = env.reset()
        return finished, num_steps

    def _step(self, i, p, obs, action, feature):
        labels = self.envs[i].decision_labels(p)
        record = (np.asarray(obs).copy(), feature.copy())
        self.steps[i][p].append(record if labels is None else record + (np.asarray(labels).copy(),))
        self.states[i] = self.envs[i].step(action)
        self.lengths[i] += 1
        if self.lengths[i] > self.max_episode_steps:
            raise RuntimeError('Episode exceeded max_episode_steps; no truncated targets were emitted')


def actor_worker(actor_id, seed, env, model, model_lock, buffers,
                 free_queues, full_queues, stop, epsilon, errors, counters,
                 T, count, backend, adapter_class, max_actions, max_episode_steps,
                 inference_device='cpu', policy_version=None, precision='fp32',
                 actor_cuda_graphs=False, actor_half_weights=False, actor_poll_interval=.005):
    try:
        torch.set_num_threads(1)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        shared_model = model
        # All workers own their inference model. CPU actors can infer concurrently.
        with model_lock:
            model = copy.deepcopy(shared_model)
            local_version = policy_version.value
        inference_lock = contextlib.nullcontext()
        if inference_device != 'cpu':
            for agent in {id(a): a for a in model.get_agents()}.values():
                agent.device = 'cuda:' + str(inference_device)
                agent.net.to(agent.device)
                agent.inference_dtype = (torch.bfloat16 if precision == 'bf16'
                                         else torch.float16 if precision == 'fp16' else None)
                if actor_half_weights:
                    for layer in agent.net.modules():
                        if isinstance(layer, torch.nn.Linear):
                            layer.to(dtype=agent.inference_dtype)
                if actor_cuda_graphs:
                    from .context_model import ContextInferenceGraphs
                    agent.inference_runner = ContextInferenceGraphs(agent.net, dtype=agent.inference_dtype)
        if backend == 'cpp':
            from rlcard.envs.a3dizhu.dmc import NativePool
            pool = NativePool(env() if callable(env) else env, count, seed, max_episode_steps)
        else:
            pool = PythonPool(env, count, seed, adapter_class, max_episode_steps)
        columnar = getattr(pool, 'columnar', False)
        writer_type = ColumnTrajectoryWriter if columnar else TrajectoryWriter
        writer = writer_type(T, free_queues, full_queues, buffers)
        while not stop.is_set():
            if policy_version.value != local_version:
                with model_lock:
                    for p in ([0] if model.shared else range(len(model.get_agents()))):
                        model.get_agent(p).load_state_dict(shared_model.get_agent(p).state_dict())
                    local_version = policy_version.value
            writer.flush()
            if writer.congested():
                stop.wait(actor_poll_interval)
                continue
            episodes, steps = pool.round(model, epsilon.value, max_actions, inference_lock)
            for episode in episodes:
                if columnar:
                    writer.add_episode(episode)
                else:
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
