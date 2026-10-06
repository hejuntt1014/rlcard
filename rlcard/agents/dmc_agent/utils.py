# Modifications copyright (c) 2026 hejuntt1014.
# Modified for batched training, optional backends, and runtime portability.
# Copyright 2021 RLCard Team of Texas A&M University
# Copyright 2021 DouZero Team of Kwai
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bounded CPU shared-memory transport for DMC trajectories."""
import logging
from collections import deque
from queue import Empty

import numpy as np
import torch

log = logging.getLogger(__name__)


def create_buffers(T, num_buffers, state_shape, action_shape, device_iterator,
                   aux_size=0, dtype=torch.float32):
    buffers = {}
    for device in device_iterator:
        buffers[device] = []
        for obs_shape, act_shape in zip(state_shape, action_shape):
            specs = {
                'done': ((T,), torch.bool),
                'episode_return': ((T,), torch.float32),
                'target': ((T,), torch.float32),
                'state': ((T,) + tuple(obs_shape), dtype),
                'action': ((T,) + tuple(act_shape), dtype),
            }
            if aux_size:
                specs['aux_target'] = ((T, aux_size), torch.int64)
            buffers[device].append({
                key: torch.empty((num_buffers,) + shape, dtype=kind).share_memory_()
                for key, (shape, kind) in specs.items()
            })
    return buffers


class BatchReader:
    """One consumer; hold partial batches locally instead of requeueing them."""
    def __init__(self, free_queue, full_queue, buffers, batch_size):
        self.free_queue = free_queue
        self.full_queue = full_queue
        self.buffers = buffers
        self.batch_size = batch_size
        self.pending = []

    def get(self):
        while len(self.pending) < self.batch_size:
            try:
                self.pending.append(self.full_queue.get_nowait())
            except Empty:
                return None
        indices = torch.tensor(self.pending, dtype=torch.long)
        batch = {key: value.index_select(0, indices).transpose(0, 1).contiguous()
                 for key, value in self.buffers.items()}
        for index in self.pending:
            self.free_queue.put(index)
        self.pending = []
        return batch


def _compute_targets_with_step_rewards(payoff, step_rewards):
    total = float(payoff)
    targets = [0.] * len(step_rewards)
    for i in range(len(step_rewards) - 1, -1, -1):
        total += float(step_rewards[i])
        targets[i] = total
    return targets


class TrajectoryWriter:
    """Flush whole unrolls without blocking one role behind another."""
    def __init__(self, T, free_queues, full_queues, buffers):
        self.T = T
        self.free_queues = free_queues
        self.full_queues = full_queues
        self.buffers = buffers
        self.pending = [{key: deque() for key in fields} for fields in buffers]

    def add_episode(self, steps, payoffs, rewards=None, auxiliary=None):
        for p, trajectory in enumerate(steps):
            n = len(trajectory)
            if not n:
                continue
            sr = [0.] * n if rewards is None else rewards[p]
            if len(sr) != n:
                raise ValueError('Step rewards and recorded actions must have identical lengths')
            out = self.pending[p]
            out['state'].extend(s[0] for s in trajectory)
            out['action'].extend(s[1] for s in trajectory)
            out['target'].extend(_compute_targets_with_step_rewards(payoffs[p], sr))
            out['done'].extend([False] * (n - 1) + [True])
            out['episode_return'].extend([0.] * (n - 1) + [float(payoffs[p]) + sum(sr)])
            if 'aux_target' in out:
                if auxiliary is None:
                    raise ValueError('Auxiliary task enabled but episode has no labels')
                out['aux_target'].extend([auxiliary[p]] * n)

    def flush(self):
        for p, data in enumerate(self.pending):
            while len(data['target']) >= self.T:
                try:
                    index = self.free_queues[p].get_nowait()
                except Empty:
                    break
                for key, values in data.items():
                    block = np.asarray([values.popleft() for _ in range(self.T)])
                    self.buffers[p][key][index].copy_(torch.as_tensor(block))
                self.full_queues[p].put(index)

    def congested(self):
        return any(len(data['target']) >= self.T * 2 for data in self.pending)


def create_optimizers(num_players, learning_rate, momentum, epsilon, alpha, learner_model):
    unique = {}
    optimizers = []
    for p in range(num_players):
        agent = learner_model.get_agent(p)
        if id(agent) not in unique:
            unique[id(agent)] = torch.optim.RMSprop(
                agent.parameters(), lr=learning_rate, momentum=momentum, eps=epsilon, alpha=alpha)
        optimizers.append(unique[id(agent)])
    return optimizers
